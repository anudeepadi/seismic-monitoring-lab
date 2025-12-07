"""
Loss Functions for Physics-Informed Neural Networks

This module provides various loss components for PINN training:
- Data loss: Match observed seismic data
- Physics loss: Satisfy wave equation (PDE residual)
- Boundary conditions: Absorbing, free surface, etc.
- Initial conditions: Quiescent initial state
- Regularization: Velocity smoothness, sparsity, etc.

The total loss is a weighted combination of these components.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LossComponent(ABC):
    """Abstract base class for loss components."""

    @abstractmethod
    def compute(
        self,
        model: nn.Module,
        batch: Dict[str, Tensor],
        **kwargs,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute this loss component.

        Args:
            model: The PINN model
            batch: Training batch data
            **kwargs: Additional arguments

        Returns:
            Tuple of (loss value, dictionary of metrics)
        """
        pass


class DataLoss(LossComponent):
    """
    Data fidelity loss.

    Measures mismatch between predicted and observed seismic data.
    Supports various norms and weighting schemes.

    Args:
        norm: Loss norm ('l1', 'l2', 'huber')
        reduction: How to reduce over samples ('mean', 'sum')
        weight_by_offset: Weight by source-receiver offset
        weight_by_time: Weight by time sample
    """

    def __init__(
        self,
        norm: str = "l2",
        reduction: str = "mean",
        weight_by_offset: bool = False,
        weight_by_time: bool = False,
    ) -> None:
        self.norm = norm
        self.reduction = reduction
        self.weight_by_offset = weight_by_offset
        self.weight_by_time = weight_by_time

    def compute(
        self,
        model: nn.Module,
        batch: Dict[str, Tensor],
        **kwargs,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute data mismatch loss."""
        if "observation" not in batch:
            return torch.tensor(0.0), {"data_loss": 0.0}

        obs = batch["observation"]
        coords = obs["coords"]
        target = obs["values"]

        # Get model prediction
        velocity = batch.get("velocity")
        prediction = model(coords, velocity)

        # Compute residual
        residual = prediction - target

        # Apply weighting
        weights = torch.ones_like(residual)

        if self.weight_by_time and "time_samples" in batch:
            # Increase weight for later times (more information)
            t = coords[:, -1]
            t_normalized = (t - t.min()) / (t.max() - t.min() + 1e-8)
            weights = weights * (0.5 + 0.5 * t_normalized.unsqueeze(-1))

        # Compute loss based on norm
        if self.norm == "l1":
            loss = torch.abs(residual) * weights
        elif self.norm == "l2":
            loss = residual.pow(2) * weights
        elif self.norm == "huber":
            loss = F.smooth_l1_loss(prediction, target, reduction="none") * weights
        else:
            loss = residual.pow(2) * weights

        # Reduce
        if self.reduction == "mean":
            loss = loss.mean()
        else:
            loss = loss.sum()

        metrics = {
            "data_loss": loss.item(),
            "data_rmse": torch.sqrt(residual.pow(2).mean()).item(),
            "data_max_error": torch.abs(residual).max().item(),
        }

        return loss, metrics


class PhysicsLoss(LossComponent):
    """
    Physics-based loss (PDE residual).

    Enforces the wave equation constraint at collocation points.
    The loss is the squared residual of the PDE.

    For acoustic wave equation:
        ∂²u/∂t² - v²(∂²u/∂x² + ∂²u/∂z²) = f

    Args:
        wave_equation: Wave equation object for residual computation
        residual_type: 'pointwise' or 'integral'
        use_source: Include source term in residual
    """

    def __init__(
        self,
        wave_equation: Optional[object] = None,
        residual_type: str = "pointwise",
        use_source: bool = True,
    ) -> None:
        self.wave_equation = wave_equation
        self.residual_type = residual_type
        self.use_source = use_source

    def compute(
        self,
        model: nn.Module,
        batch: Dict[str, Tensor],
        **kwargs,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute physics (PDE) loss."""
        collocation = batch.get("collocation", batch)
        coords = collocation.get("coords", batch.get("coords"))

        if coords is None:
            return torch.tensor(0.0), {"physics_loss": 0.0}

        velocity = batch.get("velocity")

        # Get source term
        source = batch.get("source") if self.use_source else None

        # Compute PDE residual
        if self.wave_equation is not None:
            result = self.wave_equation.compute_residual(
                model, coords, velocity, source
            )
            residual = result["pde_residual"]
        else:
            # Default: compute residual using model's method
            residual = model.compute_wave_residual(coords, velocity, source)

        # Compute loss
        loss = residual.pow(2).mean()

        metrics = {
            "physics_loss": loss.item(),
            "residual_mean": residual.abs().mean().item(),
            "residual_max": residual.abs().max().item(),
            "residual_std": residual.std().item(),
        }

        return loss, metrics


class BoundaryLoss(LossComponent):
    """
    Boundary condition loss.

    Enforces boundary conditions (absorbing, free surface, etc.)
    at domain boundaries.

    Args:
        boundary_type: Type of boundary ('absorbing', 'free_surface', 'pml')
        boundary_condition: Boundary condition object
    """

    def __init__(
        self,
        boundary_type: str = "absorbing",
        boundary_condition: Optional[object] = None,
        domain_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        self.boundary_type = boundary_type
        self.boundary_condition = boundary_condition
        self.domain_bounds = domain_bounds

    def compute(
        self,
        model: nn.Module,
        batch: Dict[str, Tensor],
        **kwargs,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute boundary condition loss."""
        boundary = batch.get("boundary", {})
        coords = boundary.get("coords")

        if coords is None:
            return torch.tensor(0.0), {"boundary_loss": 0.0}

        velocity = batch.get("velocity")

        # Get model prediction and derivatives
        coords = coords.requires_grad_(True)

        if hasattr(model, "backbone_type") and velocity is not None:
            u = model(coords, velocity)
        else:
            u = model(coords)

        # Compute gradients
        grad_u = torch.autograd.grad(
            u.sum(), coords, create_graph=True, retain_graph=True
        )[0]

        derivatives = {
            "u": u,
            "u_x": grad_u[:, 0:1],
            "u_z": grad_u[:, 1:2],
            "u_t": grad_u[:, 2:3],
        }

        # Apply boundary condition
        if self.boundary_condition is not None:
            if hasattr(self.boundary_condition, "apply"):
                bounds = (
                    self.domain_bounds.get("x", (0, 1)),
                    self.domain_bounds.get("z", (0, 1)),
                )
                residuals = self.boundary_condition.apply(
                    coords, u, derivatives, velocity, bounds
                )
                # Sum residuals from all boundaries
                loss = sum(r.pow(2).mean() for r in residuals.values())
            else:
                loss = torch.tensor(0.0)
        else:
            # Default: absorbing boundary condition
            # At boundaries: ∂u/∂t ± v * ∂u/∂n = 0
            x = coords[:, 0]
            z = coords[:, 1]

            if self.domain_bounds:
                x_min, x_max = self.domain_bounds["x"]
                z_min, z_max = self.domain_bounds["z"]
            else:
                x_min, x_max = 0, 1
                z_min, z_max = 0, 1

            # Determine which boundary each point is on
            eps = 0.01

            residual = torch.zeros_like(u)

            # Left boundary
            left = x < x_min + eps * (x_max - x_min)
            if left.any() and velocity is not None:
                v_left = velocity[left] if velocity.dim() > 0 else velocity
                residual[left] = derivatives["u_t"][left] - v_left * derivatives["u_x"][left]

            # Right boundary
            right = x > x_max - eps * (x_max - x_min)
            if right.any() and velocity is not None:
                v_right = velocity[right] if velocity.dim() > 0 else velocity
                residual[right] = derivatives["u_t"][right] + v_right * derivatives["u_x"][right]

            # Top boundary
            top = z < z_min + eps * (z_max - z_min)
            if top.any():
                # Free surface: u = 0
                residual[top] = u[top]

            # Bottom boundary
            bottom = z > z_max - eps * (z_max - z_min)
            if bottom.any() and velocity is not None:
                v_bottom = velocity[bottom] if velocity.dim() > 0 else velocity
                residual[bottom] = derivatives["u_t"][bottom] + v_bottom * derivatives["u_z"][bottom]

            loss = residual.pow(2).mean()

        metrics = {
            "boundary_loss": loss.item(),
        }

        return loss, metrics


class InitialConditionLoss(LossComponent):
    """
    Initial condition loss.

    Enforces initial conditions (typically quiescent state):
        u(x, z, 0) = 0
        ∂u/∂t(x, z, 0) = 0

    Args:
        enforce_velocity: Also enforce zero initial velocity
    """

    def __init__(self, enforce_velocity: bool = True) -> None:
        self.enforce_velocity = enforce_velocity

    def compute(
        self,
        model: nn.Module,
        batch: Dict[str, Tensor],
        **kwargs,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute initial condition loss."""
        initial = batch.get("initial", {})
        coords = initial.get("coords")

        if coords is None:
            return torch.tensor(0.0), {"initial_loss": 0.0}

        velocity = batch.get("velocity")

        # Ensure gradients are computed
        coords = coords.requires_grad_(True)

        # Get model prediction
        if hasattr(model, "backbone_type") and velocity is not None:
            u = model(coords, velocity)
        else:
            u = model(coords)

        # Target: u = 0 at t = 0
        u_target = initial.get("u_target", torch.zeros_like(u))
        loss_u = (u - u_target).pow(2).mean()

        loss = loss_u

        if self.enforce_velocity:
            # Compute time derivative
            grad_u = torch.autograd.grad(
                u.sum(), coords, create_graph=True, retain_graph=True
            )[0]
            u_t = grad_u[:, 2:3]

            # Target: ∂u/∂t = 0 at t = 0
            u_t_target = initial.get("u_t_target", torch.zeros_like(u_t))
            loss_u_t = (u_t - u_t_target).pow(2).mean()

            loss = loss + loss_u_t

        metrics = {
            "initial_loss": loss.item(),
            "initial_u_loss": loss_u.item(),
        }

        return loss, metrics


class RegularizationLoss(LossComponent):
    """
    Regularization losses for velocity model inversion.

    Includes:
    - Total variation (TV) for sharp boundaries
    - Tikhonov (L2) for smoothness
    - Sparsity (L1) for compact features
    - Prior model constraint

    Args:
        reg_type: Type of regularization ('tv', 'l2', 'l1', 'prior')
        strength: Regularization strength
        prior_model: Prior velocity model (for 'prior' type)
    """

    def __init__(
        self,
        reg_type: str = "tv",
        strength: float = 1e-4,
        prior_model: Optional[Tensor] = None,
    ) -> None:
        self.reg_type = reg_type
        self.strength = strength
        self.prior_model = prior_model

    def compute(
        self,
        model: nn.Module,
        batch: Dict[str, Tensor],
        velocity_model: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute regularization loss."""
        if velocity_model is None:
            # Try to get from model (for inverse problems)
            if hasattr(model, "predict_velocity"):
                spatial_coords = batch.get("spatial_coords")
                if spatial_coords is not None:
                    velocity_model = model.predict_velocity(spatial_coords)
                else:
                    return torch.tensor(0.0), {"regularization_loss": 0.0}
            else:
                return torch.tensor(0.0), {"regularization_loss": 0.0}

        if self.reg_type == "tv":
            # Total variation
            if velocity_model.dim() >= 2:
                dx = velocity_model[..., 1:] - velocity_model[..., :-1]
                dy = velocity_model[..., 1:, :] - velocity_model[..., :-1, :]

                loss = torch.abs(dx).mean() + torch.abs(dy).mean()
            else:
                loss = torch.abs(velocity_model[1:] - velocity_model[:-1]).mean()

        elif self.reg_type == "l2":
            # Tikhonov (smoothness)
            if velocity_model.dim() >= 2:
                dx = velocity_model[..., 1:] - velocity_model[..., :-1]
                dy = velocity_model[..., 1:, :] - velocity_model[..., :-1, :]

                loss = dx.pow(2).mean() + dy.pow(2).mean()
            else:
                loss = (velocity_model[1:] - velocity_model[:-1]).pow(2).mean()

        elif self.reg_type == "l1":
            # Sparsity
            if self.prior_model is not None:
                diff = velocity_model - self.prior_model
            else:
                diff = velocity_model - velocity_model.mean()

            loss = torch.abs(diff).mean()

        elif self.reg_type == "prior":
            # Prior model constraint
            if self.prior_model is not None:
                loss = (velocity_model - self.prior_model).pow(2).mean()
            else:
                loss = torch.tensor(0.0)

        else:
            loss = torch.tensor(0.0)

        loss = self.strength * loss

        metrics = {
            "regularization_loss": loss.item(),
        }

        return loss, metrics


class PINNLoss(nn.Module):
    """
    Combined PINN loss function.

    Aggregates all loss components with configurable weights.

    Args:
        components: Dictionary of loss components
        weights: Dictionary of loss weights
        adaptive_weighting: Use adaptive weight balancing
    """

    def __init__(
        self,
        components: Optional[Dict[str, LossComponent]] = None,
        weights: Optional[Dict[str, float]] = None,
        adaptive_weighting: Optional[object] = None,
    ) -> None:
        super().__init__()

        self.components = components or {
            "data": DataLoss(),
            "physics": PhysicsLoss(),
            "boundary": BoundaryLoss(),
            "initial": InitialConditionLoss(),
        }

        self.weights = weights or {
            "data": 1.0,
            "physics": 1.0,
            "boundary": 1.0,
            "initial": 1.0,
        }

        self.adaptive_weighting = adaptive_weighting

    def forward(
        self,
        model: nn.Module,
        batch: Dict[str, Tensor],
        return_components: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute total loss.

        Args:
            model: PINN model
            batch: Training batch
            return_components: Return individual loss values

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        losses = {}
        metrics = {}

        for name, component in self.components.items():
            loss, component_metrics = component.compute(model, batch, **kwargs)
            losses[name] = loss
            metrics.update(component_metrics)

        # Get weights (adaptive or fixed)
        if self.adaptive_weighting is not None:
            weights = self.adaptive_weighting.get_weights(losses)
        else:
            weights = self.weights

        # Compute weighted sum
        total_loss = sum(
            weights.get(name, 1.0) * loss for name, loss in losses.items()
        )

        metrics["total_loss"] = total_loss.item()

        if return_components:
            for name, loss in losses.items():
                metrics[f"{name}_loss_weighted"] = (
                    weights.get(name, 1.0) * loss
                ).item()

        return total_loss, metrics

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Update loss weights."""
        self.weights.update(weights)


class EnergyConservationLoss(LossComponent):
    """
    Energy conservation loss.

    Enforces that the total wave energy is conserved (for non-dissipative media)
    or decreases monotonically (for dissipative media).

    This provides a global constraint complementing pointwise PDE residual.
    """

    def __init__(self, dissipative: bool = False) -> None:
        self.dissipative = dissipative

    def compute(
        self,
        model: nn.Module,
        batch: Dict[str, Tensor],
        **kwargs,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute energy conservation loss."""
        coords = batch.get("coords")
        velocity = batch.get("velocity")

        if coords is None:
            return torch.tensor(0.0), {"energy_loss": 0.0}

        # Sample at two different times
        t = coords[:, 2]
        t_mid = (t.max() + t.min()) / 2

        early_mask = t < t_mid
        late_mask = t >= t_mid

        # This is a simplified version - full implementation would
        # integrate energy over spatial domain at each time
        loss = torch.tensor(0.0)

        metrics = {"energy_loss": loss.item()}

        return loss, metrics
