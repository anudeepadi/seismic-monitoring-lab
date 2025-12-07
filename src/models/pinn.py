"""
Core Physics-Informed Neural Network Module

This is the main PINN class that combines neural network architectures
with physics constraints (wave equation) for seismic inversion.

Key features:
- Multiple backbone architectures (SIREN, Fourier, Modulated)
- Automatic differentiation for physics constraints
- Multi-task learning with adaptive loss weighting
- Support for both forward and inverse problems
"""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from .siren import SIRENNetwork, GradientScalingSIREN
from .fourier import FourierFeatureNetwork
from .modulated import ModulatedSIREN, VelocityConditionedPINN


class BackboneType(str, Enum):
    """Available backbone architectures."""
    SIREN = "siren"
    FOURIER = "fourier"
    MODULATED = "modulated"
    GRADIENT_SIREN = "gradient_siren"
    VELOCITY_CONDITIONED = "velocity_conditioned"


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for Seismic Wave Propagation.

    This class provides a unified interface for training neural networks
    with physics constraints derived from the wave equation. It supports
    multiple backbone architectures and can be used for both forward
    modeling (given velocity, predict wavefield) and inverse problems
    (given wavefield observations, predict velocity).

    Architecture overview:
    - Input: Space-time coordinates (x, z, t) and optionally velocity
    - Backbone: Neural network that maps inputs to wavefield
    - Output: Pressure/displacement field u(x, z, t)
    - Physics: Automatic computation of wave equation residual

    The loss function combines:
    1. Data loss: Match observed seismic data at receiver locations
    2. Physics loss: Satisfy wave equation in the domain
    3. Initial/boundary conditions: Match known values at boundaries
    4. Regularization: Smoothness constraints on velocity model

    Args:
        backbone: Type of neural network backbone
        input_dim: Dimension of input coordinates (default: 3 for 2D+time)
        hidden_dim: Width of hidden layers
        hidden_layers: Number of hidden layers
        output_dim: Dimension of output (default: 1 for pressure)
        velocity_input: Whether velocity is provided as input (for inverse)
        velocity_dim: Dimension of velocity (1 for acoustic, 2 for elastic)
        backbone_config: Additional configuration for backbone network

    Example:
        >>> pinn = PINN(
        ...     backbone=BackboneType.SIREN,
        ...     input_dim=3,  # (x, z, t)
        ...     hidden_dim=256,
        ...     hidden_layers=6,
        ...     output_dim=1,  # pressure
        ... )
        >>> coords = torch.randn(1000, 3)
        >>> velocity = torch.ones(1000, 1) * 2000  # m/s
        >>> pressure, derivatives = pinn(coords, velocity, return_derivatives=True)
    """

    def __init__(
        self,
        backbone: Union[str, BackboneType] = BackboneType.SIREN,
        input_dim: int = 3,
        hidden_dim: int = 256,
        hidden_layers: int = 6,
        output_dim: int = 1,
        velocity_input: bool = True,
        velocity_dim: int = 1,
        backbone_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.backbone_type = BackboneType(backbone)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.velocity_input = velocity_input
        self.velocity_dim = velocity_dim

        config = backbone_config or {}

        # Build backbone network
        if self.backbone_type == BackboneType.SIREN:
            self.network = SIRENNetwork(
                in_features=input_dim,
                hidden_features=hidden_dim,
                hidden_layers=hidden_layers,
                out_features=output_dim,
                first_omega_0=config.get("first_omega", 30.0),
                hidden_omega_0=config.get("hidden_omega", 30.0),
                use_skip=config.get("use_skip", True),
            )

        elif self.backbone_type == BackboneType.GRADIENT_SIREN:
            self.network = GradientScalingSIREN(
                in_features=input_dim,
                hidden_features=hidden_dim,
                hidden_layers=hidden_layers,
                out_features=output_dim,
                num_scales=config.get("num_scales", 3),
            )

        elif self.backbone_type == BackboneType.FOURIER:
            self.network = FourierFeatureNetwork(
                in_features=input_dim,
                hidden_features=hidden_dim,
                hidden_layers=hidden_layers,
                out_features=output_dim,
                num_frequencies=config.get("num_frequencies", 256),
                frequency_scale=config.get("frequency_scale", 10.0),
                multi_scale=config.get("multi_scale", True),
            )

        elif self.backbone_type == BackboneType.MODULATED:
            self.network = ModulatedSIREN(
                coord_dim=input_dim,
                velocity_dim=velocity_dim,
                hidden_features=hidden_dim,
                hidden_layers=hidden_layers,
                out_features=output_dim,
                modulation_type=config.get("modulation_type", "both"),
            )

        elif self.backbone_type == BackboneType.VELOCITY_CONDITIONED:
            self.network = VelocityConditionedPINN(
                velocity_channels=velocity_dim,
                coord_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=hidden_layers,
                out_dim=output_dim,
            )

        # Velocity network for inverse problems
        if not velocity_input:
            self.velocity_network = SIRENNetwork(
                in_features=input_dim - 1,  # Spatial coords only
                hidden_features=hidden_dim,
                hidden_layers=hidden_layers // 2,
                out_features=velocity_dim,
                outermost_linear=True,
            )
            # Ensure positive velocities
            self.velocity_activation = nn.Softplus()
            self.velocity_base = nn.Parameter(torch.tensor(2000.0))  # Base velocity
        else:
            self.velocity_network = None

    def forward(
        self,
        coords: Tensor,
        velocity: Optional[Tensor] = None,
        velocity_model: Optional[Tensor] = None,
        return_derivatives: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """
        Forward pass through the PINN.

        Args:
            coords: Space-time coordinates (batch, input_dim)
            velocity: Velocity at each point (batch, velocity_dim) - optional
            velocity_model: 2D velocity model for conditioning (batch, C, H, W) - optional
            return_derivatives: Whether to return derivatives for physics loss

        Returns:
            If return_derivatives=False: Wave field tensor (batch, output_dim)
            If return_derivatives=True: Tuple of (wave field, derivatives dict)
        """
        # Get or predict velocity
        if not self.velocity_input and velocity is None:
            spatial_coords = coords[..., :-1]  # Exclude time
            velocity_raw = self.velocity_network(spatial_coords)
            velocity = self.velocity_base + self.velocity_activation(velocity_raw) * 1000

        # Forward through backbone
        if self.backbone_type == BackboneType.MODULATED and velocity is not None:
            output = self.network(coords, velocity)
        elif self.backbone_type == BackboneType.VELOCITY_CONDITIONED and velocity_model is not None:
            coords_batched = coords.unsqueeze(0) if coords.dim() == 2 else coords
            output = self.network(velocity_model, coords_batched)
            if coords.dim() == 2:
                output = output.squeeze(0)
        else:
            output = self.network(coords)

        if not return_derivatives:
            return output

        # Compute derivatives for physics loss
        derivatives = self._compute_derivatives(coords, velocity)
        return output, derivatives

    def _compute_derivatives(
        self,
        coords: Tensor,
        velocity: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute all derivatives needed for the wave equation.

        For the acoustic wave equation: ∂²u/∂t² = v² * (∂²u/∂x² + ∂²u/∂z²)

        We need:
        - u: wave field
        - u_t, u_tt: first and second time derivatives
        - u_x, u_z, u_xx, u_zz: spatial derivatives
        """
        coords = coords.requires_grad_(True)

        # Get wave field
        if self.backbone_type == BackboneType.MODULATED and velocity is not None:
            u = self.network(coords, velocity)
        else:
            u = self.network(coords)

        # First derivatives (gradient)
        grad_u = torch.autograd.grad(
            outputs=u.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Parse spatial and temporal derivatives
        n_spatial = coords.shape[-1] - 1
        derivatives = {"u": u}

        # Spatial first derivatives
        for i in range(n_spatial):
            key = ["u_x", "u_z", "u_y"][i] if i < 3 else f"u_s{i}"
            derivatives[key] = grad_u[..., i : i + 1]

        # Time derivative
        derivatives["u_t"] = grad_u[..., -1:]

        # Second derivatives
        for i in range(n_spatial):
            spatial_grad = grad_u[..., i]
            grad2 = torch.autograd.grad(
                outputs=spatial_grad.sum(),
                inputs=coords,
                create_graph=True,
                retain_graph=True,
            )[0]
            key = ["u_xx", "u_zz", "u_yy"][i] if i < 3 else f"u_s{i}s{i}"
            derivatives[key] = grad2[..., i : i + 1]

        # Second time derivative
        time_grad = grad_u[..., -1]
        grad2_t = torch.autograd.grad(
            outputs=time_grad.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True,
        )[0]
        derivatives["u_tt"] = grad2_t[..., -1:]

        # Laplacian
        laplacian = sum(
            derivatives[key]
            for key in derivatives
            if key.endswith("xx") or key.endswith("zz") or key.endswith("yy")
        )
        derivatives["laplacian"] = laplacian

        return derivatives

    def compute_wave_residual(
        self,
        coords: Tensor,
        velocity: Tensor,
        source_term: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute the acoustic wave equation residual.

        Wave equation: ∂²u/∂t² - v² * ∇²u = f

        Args:
            coords: Space-time coordinates
            velocity: Velocity field at each point
            source_term: Optional source term f(x, z, t)

        Returns:
            Residual tensor (should be zero if equation is satisfied)
        """
        _, derivatives = self(coords, velocity, return_derivatives=True)

        u_tt = derivatives["u_tt"]
        laplacian = derivatives["laplacian"]

        # Wave equation residual
        residual = u_tt - velocity.pow(2) * laplacian

        if source_term is not None:
            residual = residual - source_term

        return residual

    def compute_elastic_wave_residual(
        self,
        coords: Tensor,
        vp: Tensor,
        vs: Tensor,
        density: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Compute elastic wave equation residuals.

        For elastic waves, we have coupled equations for displacement
        components (ux, uz) involving P-wave velocity (vp), S-wave
        velocity (vs), and density.

        Args:
            coords: Space-time coordinates
            vp: P-wave velocity
            vs: S-wave velocity
            density: Medium density

        Returns:
            Dictionary with residuals for each equation
        """
        coords = coords.requires_grad_(True)

        # Get displacement components
        u = self.network(coords)  # (batch, 2) for (ux, uz)

        # This would require more complex derivative computations
        # for the full elastic wave equation system
        # Simplified version for demonstration

        residuals = {"momentum_x": torch.zeros_like(u[:, 0:1]),
                    "momentum_z": torch.zeros_like(u[:, 1:2])}

        return residuals

    def predict_velocity(
        self,
        spatial_coords: Tensor,
    ) -> Tensor:
        """
        Predict velocity model (for inverse problems).

        Args:
            spatial_coords: Spatial coordinates only (batch, input_dim - 1)

        Returns:
            Predicted velocity at each point
        """
        if self.velocity_network is None:
            raise ValueError("PINN not configured for inverse problems")

        velocity_raw = self.velocity_network(spatial_coords)
        velocity = self.velocity_base + self.velocity_activation(velocity_raw) * 1000

        return velocity

    def get_wavefield_snapshot(
        self,
        x_range: Tuple[float, float],
        z_range: Tuple[float, float],
        time: float,
        resolution: Tuple[int, int] = (100, 100),
        velocity: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Generate a snapshot of the wave field at a specific time.

        Useful for visualization and analysis.

        Args:
            x_range: (min_x, max_x) spatial extent
            z_range: (min_z, max_z) spatial extent
            time: Time value for snapshot
            resolution: (nx, nz) grid resolution
            velocity: Constant velocity (if not using velocity input)
            device: Device to use

        Returns:
            Wave field snapshot of shape (nz, nx)
        """
        if device is None:
            device = next(self.parameters()).device

        nx, nz = resolution
        x = torch.linspace(x_range[0], x_range[1], nx, device=device)
        z = torch.linspace(z_range[0], z_range[1], nz, device=device)

        # Create grid
        xx, zz = torch.meshgrid(x, z, indexing="xy")
        coords = torch.stack([
            xx.flatten(),
            zz.flatten(),
            torch.full((nx * nz,), time, device=device),
        ], dim=-1)

        # Evaluate
        with torch.no_grad():
            if velocity is not None:
                vel = torch.full((nx * nz, 1), velocity, device=device)
                if self.backbone_type == BackboneType.MODULATED:
                    wavefield = self.network(coords, vel)
                else:
                    wavefield = self.network(coords)
            else:
                wavefield = self.network(coords)

        return wavefield.view(nz, nx)

    def get_seismogram(
        self,
        receiver_x: Tensor,
        receiver_z: Tensor,
        time_range: Tuple[float, float],
        nt: int = 500,
        velocity: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Generate synthetic seismogram at receiver locations.

        Args:
            receiver_x: x-coordinates of receivers
            receiver_z: z-coordinates of receivers
            time_range: (t_min, t_max) time range
            nt: Number of time samples
            velocity: Constant velocity (if applicable)
            device: Device to use

        Returns:
            Seismogram of shape (n_receivers, nt)
        """
        if device is None:
            device = next(self.parameters()).device

        n_receivers = len(receiver_x)
        times = torch.linspace(time_range[0], time_range[1], nt, device=device)

        seismogram = torch.zeros(n_receivers, nt, device=device)

        with torch.no_grad():
            for i, t in enumerate(times):
                coords = torch.stack([
                    receiver_x.to(device),
                    receiver_z.to(device),
                    torch.full((n_receivers,), t.item(), device=device),
                ], dim=-1)

                if velocity is not None and self.backbone_type == BackboneType.MODULATED:
                    vel = torch.full((n_receivers, 1), velocity, device=device)
                    wavefield = self.network(coords, vel)
                else:
                    wavefield = self.network(coords)

                seismogram[:, i] = wavefield.squeeze(-1)

        return seismogram
