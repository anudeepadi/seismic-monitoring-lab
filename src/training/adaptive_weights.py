"""
Adaptive Loss Weighting Strategies for Multi-Task PINN Training

Training PINNs involves balancing multiple loss components (data, physics,
boundaries, initial conditions). Adaptive weighting strategies automatically
adjust the relative importance of each component during training.

Implemented strategies:
- GradNorm: Balance gradient magnitudes across tasks
- Uncertainty weighting: Learn task-specific uncertainties
- SoftAdapt: Dynamic adaptation based on loss changes
- Self-paced learning: Curriculum-based weighting
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class AdaptiveLossWeighting(ABC):
    """Abstract base class for adaptive loss weighting strategies."""

    @abstractmethod
    def get_weights(self, losses: Dict[str, Tensor]) -> Dict[str, float]:
        """
        Compute loss weights based on current losses.

        Args:
            losses: Dictionary of individual loss components

        Returns:
            Dictionary of weights for each loss component
        """
        pass

    @abstractmethod
    def update(self, losses: Dict[str, Tensor], grads: Optional[Dict] = None) -> None:
        """
        Update internal state after training step.

        Args:
            losses: Current loss values
            grads: Optional gradient information
        """
        pass


class GradNormWeighting(AdaptiveLossWeighting, nn.Module):
    """
    GradNorm Adaptive Weighting.

    Balances losses by ensuring gradient magnitudes are similar across
    all tasks. This prevents any single loss from dominating training.

    Reference: "GradNorm: Gradient Normalization for Adaptive Loss Balancing
    in Deep Multitask Networks" (Chen et al., 2018)

    Args:
        loss_names: Names of loss components
        alpha: Asymmetry parameter (higher = more aggressive balancing)
        lr: Learning rate for weight updates
    """

    def __init__(
        self,
        loss_names: List[str],
        alpha: float = 1.5,
        lr: float = 0.01,
    ) -> None:
        super().__init__()

        self.loss_names = loss_names
        self.alpha = alpha
        self.lr = lr

        # Learnable log-weights (ensures positivity)
        self.log_weights = nn.Parameter(torch.zeros(len(loss_names)))

        # Track initial losses for relative weighting
        self.initial_losses: Dict[str, float] = {}
        self.initialized = False

        # History for smoothing
        self.loss_history: Dict[str, List[float]] = {name: [] for name in loss_names}

    def get_weights(self, losses: Dict[str, Tensor]) -> Dict[str, float]:
        """Compute current weights."""
        weights = torch.softmax(self.log_weights, dim=0)
        return {
            name: weights[i].item() * len(self.loss_names)
            for i, name in enumerate(self.loss_names)
        }

    def update(
        self,
        losses: Dict[str, Tensor],
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """Update weights based on gradient norms."""
        # Initialize with first losses
        if not self.initialized:
            self.initial_losses = {
                name: losses.get(name, torch.tensor(1.0)).item()
                for name in self.loss_names
            }
            self.initialized = True

        # Track loss history
        for name in self.loss_names:
            if name in losses:
                self.loss_history[name].append(losses[name].item())
                if len(self.loss_history[name]) > 100:
                    self.loss_history[name] = self.loss_history[name][-100:]

        if model is None or optimizer is None:
            return

        # Compute gradient norms for each loss
        grad_norms = []
        weights = torch.softmax(self.log_weights, dim=0)

        for i, name in enumerate(self.loss_names):
            if name not in losses:
                grad_norms.append(torch.tensor(1.0))
                continue

            loss = losses[name]
            if not loss.requires_grad:
                grad_norms.append(torch.tensor(1.0))
                continue

            # Get last layer gradients
            model.zero_grad()
            weighted_loss = weights[i] * loss
            weighted_loss.backward(retain_graph=True)

            # Compute norm of last layer gradients
            total_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    total_norm += param.grad.norm().item() ** 2
            grad_norms.append(torch.tensor(total_norm ** 0.5))

        grad_norms = torch.stack(grad_norms)
        avg_grad_norm = grad_norms.mean()

        # Compute relative inverse training rates
        relative_losses = torch.tensor([
            losses.get(name, torch.tensor(1.0)).item() /
            max(self.initial_losses.get(name, 1.0), 1e-8)
            for name in self.loss_names
        ])
        avg_relative_loss = relative_losses.mean()
        inverse_train_rates = relative_losses / avg_relative_loss

        # Target gradient norms
        target_grad_norms = avg_grad_norm * inverse_train_rates.pow(self.alpha)

        # GradNorm loss
        grad_norm_loss = torch.abs(grad_norms - target_grad_norms).sum()

        # Update weights
        self.log_weights.grad = torch.autograd.grad(
            grad_norm_loss, self.log_weights, retain_graph=True
        )[0]

        with torch.no_grad():
            self.log_weights -= self.lr * self.log_weights.grad

            # Renormalize
            self.log_weights -= self.log_weights.mean()


class UncertaintyWeighting(AdaptiveLossWeighting, nn.Module):
    """
    Uncertainty-based Multi-Task Weighting.

    Learns homoscedastic uncertainty for each task and uses it to
    weight losses. Tasks with higher uncertainty get lower weight.

    Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses
    for Scene Geometry and Semantics" (Kendall et al., 2018)

    The loss for task i becomes: L_i / (2 * σ_i²) + log(σ_i)
    where σ_i is the learned uncertainty.

    Args:
        loss_names: Names of loss components
        init_uncertainty: Initial uncertainty values
    """

    def __init__(
        self,
        loss_names: List[str],
        init_uncertainty: float = 1.0,
    ) -> None:
        super().__init__()

        self.loss_names = loss_names

        # Log variance parameters (σ² = exp(log_var))
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(2.0 * init_uncertainty).log())
            for name in loss_names
        })

    def get_weights(self, losses: Dict[str, Tensor]) -> Dict[str, float]:
        """Compute weights from learned uncertainties."""
        weights = {}
        for name in self.loss_names:
            # Weight = 1 / (2 * σ²) = 0.5 * exp(-log_var)
            if name in self.log_vars:
                weights[name] = 0.5 * torch.exp(-self.log_vars[name]).item()
            else:
                weights[name] = 1.0
        return weights

    def compute_weighted_loss(
        self,
        losses: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute total loss with uncertainty weighting.

        Returns both the total loss and individual weights.
        """
        total_loss = torch.tensor(0.0, device=next(iter(losses.values())).device)
        weights = {}

        for name in self.loss_names:
            if name not in losses:
                continue

            loss = losses[name]
            log_var = self.log_vars[name]

            # Uncertainty-weighted loss: L / (2σ²) + log(σ)
            precision = torch.exp(-log_var)
            weighted = 0.5 * precision * loss + 0.5 * log_var

            total_loss = total_loss + weighted
            weights[name] = (0.5 * precision).item()

        return total_loss, weights

    def update(self, losses: Dict[str, Tensor], grads: Optional[Dict] = None) -> None:
        """No explicit update needed - uncertainties learned via backprop."""
        pass


class SoftAdaptWeighting(AdaptiveLossWeighting):
    """
    SoftAdapt Dynamic Weighting.

    Adjusts weights based on the rate of change of each loss.
    Losses that are improving slowly get higher weight.

    Reference: "SoftAdapt: Techniques for Adaptive Loss Weighting of
    Neural Networks with Multi-Part Loss Functions" (Heydari et al., 2019)

    Args:
        loss_names: Names of loss components
        beta: Softmax temperature (higher = more uniform)
        lookback: Number of steps to look back for rate computation
    """

    def __init__(
        self,
        loss_names: List[str],
        beta: float = 0.1,
        lookback: int = 10,
    ) -> None:
        self.loss_names = loss_names
        self.beta = beta
        self.lookback = lookback

        self.loss_history: Dict[str, List[float]] = {name: [] for name in loss_names}
        self.current_weights: Dict[str, float] = {name: 1.0 for name in loss_names}

    def get_weights(self, losses: Dict[str, Tensor]) -> Dict[str, float]:
        """Return current weights."""
        return self.current_weights.copy()

    def update(self, losses: Dict[str, Tensor], grads: Optional[Dict] = None) -> None:
        """Update weights based on loss improvement rates."""
        # Update history
        for name in self.loss_names:
            if name in losses:
                loss_val = losses[name].item() if isinstance(losses[name], Tensor) else losses[name]
                self.loss_history[name].append(loss_val)

                # Keep only lookback entries
                if len(self.loss_history[name]) > self.lookback:
                    self.loss_history[name] = self.loss_history[name][-self.lookback:]

        # Need at least 2 entries to compute rate
        if all(len(h) >= 2 for h in self.loss_history.values()):
            # Compute improvement rates
            rates = []
            for name in self.loss_names:
                history = self.loss_history[name]
                if len(history) >= 2:
                    # Rate of change (positive = improving)
                    rate = (history[-2] - history[-1]) / (abs(history[-2]) + 1e-8)
                    rates.append(-rate)  # Invert: slower improvement = higher weight
                else:
                    rates.append(0.0)

            # Apply softmax with temperature
            rates_tensor = torch.tensor(rates) / self.beta
            weights = torch.softmax(rates_tensor, dim=0) * len(self.loss_names)

            self.current_weights = {
                name: weights[i].item()
                for i, name in enumerate(self.loss_names)
            }


class ReLoWeighting(AdaptiveLossWeighting):
    """
    Relative Loss Balancing with Random Lookback (ReLoBRaLo).

    Balances losses using their relative magnitudes with
    random lookback windows to prevent oscillations.

    Args:
        loss_names: Names of loss components
        temperature: Softmax temperature
        max_lookback: Maximum lookback window size
    """

    def __init__(
        self,
        loss_names: List[str],
        temperature: float = 1.0,
        max_lookback: int = 20,
    ) -> None:
        self.loss_names = loss_names
        self.temperature = temperature
        self.max_lookback = max_lookback

        self.loss_history: Dict[str, List[float]] = {name: [] for name in loss_names}
        self.current_weights: Dict[str, float] = {name: 1.0 for name in loss_names}

    def get_weights(self, losses: Dict[str, Tensor]) -> Dict[str, float]:
        """Return current weights."""
        return self.current_weights.copy()

    def update(self, losses: Dict[str, Tensor], grads: Optional[Dict] = None) -> None:
        """Update weights using random lookback."""
        # Update history
        for name in self.loss_names:
            if name in losses:
                loss_val = losses[name].item() if isinstance(losses[name], Tensor) else losses[name]
                self.loss_history[name].append(loss_val)

                if len(self.loss_history[name]) > self.max_lookback:
                    self.loss_history[name] = self.loss_history[name][-self.max_lookback:]

        # Random lookback
        lookback = torch.randint(1, self.max_lookback + 1, (1,)).item()

        # Compute relative changes
        relative_changes = []
        for name in self.loss_names:
            history = self.loss_history[name]
            if len(history) >= lookback + 1:
                old_val = history[-lookback - 1]
                new_val = history[-1]
                change = new_val / (old_val + 1e-8)
                relative_changes.append(change)
            else:
                relative_changes.append(1.0)

        # Softmax weighting
        changes_tensor = torch.tensor(relative_changes) / self.temperature
        weights = torch.softmax(changes_tensor, dim=0) * len(self.loss_names)

        self.current_weights = {
            name: weights[i].item()
            for i, name in enumerate(self.loss_names)
        }


class CausalWeighting(AdaptiveLossWeighting):
    """
    Causal Training Strategy for PINNs.

    Weights physics loss based on temporal causality - earlier
    time points are weighted more heavily initially, and the
    training frontier advances as the solution converges.

    Reference: "Respecting causality is all you need for
    training physics-informed neural networks" (Wang et al., 2022)

    Args:
        t_range: Time range (t_min, t_max)
        epsilon: Convergence threshold for advancing frontier
        n_steps: Number of causal steps
    """

    def __init__(
        self,
        t_range: Tuple[float, float],
        epsilon: float = 1e-3,
        n_steps: int = 10,
    ) -> None:
        self.t_min, self.t_max = t_range
        self.epsilon = epsilon
        self.n_steps = n_steps

        self.current_frontier = 0  # Index of current causal step
        self.step_boundaries = torch.linspace(self.t_min, self.t_max, n_steps + 1)
        self.residual_history: List[float] = []

    def get_time_weights(self, t: Tensor) -> Tensor:
        """
        Compute time-dependent weights for causal training.

        Points before the frontier get full weight,
        points after get exponentially decaying weight.
        """
        t_frontier = self.step_boundaries[min(self.current_frontier + 1, self.n_steps)]

        # Exponential decay after frontier
        weights = torch.exp(-10.0 * torch.relu(t - t_frontier) / (self.t_max - self.t_min))

        return weights

    def get_weights(self, losses: Dict[str, Tensor]) -> Dict[str, float]:
        """Return standard weights (causal weighting is applied to points)."""
        return {"physics": 1.0, "data": 1.0, "boundary": 1.0, "initial": 1.0}

    def update(self, losses: Dict[str, Tensor], grads: Optional[Dict] = None) -> None:
        """Update causal frontier based on residual."""
        if "physics" in losses:
            residual = losses["physics"].item()
            self.residual_history.append(residual)

            # Check if converged at current frontier
            if len(self.residual_history) >= 10:
                recent = self.residual_history[-10:]
                if max(recent) < self.epsilon:
                    # Advance frontier
                    self.current_frontier = min(
                        self.current_frontier + 1, self.n_steps - 1
                    )
                    self.residual_history = []


from typing import Tuple
