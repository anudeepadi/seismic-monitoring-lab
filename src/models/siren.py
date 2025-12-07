"""
SIREN (Sinusoidal Representation Networks) Implementation

Based on "Implicit Neural Representations with Periodic Activation Functions"
by Sitzmann et al. (2020). SIREN networks excel at representing complex signals
and their derivatives, making them ideal for physics-informed learning.

Key innovations:
- Periodic sine activations capture high-frequency details
- Principled weight initialization for stable training
- Smooth derivatives enable accurate physics constraint computation
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SineLayer(nn.Module):
    """
    Sine activation layer with specialized initialization.

    The key insight from SIREN is that using sine activations with
    carefully designed initialization allows networks to represent
    complex signals and their derivatives accurately.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        bias: Whether to include bias term
        is_first: Whether this is the first layer (affects initialization)
        omega_0: Frequency scaling factor (default: 30.0 for hidden layers)
        learn_omega: Whether to learn omega_0 during training
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
        learn_omega: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_first = is_first

        # Omega controls the frequency of the sine activation
        if learn_omega:
            self.omega_0 = nn.Parameter(torch.tensor(omega_0))
        else:
            self.register_buffer("omega_0", torch.tensor(omega_0))

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize weights according to SIREN paper.

        First layer: Uniform in [-1/in_features, 1/in_features]
        Other layers: Uniform in [-sqrt(6/in_features)/omega_0, sqrt(6/in_features)/omega_0]

        This ensures that the distribution of activations is preserved through the network.
        """
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0.item()

            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """Apply linear transformation followed by sine activation."""
        return torch.sin(self.omega_0 * self.linear(x))

    def forward_with_intermediate(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Return both pre-activation and post-activation values."""
        pre_activation = self.omega_0 * self.linear(x)
        return torch.sin(pre_activation), pre_activation


class SIRENNetwork(nn.Module):
    """
    Complete SIREN network for physics-informed learning.

    This network is specifically designed for problems requiring accurate
    derivative computation, such as PDEs in physics-informed neural networks.

    Architecture features:
    - Input encoding with configurable first-layer omega
    - Multiple hidden layers with sine activations
    - Optional skip connections for deeper networks
    - Flexible output configuration for multi-task learning

    Args:
        in_features: Dimension of input (e.g., 3 for x, z, t in 2D+time)
        hidden_features: Width of hidden layers
        hidden_layers: Number of hidden layers
        out_features: Dimension of output (e.g., 1 for pressure, 2 for velocity components)
        outermost_linear: Use linear activation on final layer
        first_omega_0: Frequency for first layer (typically 30)
        hidden_omega_0: Frequency for hidden layers (typically 30)
        use_skip: Enable skip connections every N layers
        skip_interval: Interval between skip connections

    Example:
        >>> model = SIRENNetwork(
        ...     in_features=3,  # (x, z, t)
        ...     hidden_features=256,
        ...     hidden_layers=5,
        ...     out_features=1,  # pressure field
        ... )
        >>> coords = torch.randn(1000, 3)  # 1000 points
        >>> pressure = model(coords)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear: bool = True,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        use_skip: bool = True,
        skip_interval: int = 4,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.use_skip = use_skip
        self.skip_interval = skip_interval

        # Build network layers
        self.layers = nn.ModuleList()

        # First layer with special initialization
        self.layers.append(
            SineLayer(
                in_features,
                hidden_features,
                is_first=True,
                omega_0=first_omega_0,
            )
        )

        # Hidden layers
        for i in range(hidden_layers):
            # Adjust input size for skip connections
            layer_in_features = hidden_features
            if use_skip and i > 0 and i % skip_interval == 0:
                layer_in_features = hidden_features + in_features

            self.layers.append(
                SineLayer(
                    layer_in_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        # Output layer
        if outermost_linear:
            final_layer = nn.Linear(hidden_features, out_features)
            # Initialize output layer for stable training
            with torch.no_grad():
                bound = math.sqrt(6.0 / hidden_features) / hidden_omega_0
                final_layer.weight.uniform_(-bound, bound)
                if final_layer.bias is not None:
                    final_layer.bias.zero_()
            self.final_layer = final_layer
        else:
            self.final_layer = SineLayer(
                hidden_features,
                out_features,
                is_first=False,
                omega_0=hidden_omega_0,
            )

        self.outermost_linear = outermost_linear

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input coordinates of shape (batch, in_features)

        Returns:
            Network output of shape (batch, out_features)
        """
        input_coords = x

        for i, layer in enumerate(self.layers):
            if self.use_skip and i > 1 and (i - 1) % self.skip_interval == 0:
                # Concatenate input for skip connection
                x = torch.cat([x, input_coords], dim=-1)
            x = layer(x)

        return self.final_layer(x)

    def forward_with_activations(
        self, x: Tensor, retain_grad: bool = False
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Forward pass returning intermediate activations.

        Useful for visualization and analysis of learned representations.

        Args:
            x: Input coordinates
            retain_grad: Whether to retain gradients for activations

        Returns:
            Tuple of (output, list of intermediate activations)
        """
        activations = []
        input_coords = x

        for i, layer in enumerate(self.layers):
            if self.use_skip and i > 1 and (i - 1) % self.skip_interval == 0:
                x = torch.cat([x, input_coords], dim=-1)
            x = layer(x)
            if retain_grad:
                x.retain_grad()
            activations.append(x)

        output = self.final_layer(x)

        return output, activations

    def get_gradient(
        self, x: Tensor, output_idx: int = 0, create_graph: bool = True
    ) -> Tensor:
        """
        Compute gradient of specified output with respect to input.

        This is essential for physics-informed learning where we need
        derivatives like du/dx, du/dt for the wave equation.

        Args:
            x: Input coordinates requiring gradients
            output_idx: Which output component to differentiate
            create_graph: Whether to create graph for higher-order derivatives

        Returns:
            Gradient tensor of shape (batch, in_features)
        """
        x = x.requires_grad_(True)
        output = self(x)

        if output.dim() > 1 and output.shape[-1] > 1:
            output = output[..., output_idx]

        grad = torch.autograd.grad(
            outputs=output,
            inputs=x,
            grad_outputs=torch.ones_like(output),
            create_graph=create_graph,
            retain_graph=True,
        )[0]

        return grad

    def get_laplacian(
        self, x: Tensor, output_idx: int = 0, spatial_dims: Optional[List[int]] = None
    ) -> Tensor:
        """
        Compute Laplacian (sum of second derivatives) for specified spatial dimensions.

        For the wave equation, we need ∇²u = ∂²u/∂x² + ∂²u/∂z²

        Args:
            x: Input coordinates
            output_idx: Which output component to differentiate
            spatial_dims: Which input dimensions are spatial (default: all except last)

        Returns:
            Laplacian values at each input point
        """
        if spatial_dims is None:
            spatial_dims = list(range(x.shape[-1] - 1))  # Assume last dim is time

        x = x.requires_grad_(True)
        output = self(x)

        if output.dim() > 1 and output.shape[-1] > 1:
            output = output[..., output_idx]

        # First derivatives
        grad = torch.autograd.grad(
            outputs=output,
            inputs=x,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Second derivatives for spatial dimensions
        laplacian = torch.zeros_like(output)
        for dim in spatial_dims:
            grad_component = grad[..., dim]
            grad2 = torch.autograd.grad(
                outputs=grad_component,
                inputs=x,
                grad_outputs=torch.ones_like(grad_component),
                create_graph=True,
                retain_graph=True,
            )[0]
            laplacian = laplacian + grad2[..., dim]

        return laplacian


class GradientScalingSIREN(SIRENNetwork):
    """
    SIREN with learnable gradient scaling for multi-scale physics.

    In seismic applications, different physical quantities may have
    vastly different magnitudes. This variant learns to scale gradients
    appropriately during training.
    """

    def __init__(self, *args, num_scales: int = 3, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Learnable scales for different derivative orders
        self.gradient_scales = nn.Parameter(torch.ones(num_scales))

    def get_scaled_derivatives(
        self, x: Tensor, output_idx: int = 0
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get output and first two derivative orders with learned scaling.

        Returns:
            Tuple of (output, first_derivative, second_derivative)
        """
        x = x.requires_grad_(True)
        output = self(x) * self.gradient_scales[0]

        if output.dim() > 1 and output.shape[-1] > 1:
            output_component = output[..., output_idx]
        else:
            output_component = output.squeeze(-1)

        grad1 = torch.autograd.grad(
            outputs=output_component,
            inputs=x,
            grad_outputs=torch.ones_like(output_component),
            create_graph=True,
            retain_graph=True,
        )[0] * self.gradient_scales[1]

        # Second derivative (for Laplacian computation)
        grad2_components = []
        for i in range(x.shape[-1]):
            g2 = torch.autograd.grad(
                outputs=grad1[..., i],
                inputs=x,
                grad_outputs=torch.ones_like(grad1[..., i]),
                create_graph=True,
                retain_graph=True,
            )[0]
            grad2_components.append(g2)

        grad2 = torch.stack(grad2_components, dim=-2) * self.gradient_scales[2]

        return output, grad1, grad2
