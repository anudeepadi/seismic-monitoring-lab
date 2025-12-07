"""
Fourier Feature Networks for Physics-Informed Neural Networks

Based on "Fourier Features Let Networks Learn High Frequency Functions
in Low Dimensional Domains" by Tancik et al. (2020).

Key insight: Standard MLPs have spectral bias toward low frequencies.
Random Fourier features enable learning high-frequency functions,
which is critical for capturing sharp wavefronts in seismic data.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


class GaussianFourierFeatures(nn.Module):
    """
    Gaussian Fourier feature mapping for positional encoding.

    Maps input coordinates to a higher-dimensional space using
    sinusoidal functions with random frequencies, enabling the
    network to learn high-frequency patterns.

    The mapping is: γ(x) = [cos(2πBx), sin(2πBx)]
    where B is a random matrix with entries from N(0, σ²)

    Args:
        in_features: Input dimension
        num_frequencies: Number of frequency components (output dim = 2 * num_frequencies)
        scale: Standard deviation of the Gaussian (controls frequency range)
        learnable: Whether to learn the frequency matrix during training
    """

    def __init__(
        self,
        in_features: int,
        num_frequencies: int = 256,
        scale: float = 10.0,
        learnable: bool = False,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.scale = scale

        # Random frequency matrix
        B = torch.randn(in_features, num_frequencies) * scale

        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

    @property
    def out_features(self) -> int:
        """Output dimension after Fourier mapping."""
        return 2 * self.num_frequencies

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Fourier feature mapping.

        Args:
            x: Input coordinates of shape (..., in_features)

        Returns:
            Fourier features of shape (..., 2 * num_frequencies)
        """
        # Project to frequency space
        x_proj = 2 * math.pi * x @ self.B

        # Concatenate sin and cos
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


class MultiScaleFourierFeatures(nn.Module):
    """
    Multi-scale Fourier features for capturing multiple frequency bands.

    Seismic data contains information at multiple scales - from broad
    velocity gradients to sharp layer boundaries. This module uses
    multiple frequency scales to capture all relevant features.

    Args:
        in_features: Input dimension
        num_frequencies_per_scale: Frequencies per scale
        scales: List of scale factors (e.g., [1, 2, 4, 8, 16])
        base_scale: Base standard deviation for lowest frequency band
    """

    def __init__(
        self,
        in_features: int,
        num_frequencies_per_scale: int = 64,
        scales: Optional[list] = None,
        base_scale: float = 1.0,
    ) -> None:
        super().__init__()

        if scales is None:
            scales = [1.0, 2.0, 4.0, 8.0, 16.0]

        self.scales = scales
        self.num_frequencies_per_scale = num_frequencies_per_scale

        # Create Fourier features for each scale
        self.fourier_layers = nn.ModuleList([
            GaussianFourierFeatures(
                in_features,
                num_frequencies_per_scale,
                scale=base_scale * s,
            )
            for s in scales
        ])

    @property
    def out_features(self) -> int:
        """Total output dimension."""
        return 2 * self.num_frequencies_per_scale * len(self.scales)

    def forward(self, x: Tensor) -> Tensor:
        """Apply multi-scale Fourier mapping."""
        features = [layer(x) for layer in self.fourier_layers]
        return torch.cat(features, dim=-1)


class FourierFeatureNetwork(nn.Module):
    """
    Complete network with Fourier feature input encoding.

    This architecture combines Fourier feature mapping with a
    standard MLP backbone, enabling learning of high-frequency
    functions while maintaining stable training.

    Architecture:
    1. Input → Fourier Features (positional encoding)
    2. Fourier Features → MLP with residual connections
    3. MLP → Output

    Args:
        in_features: Input dimension (e.g., 3 for x, z, t)
        hidden_features: Width of hidden MLP layers
        hidden_layers: Number of hidden layers
        out_features: Output dimension
        num_frequencies: Number of Fourier frequency components
        frequency_scale: Scale for Fourier features
        activation: Activation function ('relu', 'gelu', 'swish')
        use_residual: Enable residual connections
        dropout: Dropout probability
        multi_scale: Use multi-scale Fourier features
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        num_frequencies: int = 256,
        frequency_scale: float = 10.0,
        activation: str = "gelu",
        use_residual: bool = True,
        dropout: float = 0.0,
        multi_scale: bool = True,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_residual = use_residual

        # Fourier feature encoding
        if multi_scale:
            self.fourier = MultiScaleFourierFeatures(
                in_features,
                num_frequencies_per_scale=num_frequencies // 5,
                base_scale=frequency_scale,
            )
        else:
            self.fourier = GaussianFourierFeatures(
                in_features,
                num_frequencies,
                scale=frequency_scale,
            )

        # Select activation function
        activation_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        self.activation = activation_map.get(activation, nn.GELU())

        # Build MLP layers
        fourier_out = self.fourier.out_features
        self.input_layer = nn.Linear(fourier_out, hidden_features)
        self.input_norm = nn.LayerNorm(hidden_features)

        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()

        for _ in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_features, hidden_features))
            self.hidden_norms.append(nn.LayerNorm(hidden_features))

        # Output layer
        self.output_layer = nn.Linear(hidden_features, out_features)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input coordinates of shape (batch, in_features)

        Returns:
            Network output of shape (batch, out_features)
        """
        # Fourier feature encoding
        x = self.fourier(x)

        # Input projection
        x = self.input_layer(x)
        x = self.input_norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Hidden layers with optional residual connections
        for layer, norm in zip(self.hidden_layers, self.hidden_norms):
            if self.use_residual:
                residual = x
                x = layer(x)
                x = norm(x)
                x = self.activation(x)
                x = self.dropout(x)
                x = x + residual
            else:
                x = layer(x)
                x = norm(x)
                x = self.activation(x)
                x = self.dropout(x)

        # Output projection
        return self.output_layer(x)

    def get_gradient(
        self, x: Tensor, output_idx: int = 0, create_graph: bool = True
    ) -> Tensor:
        """Compute gradient of output with respect to input."""
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


class PositionalEncodingFourier(nn.Module):
    """
    Deterministic positional encoding with logarithmically-spaced frequencies.

    Similar to the positional encoding used in NeRF, this uses fixed
    sinusoidal functions at different frequencies to encode position.

    Args:
        in_features: Input dimension
        num_frequencies: Number of frequency octaves
        min_freq: Minimum frequency
        max_freq: Maximum frequency
        include_input: Whether to concatenate original input
    """

    def __init__(
        self,
        in_features: int,
        num_frequencies: int = 10,
        min_freq: float = 1.0,
        max_freq: float = 512.0,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        # Logarithmically-spaced frequencies
        freqs = 2.0 ** torch.linspace(
            math.log2(min_freq),
            math.log2(max_freq),
            num_frequencies,
        )
        self.register_buffer("freqs", freqs)

    @property
    def out_features(self) -> int:
        """Output dimension after encoding."""
        base = 2 * self.in_features * self.num_frequencies
        if self.include_input:
            base += self.in_features
        return base

    def forward(self, x: Tensor) -> Tensor:
        """Apply positional encoding."""
        # x: (..., in_features)
        # freqs: (num_frequencies,)

        # Outer product of input with frequencies
        x_freq = x.unsqueeze(-1) * self.freqs  # (..., in_features, num_frequencies)
        x_freq = x_freq.view(*x.shape[:-1], -1)  # (..., in_features * num_frequencies)

        # Apply sin and cos
        encoded = torch.cat([torch.sin(x_freq), torch.cos(x_freq)], dim=-1)

        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)

        return encoded


class AdaptiveFourierFeatures(nn.Module):
    """
    Adaptive Fourier features that learn optimal frequency distribution.

    Instead of using fixed random frequencies, this module learns
    the frequency distribution during training, allowing the network
    to adapt to the specific frequency content of the data.

    Args:
        in_features: Input dimension
        num_frequencies: Number of frequency components
        init_scale: Initial scale for frequencies
    """

    def __init__(
        self,
        in_features: int,
        num_frequencies: int = 256,
        init_scale: float = 10.0,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.num_frequencies = num_frequencies

        # Learnable frequency matrix
        self.frequencies = nn.Parameter(
            torch.randn(in_features, num_frequencies) * init_scale
        )

        # Learnable phase shifts
        self.phases = nn.Parameter(torch.zeros(num_frequencies))

        # Learnable amplitude scaling
        self.amplitudes = nn.Parameter(torch.ones(num_frequencies))

    @property
    def out_features(self) -> int:
        """Output dimension."""
        return 2 * self.num_frequencies

    def forward(self, x: Tensor) -> Tensor:
        """Apply adaptive Fourier mapping."""
        # Project to frequency space with learnable frequencies
        x_proj = 2 * math.pi * x @ self.frequencies + self.phases

        # Apply with learnable amplitudes
        cos_features = self.amplitudes * torch.cos(x_proj)
        sin_features = self.amplitudes * torch.sin(x_proj)

        return torch.cat([cos_features, sin_features], dim=-1)

    def get_frequency_spectrum(self) -> Tensor:
        """
        Get the learned frequency spectrum for analysis.

        Returns:
            Tensor of shape (in_features, num_frequencies) with frequency magnitudes
        """
        return torch.abs(self.frequencies)
