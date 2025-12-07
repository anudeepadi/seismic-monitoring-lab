"""
Modulated SIREN and HyperNetwork for Conditional Physics Learning

This module implements advanced architectures for learning physics
conditioned on velocity models. The key innovation is using a
hypernetwork to modulate SIREN parameters based on local velocity,
enabling the network to adapt its frequency response to local
medium properties.

Based on:
- "Modulated Periodic Activations for Generalizable Local Functional Representations"
- "HyperNetworks" by Ha et al.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .siren import SineLayer, SIRENNetwork


class FilmLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.

    FiLM applies an affine transformation to features conditioned
    on an external signal (e.g., velocity at a point):

    FiLM(x | c) = γ(c) * x + β(c)

    This allows the network to adapt its behavior based on local
    physical properties like velocity.

    Args:
        feature_dim: Dimension of features to modulate
        condition_dim: Dimension of conditioning signal
        hidden_dim: Hidden dimension for γ and β networks
    """

    def __init__(
        self,
        feature_dim: int,
        condition_dim: int,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = condition_dim

        # Networks to generate scale (γ) and shift (β)
        self.gamma_net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        self.beta_net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize to identity transformation."""
        # Initialize gamma to produce ones (identity scale)
        nn.init.zeros_(self.gamma_net[-1].weight)
        nn.init.ones_(self.gamma_net[-1].bias)

        # Initialize beta to produce zeros (no shift)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """
        Apply FiLM modulation.

        Args:
            x: Features to modulate (batch, feature_dim)
            condition: Conditioning signal (batch, condition_dim)

        Returns:
            Modulated features (batch, feature_dim)
        """
        gamma = self.gamma_net(condition)
        beta = self.beta_net(condition)
        return gamma * x + beta


class ModulatedSineLayer(nn.Module):
    """
    Sine layer with external modulation.

    The frequency (omega) of the sine activation is modulated
    by an external signal, allowing the network to adapt its
    frequency response based on local physical properties.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        condition_dim: Dimension of conditioning signal
        base_omega: Base frequency
        modulation_scale: Scale factor for frequency modulation
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        condition_dim: int,
        base_omega: float = 30.0,
        modulation_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.base_omega = base_omega
        self.modulation_scale = modulation_scale

        self.linear = nn.Linear(in_features, out_features)

        # Modulation network for omega
        self.omega_mod = nn.Sequential(
            nn.Linear(condition_dim, out_features),
            nn.Softplus(),  # Ensure positive modulation
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize for stable training."""
        bound = math.sqrt(6.0 / self.in_features) / self.base_omega
        nn.init.uniform_(self.linear.weight, -bound, bound)
        if self.linear.bias is not None:
            nn.init.uniform_(self.linear.bias, -bound, bound)

        # Initialize modulation to produce ones (no modulation)
        nn.init.zeros_(self.omega_mod[-2].weight)
        nn.init.zeros_(self.omega_mod[-2].bias)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """
        Forward pass with conditional frequency modulation.

        Args:
            x: Input features
            condition: Conditioning signal (e.g., local velocity)

        Returns:
            Modulated sine activation output
        """
        # Compute frequency modulation
        omega_scale = 1.0 + self.modulation_scale * (self.omega_mod(condition) - 1.0)
        omega = self.base_omega * omega_scale

        # Apply modulated sine
        return torch.sin(omega * self.linear(x))


class ModulatedSIREN(nn.Module):
    """
    SIREN network with velocity-conditional modulation.

    This architecture learns wave propagation conditioned on the
    local velocity model. The hypernetwork modulates SIREN parameters
    based on velocity, allowing the network to learn velocity-dependent
    wave physics.

    Key innovation: Instead of learning a single wave solution, this
    network learns a family of solutions parameterized by velocity.

    Args:
        coord_dim: Dimension of spatial-temporal coordinates (e.g., 3 for x, z, t)
        velocity_dim: Dimension of velocity conditioning (1 for acoustic, 2 for elastic)
        hidden_features: Width of hidden layers
        hidden_layers: Number of hidden layers
        out_features: Output dimension (e.g., 1 for pressure)
        modulation_type: Type of modulation ('film', 'omega', 'both')
        base_omega: Base frequency for sine activations
    """

    def __init__(
        self,
        coord_dim: int = 3,
        velocity_dim: int = 1,
        hidden_features: int = 256,
        hidden_layers: int = 5,
        out_features: int = 1,
        modulation_type: str = "both",
        base_omega: float = 30.0,
    ) -> None:
        super().__init__()

        self.coord_dim = coord_dim
        self.velocity_dim = velocity_dim
        self.hidden_features = hidden_features
        self.modulation_type = modulation_type

        # Velocity encoder
        self.velocity_encoder = nn.Sequential(
            nn.Linear(velocity_dim, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
        )

        # First layer (coordinate encoding)
        self.first_layer = SineLayer(
            coord_dim, hidden_features, is_first=True, omega_0=base_omega
        )

        # Modulated hidden layers
        self.hidden_layers = nn.ModuleList()
        self.film_layers = nn.ModuleList() if modulation_type in ["film", "both"] else None
        self.omega_mods = nn.ModuleList() if modulation_type in ["omega", "both"] else None

        for _ in range(hidden_layers):
            if modulation_type == "omega":
                self.hidden_layers.append(
                    ModulatedSineLayer(
                        hidden_features, hidden_features, hidden_features, base_omega
                    )
                )
            else:
                self.hidden_layers.append(
                    SineLayer(hidden_features, hidden_features, omega_0=base_omega)
                )
                if self.film_layers is not None:
                    self.film_layers.append(
                        FilmLayer(hidden_features, hidden_features)
                    )

        # Output layer
        self.output_layer = nn.Linear(hidden_features, out_features)
        self._init_output_layer(base_omega)

    def _init_output_layer(self, omega: float) -> None:
        """Initialize output layer for stable training."""
        bound = math.sqrt(6.0 / self.hidden_features) / omega
        nn.init.uniform_(self.output_layer.weight, -bound, bound)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self, coords: Tensor, velocity: Tensor
    ) -> Tensor:
        """
        Forward pass conditioned on velocity.

        Args:
            coords: Spatial-temporal coordinates (batch, coord_dim)
            velocity: Velocity values at coordinates (batch, velocity_dim)

        Returns:
            Wave field values (batch, out_features)
        """
        # Encode velocity
        vel_encoding = self.velocity_encoder(velocity)

        # First layer (coordinate encoding)
        x = self.first_layer(coords)

        # Modulated hidden layers
        for i, layer in enumerate(self.hidden_layers):
            if self.modulation_type == "omega":
                x = layer(x, vel_encoding)
            else:
                x = layer(x)
                if self.film_layers is not None:
                    x = self.film_layers[i](x, vel_encoding)

        return self.output_layer(x)

    def get_derivatives(
        self, coords: Tensor, velocity: Tensor
    ) -> Dict[str, Tensor]:
        """
        Compute wave field and all required derivatives.

        Returns dictionary with:
        - u: Wave field
        - u_x, u_z: Spatial first derivatives
        - u_t: Temporal first derivative
        - u_xx, u_zz: Spatial second derivatives
        - u_tt: Temporal second derivative
        """
        coords = coords.requires_grad_(True)
        u = self(coords, velocity)

        # First derivatives
        grad_u = torch.autograd.grad(
            u.sum(), coords, create_graph=True, retain_graph=True
        )[0]

        derivatives = {
            "u": u,
            "u_x": grad_u[:, 0:1],
            "u_z": grad_u[:, 1:2],
            "u_t": grad_u[:, 2:3],
        }

        # Second derivatives
        for name, grad_component, dim in [
            ("u_xx", derivatives["u_x"], 0),
            ("u_zz", derivatives["u_z"], 1),
            ("u_tt", derivatives["u_t"], 2),
        ]:
            grad2 = torch.autograd.grad(
                grad_component.sum(), coords, create_graph=True, retain_graph=True
            )[0]
            derivatives[name] = grad2[:, dim:dim+1]

        return derivatives


class HyperNetwork(nn.Module):
    """
    HyperNetwork that generates SIREN weights from velocity model.

    Instead of learning fixed weights, this network generates
    the weights of a target SIREN network conditioned on the
    velocity model. This allows a single hypernetwork to represent
    an infinite family of solutions.

    Args:
        velocity_encoder_dim: Hidden dimension for velocity encoding
        target_in_features: Input dimension of target network
        target_hidden_features: Hidden dimension of target network
        target_hidden_layers: Number of layers in target network
        target_out_features: Output dimension of target network
    """

    def __init__(
        self,
        velocity_encoder_dim: int = 256,
        target_in_features: int = 3,
        target_hidden_features: int = 128,
        target_hidden_layers: int = 3,
        target_out_features: int = 1,
    ) -> None:
        super().__init__()

        self.target_in_features = target_in_features
        self.target_hidden_features = target_hidden_features
        self.target_hidden_layers = target_hidden_layers
        self.target_out_features = target_out_features

        # Calculate total number of parameters in target network
        self.param_shapes = self._get_param_shapes()
        total_params = sum(s[0] * s[1] if len(s) == 2 else s[0] for s in self.param_shapes)

        # Hypernetwork: velocity → target network parameters
        self.hyper_net = nn.Sequential(
            nn.Linear(velocity_encoder_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, total_params),
        )

        # Velocity encoder (processes velocity field)
        self.velocity_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 16, velocity_encoder_dim),
        )

    def _get_param_shapes(self) -> List[Tuple[int, ...]]:
        """Get shapes of all parameters in target network."""
        shapes = []

        # First layer
        shapes.append((self.target_hidden_features, self.target_in_features))  # weight
        shapes.append((self.target_hidden_features,))  # bias

        # Hidden layers
        for _ in range(self.target_hidden_layers):
            shapes.append((self.target_hidden_features, self.target_hidden_features))
            shapes.append((self.target_hidden_features,))

        # Output layer
        shapes.append((self.target_out_features, self.target_hidden_features))
        shapes.append((self.target_out_features,))

        return shapes

    def forward(
        self, velocity_model: Tensor, coords: Tensor
    ) -> Tensor:
        """
        Generate target network and evaluate at coordinates.

        Args:
            velocity_model: 2D velocity field (batch, 1, height, width)
            coords: Query coordinates (batch, num_points, coord_dim)

        Returns:
            Wave field at query coordinates (batch, num_points, out_features)
        """
        batch_size = velocity_model.shape[0]
        num_points = coords.shape[1]

        # Encode velocity model
        vel_encoding = self.velocity_encoder(velocity_model)

        # Generate target network parameters
        all_params = self.hyper_net(vel_encoding)

        # Split parameters for each layer
        params = []
        offset = 0
        for shape in self.param_shapes:
            size = 1
            for s in shape:
                size *= s
            param = all_params[:, offset:offset + size].view(batch_size, *shape)
            params.append(param)
            offset += size

        # Forward through generated network
        x = coords  # (batch, num_points, coord_dim)

        param_idx = 0
        omega = 30.0

        # First layer
        weight, bias = params[param_idx], params[param_idx + 1]
        x = torch.einsum("bpc,boc->bpo", x, weight) + bias.unsqueeze(1)
        x = torch.sin(omega * x)
        param_idx += 2

        # Hidden layers
        for _ in range(self.target_hidden_layers):
            weight, bias = params[param_idx], params[param_idx + 1]
            x = torch.einsum("bpo,bno->bpn", x, weight) + bias.unsqueeze(1)
            x = torch.sin(omega * x)
            param_idx += 2

        # Output layer
        weight, bias = params[param_idx], params[param_idx + 1]
        x = torch.einsum("bpo,bno->bpn", x, weight) + bias.unsqueeze(1)

        return x


class VelocityConditionedPINN(nn.Module):
    """
    Full PINN architecture conditioned on velocity model.

    Combines:
    1. Velocity encoder (CNN) to extract features from velocity model
    2. Coordinate encoder (SIREN) to process space-time coordinates
    3. Cross-attention to fuse velocity and coordinate information
    4. Modulated decoder to produce wave field

    This architecture can:
    - Learn to solve the wave equation for any velocity model
    - Generalize to unseen velocity distributions
    - Efficiently query wave field at arbitrary space-time points

    Args:
        velocity_channels: Number of velocity components (1 for acoustic)
        coord_dim: Dimension of coordinates (3 for 2D + time)
        hidden_dim: Hidden dimension throughout network
        num_layers: Number of processing layers
        out_dim: Output dimension
    """

    def __init__(
        self,
        velocity_channels: int = 1,
        coord_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 6,
        out_dim: int = 1,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim

        # Velocity model encoder (UNet-style)
        self.vel_encoder = VelocityEncoder(velocity_channels, hidden_dim)

        # Coordinate encoder (SIREN)
        self.coord_encoder = SIRENNetwork(
            in_features=coord_dim,
            hidden_features=hidden_dim,
            hidden_layers=3,
            out_features=hidden_dim,
            outermost_linear=True,
        )

        # Cross-attention layers
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
            for _ in range(num_layers // 2)
        ])

        # Modulated decoder
        self.decoder = ModulatedSIREN(
            coord_dim=hidden_dim,  # Takes encoded coordinates
            velocity_dim=hidden_dim,  # Takes velocity features
            hidden_features=hidden_dim,
            hidden_layers=num_layers // 2,
            out_features=out_dim,
        )

    def forward(
        self,
        velocity_model: Tensor,
        coords: Tensor,
        source_coords: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Predict wave field given velocity model and query coordinates.

        Args:
            velocity_model: 2D velocity field (batch, C, H, W)
            coords: Query coordinates (batch, num_points, coord_dim)
            source_coords: Optional source location (batch, coord_dim)

        Returns:
            Wave field at query points (batch, num_points, out_dim)
        """
        batch_size, num_points, _ = coords.shape

        # Encode velocity model → (batch, num_tokens, hidden_dim)
        vel_features, vel_tokens = self.vel_encoder(velocity_model)

        # Encode coordinates → (batch, num_points, hidden_dim)
        coords_flat = coords.view(-1, coords.shape[-1])
        coord_features = self.coord_encoder(coords_flat)
        coord_features = coord_features.view(batch_size, num_points, -1)

        # Cross-attention: coordinates query velocity features
        for attn in self.cross_attention:
            attended, _ = attn(coord_features, vel_tokens, vel_tokens)
            coord_features = coord_features + attended

        # Sample velocity features at coordinate locations
        # For simplicity, use global velocity feature
        vel_conditioning = vel_features.unsqueeze(1).expand(-1, num_points, -1)

        # Decode with modulated SIREN
        output = self.decoder(coord_features, vel_conditioning)

        return output


class VelocityEncoder(nn.Module):
    """
    Encoder for 2D velocity models using CNN with attention.

    Extracts multi-scale features from velocity model that can
    be used to condition the wave field prediction.
    """

    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()

        # Multi-scale CNN
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

        self.norm1 = nn.GroupNorm(8, 32)
        self.norm2 = nn.GroupNorm(8, 64)
        self.norm3 = nn.GroupNorm(8, 128)
        self.norm4 = nn.GroupNorm(8, 256)

        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(256, hidden_dim)

        # Token projection for cross-attention
        self.token_project = nn.Conv2d(256, hidden_dim, 1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode velocity model.

        Returns:
            Tuple of (global_features, token_features)
            - global_features: (batch, hidden_dim)
            - token_features: (batch, num_tokens, hidden_dim)
        """
        # Multi-scale encoding
        x = F.gelu(self.norm1(self.conv1(x)))
        x = F.gelu(self.norm2(self.conv2(x)))
        x = F.gelu(self.norm3(self.conv3(x)))
        x = F.gelu(self.norm4(self.conv4(x)))

        # Global features
        global_feat = self.global_pool(x).flatten(1)
        global_feat = self.project(global_feat)

        # Token features for cross-attention
        tokens = self.token_project(x)  # (batch, hidden_dim, H, W)
        tokens = tokens.flatten(2).transpose(1, 2)  # (batch, num_tokens, hidden_dim)

        return global_feat, tokens
