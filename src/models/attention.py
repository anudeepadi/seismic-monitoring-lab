"""
Attention Mechanisms for Spatio-Temporal Physics Learning

Advanced attention modules designed for seismic wave propagation:
- Spatial attention for velocity structure awareness
- Temporal attention for wave propagation dynamics
- Cross-modal attention for multi-source data fusion
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for attention.

    RoPE encodes position information directly into queries and keys
    through rotation, providing better extrapolation properties than
    learned positional embeddings.

    Essential for handling variable-length sequences and extrapolating
    to unseen spatial/temporal ranges.
    """

    def __init__(self, dim: int, max_positions: int = 10000) -> None:
        super().__init__()
        self.dim = dim

        # Compute frequency bands
        inv_freq = 1.0 / (
            max_positions ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """
        Apply rotary position embedding.

        Args:
            x: Input tensor (batch, seq, dim)
            positions: Position indices (batch, seq)

        Returns:
            Position-encoded tensor
        """
        # Compute rotation angles
        freqs = torch.einsum("bi,d->bid", positions.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos_emb = emb.cos()
        sin_emb = emb.sin()

        # Apply rotation
        x1, x2 = x[..., : self.dim // 2], x[..., self.dim // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)

        return x * cos_emb + rotated * sin_emb


class SpatioTemporalAttention(nn.Module):
    """
    Factorized spatio-temporal attention for efficient wave field modeling.

    Processes spatial and temporal dimensions separately to reduce
    computational complexity from O(N²) to O(N*S + N*T) where N is
    total sequence length, S is spatial size, and T is temporal size.

    This is crucial for handling high-resolution seismic data.

    Args:
        dim: Hidden dimension
        num_heads: Number of attention heads
        spatial_size: Size of spatial dimension
        temporal_size: Size of temporal dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        spatial_size: int = 64,
        temporal_size: int = 100,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Spatial attention
        self.spatial_qkv = nn.Linear(dim, 3 * dim)
        self.spatial_out = nn.Linear(dim, dim)

        # Temporal attention
        self.temporal_qkv = nn.Linear(dim, 3 * dim)
        self.temporal_out = nn.Linear(dim, dim)

        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Position embeddings
        self.spatial_rope = RotaryPositionalEmbedding(self.head_dim)
        self.temporal_rope = RotaryPositionalEmbedding(self.head_dim)

    def _attention(
        self,
        qkv_proj: nn.Linear,
        x: Tensor,
        rope: Optional[RotaryPositionalEmbedding] = None,
        positions: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute multi-head self-attention."""
        batch, seq_len, _ = x.shape

        # Project to queries, keys, values
        qkv = qkv_proj(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply rotary position embedding if provided
        if rope is not None and positions is not None:
            q = rope(q.transpose(1, 2), positions).transpose(1, 2)
            k = rope(k.transpose(1, 2), positions).transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, -1)

        return out

    def forward(
        self,
        x: Tensor,
        spatial_size: Optional[Tuple[int, int]] = None,
        temporal_size: Optional[int] = None,
    ) -> Tensor:
        """
        Apply factorized spatio-temporal attention.

        Args:
            x: Input tensor (batch, spatial*temporal, dim)
            spatial_size: (height, width) of spatial grid
            temporal_size: Number of time steps

        Returns:
            Attended tensor (batch, spatial*temporal, dim)
        """
        batch = x.shape[0]

        if spatial_size is None:
            spatial_size = (8, 8)
        if temporal_size is None:
            temporal_size = x.shape[1] // (spatial_size[0] * spatial_size[1])

        H, W = spatial_size
        T = temporal_size

        # Reshape for factorized attention
        # x: (batch, H*W*T, dim) -> (batch*T, H*W, dim)
        x_spatial = x.view(batch, T, H * W, -1).transpose(1, 2)
        x_spatial = x_spatial.reshape(batch * H * W, T, -1)

        # Temporal attention
        x_norm = self.norm1(x_spatial)
        t_positions = torch.arange(T, device=x.device).unsqueeze(0).expand(batch * H * W, -1)
        x_temporal = self._attention(self.temporal_qkv, x_norm, self.temporal_rope, t_positions)
        x_spatial = x_spatial + self.dropout(self.temporal_out(x_temporal))

        # Reshape for spatial attention
        # (batch*H*W, T, dim) -> (batch*T, H*W, dim)
        x_spatial = x_spatial.view(batch, H * W, T, -1).transpose(1, 2)
        x_spatial = x_spatial.reshape(batch * T, H * W, -1)

        # Spatial attention
        x_norm = self.norm2(x_spatial)
        s_positions = torch.arange(H * W, device=x.device).unsqueeze(0).expand(batch * T, -1)
        x_out = self._attention(self.spatial_qkv, x_norm, self.spatial_rope, s_positions)
        x_spatial = x_spatial + self.dropout(self.spatial_out(x_out))

        # Reshape back
        x_out = x_spatial.view(batch, T, H * W, -1).transpose(1, 2)
        x_out = x_out.reshape(batch, H * W * T, -1)

        return x_out


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention for fusing velocity model features with wave coordinates.

    Enables the wave field prediction network to attend to relevant
    regions of the velocity model based on the query coordinates.

    Args:
        query_dim: Dimension of query features (coordinate encoding)
        key_dim: Dimension of key/value features (velocity encoding)
        hidden_dim: Hidden dimension for attention
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, query_dim)

        # Layer norm
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_kv = nn.LayerNorm(key_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply cross-attention.

        Args:
            query: Query features (batch, num_queries, query_dim)
            key_value: Key/value features (batch, num_keys, key_dim)
            mask: Optional attention mask

        Returns:
            Attended query features (batch, num_queries, query_dim)
        """
        batch, num_q, _ = query.shape
        _, num_kv, _ = key_value.shape

        # Normalize
        query = self.norm_q(query)
        key_value = self.norm_kv(key_value)

        # Project
        q = self.query_proj(query).view(batch, num_q, self.num_heads, self.head_dim)
        k = self.key_proj(key_value).view(batch, num_kv, self.num_heads, self.head_dim)
        v = self.value_proj(key_value).view(batch, num_kv, self.num_heads, self.head_dim)

        # Transpose for attention: (batch, heads, seq, dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch, num_q, -1)

        return self.out_proj(out)


class VelocityAwareAttention(nn.Module):
    """
    Attention mechanism that uses velocity model as positional information.

    The local velocity affects wave propagation speed, so this attention
    mechanism uses velocity values to modulate attention patterns.

    Args:
        dim: Hidden dimension
        num_heads: Number of attention heads
        velocity_dim: Dimension of velocity features
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        velocity_dim: int = 1,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Standard attention projections
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

        # Velocity-based attention bias
        self.velocity_bias = nn.Sequential(
            nn.Linear(velocity_dim * 2, dim),  # Pair of velocities
            nn.ReLU(),
            nn.Linear(dim, num_heads),
        )

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
        velocity: Tensor,
    ) -> Tensor:
        """
        Apply velocity-aware attention.

        Args:
            x: Input features (batch, seq, dim)
            velocity: Velocity at each position (batch, seq, velocity_dim)

        Returns:
            Attended features (batch, seq, dim)
        """
        batch, seq_len, _ = x.shape

        # Standard QKV projection
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Standard attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Compute velocity-based attention bias
        # Compare velocity at each pair of positions
        vel_i = velocity.unsqueeze(2).expand(-1, -1, seq_len, -1)
        vel_j = velocity.unsqueeze(1).expand(-1, seq_len, -1, -1)
        vel_pairs = torch.cat([vel_i, vel_j], dim=-1)  # (batch, seq, seq, 2*vel_dim)

        vel_bias = self.velocity_bias(vel_pairs)  # (batch, seq, seq, heads)
        vel_bias = vel_bias.permute(0, 3, 1, 2)  # (batch, heads, seq, seq)

        # Add velocity bias to attention
        attn = attn + vel_bias

        # Softmax and apply
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, -1)

        return x + self.out(out)


class WavefrontAttention(nn.Module):
    """
    Specialized attention for tracking wavefront propagation.

    Uses causal masking in time and distance-based attention in space
    to model how information propagates through the medium.

    The key insight is that wave influence is limited by causality:
    a point at time t can only be influenced by points that could
    have reached it given the local velocity.

    Args:
        dim: Hidden dimension
        num_heads: Number of attention heads
        max_time_steps: Maximum number of time steps
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        max_time_steps: int = 100,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_time_steps = max_time_steps

        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

        # Learnable time decay
        self.time_decay = nn.Parameter(torch.ones(num_heads))

        # Distance kernel
        self.distance_kernel = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, num_heads),
        )

        self.norm = nn.LayerNorm(dim)

    def _compute_causal_mask(
        self,
        coords: Tensor,
        velocity: Tensor,
    ) -> Tensor:
        """
        Compute causal mask based on wave propagation.

        A point (x1, t1) can only be influenced by (x2, t2) if:
        |x1 - x2| <= v * (t1 - t2) and t2 < t1
        """
        batch, seq_len, _ = coords.shape

        # Extract spatial and temporal coordinates
        spatial = coords[..., :-1]  # All but last dimension
        time = coords[..., -1:]  # Last dimension

        # Compute spatial distances
        spatial_diff = spatial.unsqueeze(2) - spatial.unsqueeze(1)
        distances = torch.norm(spatial_diff, dim=-1)  # (batch, seq, seq)

        # Compute time differences
        time_diff = time.squeeze(-1).unsqueeze(2) - time.squeeze(-1).unsqueeze(1)

        # Average velocity for simplicity (could be improved)
        avg_velocity = velocity.mean()

        # Causal condition: distance <= velocity * time_diff and time_diff > 0
        causal_mask = (distances > avg_velocity * time_diff) | (time_diff <= 0)

        return causal_mask

    def forward(
        self,
        x: Tensor,
        coords: Tensor,
        velocity: Tensor,
    ) -> Tensor:
        """
        Apply wavefront-aware attention.

        Args:
            x: Input features (batch, seq, dim)
            coords: Coordinates (batch, seq, coord_dim) with last dim as time
            velocity: Velocity values (batch, seq, 1) or scalar

        Returns:
            Attended features (batch, seq, dim)
        """
        batch, seq_len, _ = x.shape

        # QKV projection
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute base attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add distance-based bias
        spatial = coords[..., :-1]
        spatial_diff = spatial.unsqueeze(2) - spatial.unsqueeze(1)
        distances = torch.norm(spatial_diff, dim=-1, keepdim=True)
        distance_bias = self.distance_kernel(distances)
        distance_bias = distance_bias.permute(0, 3, 1, 2)
        attn = attn + distance_bias

        # Add temporal decay
        time = coords[..., -1]
        time_diff = time.unsqueeze(2) - time.unsqueeze(1)
        time_decay = torch.exp(-self.time_decay.abs().view(1, -1, 1, 1) * time_diff.unsqueeze(1).abs())
        attn = attn * time_decay

        # Apply causal mask
        causal_mask = self._compute_causal_mask(coords, velocity)
        attn = attn.masked_fill(causal_mask.unsqueeze(1), float("-inf"))

        # Softmax and apply
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, 0.0)  # Handle all-masked rows

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, -1)

        return x + self.out(out)
