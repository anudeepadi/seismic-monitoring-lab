"""
Seismic Source Functions

This module provides various source functions commonly used in
seismic modeling and inversion:
- Ricker wavelet (Mexican hat)
- Gaussian sources
- Point sources
- Plane waves

These sources inject energy into the medium and drive wave propagation.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
import math


class SourceFunction(ABC):
    """
    Abstract base class for seismic source functions.

    A source function defines how energy is injected into the medium.
    It has both spatial and temporal components:

        f(x, z, t) = S(x, z) * g(t)

    where S(x, z) is the spatial distribution and g(t) is the time function.
    """

    @abstractmethod
    def evaluate(
        self,
        x: Tensor,
        z: Tensor,
        t: Tensor,
    ) -> Tensor:
        """
        Evaluate source function at given locations and times.

        Args:
            x: x-coordinates
            z: z-coordinates
            t: time values

        Returns:
            Source amplitude at each point
        """
        pass

    @abstractmethod
    def time_function(self, t: Tensor) -> Tensor:
        """
        Evaluate the temporal part of the source.

        Args:
            t: time values

        Returns:
            Source amplitude at each time
        """
        pass


class RickerWavelet(SourceFunction):
    """
    Ricker Wavelet (Mexican Hat) Source.

    The Ricker wavelet is the second derivative of a Gaussian and
    is the most commonly used source in seismic modeling:

        g(t) = (1 - 2π²f₀²(t-t₀)²) * exp(-π²f₀²(t-t₀)²)

    where f₀ is the peak frequency and t₀ is the time delay.

    Properties:
    - Zero DC component (no low-frequency content)
    - Well-defined peak frequency
    - Smooth temporal behavior

    Args:
        peak_frequency: Peak frequency in Hz
        time_delay: Time shift (centers the wavelet)
        source_x: x-coordinate of source location
        source_z: z-coordinate of source location
        amplitude: Source amplitude scaling
        spatial_sigma: Width of spatial Gaussian (point source if small)
    """

    def __init__(
        self,
        peak_frequency: float = 20.0,
        time_delay: float = 0.0,
        source_x: float = 0.0,
        source_z: float = 0.0,
        amplitude: float = 1.0,
        spatial_sigma: float = 1e-6,
    ) -> None:
        self.peak_frequency = peak_frequency
        self.time_delay = time_delay
        self.source_x = source_x
        self.source_z = source_z
        self.amplitude = amplitude
        self.spatial_sigma = spatial_sigma

        # Derived constants
        self.pi_f = math.pi * peak_frequency

    def time_function(self, t: Tensor) -> Tensor:
        """
        Compute Ricker wavelet time function.

        Returns:
            Wavelet amplitude at each time
        """
        tau = t - self.time_delay
        arg = (self.pi_f * tau) ** 2

        return self.amplitude * (1.0 - 2.0 * arg) * torch.exp(-arg)

    def spatial_function(self, x: Tensor, z: Tensor) -> Tensor:
        """
        Compute spatial distribution (Gaussian around source point).

        Args:
            x: x-coordinates
            z: z-coordinates

        Returns:
            Spatial weight at each point
        """
        dx = x - self.source_x
        dz = z - self.source_z
        r2 = dx**2 + dz**2

        return torch.exp(-r2 / (2.0 * self.spatial_sigma**2))

    def evaluate(self, x: Tensor, z: Tensor, t: Tensor) -> Tensor:
        """
        Evaluate full source function.

        Args:
            x: x-coordinates
            z: z-coordinates
            t: time values

        Returns:
            Source amplitude at each (x, z, t)
        """
        temporal = self.time_function(t)
        spatial = self.spatial_function(x, z)

        return temporal * spatial

    def get_source_signature(
        self,
        time_vector: Tensor,
    ) -> Tensor:
        """
        Get the source time signature for plotting.

        Args:
            time_vector: Time samples

        Returns:
            Source amplitude at each time
        """
        return self.time_function(time_vector)

    def get_frequency_spectrum(
        self,
        time_vector: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute the frequency spectrum of the source.

        Args:
            time_vector: Time samples (uniformly spaced)

        Returns:
            Tuple of (frequencies, amplitudes)
        """
        dt = time_vector[1] - time_vector[0]
        n = len(time_vector)

        signal = self.time_function(time_vector)
        spectrum = torch.fft.rfft(signal)
        frequencies = torch.fft.rfftfreq(n, dt)

        return frequencies, torch.abs(spectrum)


class GaussianSource(SourceFunction):
    """
    Gaussian Source Function.

    A simple Gaussian pulse in time:

        g(t) = A * exp(-(t-t₀)²/(2σ²))

    Useful for smooth, broadband sources.

    Args:
        sigma_t: Temporal standard deviation (controls duration)
        time_delay: Time shift
        source_x: x-coordinate of source
        source_z: z-coordinate of source
        amplitude: Source amplitude
        spatial_sigma: Spatial width
    """

    def __init__(
        self,
        sigma_t: float = 0.01,
        time_delay: float = 0.05,
        source_x: float = 0.0,
        source_z: float = 0.0,
        amplitude: float = 1.0,
        spatial_sigma: float = 1e-6,
    ) -> None:
        self.sigma_t = sigma_t
        self.time_delay = time_delay
        self.source_x = source_x
        self.source_z = source_z
        self.amplitude = amplitude
        self.spatial_sigma = spatial_sigma

    def time_function(self, t: Tensor) -> Tensor:
        """Gaussian time function."""
        tau = t - self.time_delay
        return self.amplitude * torch.exp(-tau**2 / (2.0 * self.sigma_t**2))

    def spatial_function(self, x: Tensor, z: Tensor) -> Tensor:
        """Gaussian spatial distribution."""
        dx = x - self.source_x
        dz = z - self.source_z
        r2 = dx**2 + dz**2
        return torch.exp(-r2 / (2.0 * self.spatial_sigma**2))

    def evaluate(self, x: Tensor, z: Tensor, t: Tensor) -> Tensor:
        """Evaluate full source function."""
        return self.time_function(t) * self.spatial_function(x, z)


class GaussianDerivativeSource(SourceFunction):
    """
    Gaussian First Derivative Source.

    The first derivative of a Gaussian, providing a more impulsive source
    with less low-frequency content than a pure Gaussian:

        g(t) = -A * (t-t₀)/σ² * exp(-(t-t₀)²/(2σ²))

    Args:
        sigma_t: Temporal standard deviation
        time_delay: Time shift
        source_x: x-coordinate of source
        source_z: z-coordinate of source
        amplitude: Source amplitude
        spatial_sigma: Spatial width
    """

    def __init__(
        self,
        sigma_t: float = 0.01,
        time_delay: float = 0.05,
        source_x: float = 0.0,
        source_z: float = 0.0,
        amplitude: float = 1.0,
        spatial_sigma: float = 1e-6,
    ) -> None:
        self.sigma_t = sigma_t
        self.time_delay = time_delay
        self.source_x = source_x
        self.source_z = source_z
        self.amplitude = amplitude
        self.spatial_sigma = spatial_sigma

    def time_function(self, t: Tensor) -> Tensor:
        """Gaussian derivative time function."""
        tau = t - self.time_delay
        gaussian = torch.exp(-tau**2 / (2.0 * self.sigma_t**2))
        return -self.amplitude * tau / self.sigma_t**2 * gaussian

    def spatial_function(self, x: Tensor, z: Tensor) -> Tensor:
        """Gaussian spatial distribution."""
        dx = x - self.source_x
        dz = z - self.source_z
        r2 = dx**2 + dz**2
        return torch.exp(-r2 / (2.0 * self.spatial_sigma**2))

    def evaluate(self, x: Tensor, z: Tensor, t: Tensor) -> Tensor:
        """Evaluate full source function."""
        return self.time_function(t) * self.spatial_function(x, z)


class PointSource(SourceFunction):
    """
    Delta Function Point Source.

    Approximates a point source using a narrow Gaussian:

        S(x, z) ≈ δ(x - x_s) * δ(z - z_s)

    This is a thin wrapper around other source types that
    enforces a very small spatial extent.

    Args:
        wavelet: Time function to use (e.g., RickerWavelet)
        source_x: x-coordinate of source
        source_z: z-coordinate of source
        epsilon: Regularization width for delta approximation
    """

    def __init__(
        self,
        wavelet: SourceFunction,
        source_x: float = 0.0,
        source_z: float = 0.0,
        epsilon: float = 1e-6,
    ) -> None:
        self.wavelet = wavelet
        self.source_x = source_x
        self.source_z = source_z
        self.epsilon = epsilon

        # Override wavelet spatial parameters
        self.wavelet.source_x = source_x
        self.wavelet.source_z = source_z
        self.wavelet.spatial_sigma = epsilon

    def time_function(self, t: Tensor) -> Tensor:
        """Delegate to underlying wavelet."""
        return self.wavelet.time_function(t)

    def evaluate(self, x: Tensor, z: Tensor, t: Tensor) -> Tensor:
        """Evaluate point source."""
        return self.wavelet.evaluate(x, z, t)


class PlaneWaveSource(SourceFunction):
    """
    Plane Wave Source for Illumination from Specific Direction.

    Creates a plane wave traveling at a specified angle:

        f(x, z, t) = g(t - (x*sin(θ) + z*cos(θ))/v)

    where θ is the incidence angle (0 = vertical, 90 = horizontal).

    Useful for:
    - Migration velocity analysis
    - Angle-domain imaging
    - Plane wave decomposition

    Args:
        wavelet: Time function to use
        angle_degrees: Incidence angle in degrees (0 = vertical)
        velocity: Propagation velocity for time shift
        amplitude: Wave amplitude
    """

    def __init__(
        self,
        wavelet: SourceFunction,
        angle_degrees: float = 0.0,
        velocity: float = 2000.0,
        amplitude: float = 1.0,
    ) -> None:
        self.wavelet = wavelet
        self.angle = math.radians(angle_degrees)
        self.velocity = velocity
        self.amplitude = amplitude

        # Direction cosines
        self.px = math.sin(self.angle) / velocity  # Horizontal slowness
        self.pz = math.cos(self.angle) / velocity  # Vertical slowness

    def time_function(self, t: Tensor) -> Tensor:
        """Base time function (without spatial delay)."""
        return self.wavelet.time_function(t)

    def evaluate(self, x: Tensor, z: Tensor, t: Tensor) -> Tensor:
        """
        Evaluate plane wave at each point.

        The wave arrives at different times depending on position:
        t_arrival = px*x + pz*z
        """
        # Time delay based on position
        time_delay = self.px * x + self.pz * z

        # Evaluate wavelet at delayed time
        delayed_t = t - time_delay

        # Use base wavelet time function
        return self.amplitude * self.wavelet.time_function(delayed_t)


class MultipointSource(SourceFunction):
    """
    Multiple Point Sources for Array Simulation.

    Simulates an array of sources firing with specified delays
    (like an air gun array in marine seismic).

    Args:
        source_positions: List of (x, z) source positions
        source_delays: Time delays for each source (for beam steering)
        wavelet: Time function for each source
        amplitudes: Amplitude for each source (optional)
    """

    def __init__(
        self,
        source_positions: list,
        source_delays: Optional[list] = None,
        wavelet: Optional[SourceFunction] = None,
        amplitudes: Optional[list] = None,
        spatial_sigma: float = 1e-6,
    ) -> None:
        self.source_positions = source_positions
        self.n_sources = len(source_positions)

        self.source_delays = source_delays or [0.0] * self.n_sources
        self.amplitudes = amplitudes or [1.0] * self.n_sources
        self.spatial_sigma = spatial_sigma

        # Default wavelet
        if wavelet is None:
            self.wavelet = RickerWavelet()
        else:
            self.wavelet = wavelet

    def time_function(self, t: Tensor) -> Tensor:
        """Aggregate time function (not well-defined for multi-source)."""
        return self.wavelet.time_function(t)

    def evaluate(self, x: Tensor, z: Tensor, t: Tensor) -> Tensor:
        """
        Evaluate all sources and sum contributions.

        Args:
            x: x-coordinates
            z: z-coordinates
            t: time values

        Returns:
            Total source amplitude from all sources
        """
        total = torch.zeros_like(t)

        for i, (sx, sz) in enumerate(self.source_positions):
            # Spatial weight
            dx = x - sx
            dz = z - sz
            r2 = dx**2 + dz**2
            spatial = torch.exp(-r2 / (2.0 * self.spatial_sigma**2))

            # Delayed time function
            delayed_t = t - self.source_delays[i]
            temporal = self.wavelet.time_function(delayed_t)

            # Add contribution
            total = total + self.amplitudes[i] * spatial * temporal

        return total


class LearnableSource(nn.Module, SourceFunction):
    """
    Learnable Source Function.

    Parameterizes the source as a neural network, allowing the
    source signature to be learned during inversion.

    Useful for:
    - Source wavelet estimation
    - Deblending
    - Joint source-velocity inversion

    Args:
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
        max_time: Maximum time for normalization
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        max_time: float = 1.0,
        source_x: float = 0.0,
        source_z: float = 0.0,
        spatial_sigma: float = 1e-6,
    ) -> None:
        super().__init__()

        self.max_time = max_time
        self.source_x = source_x
        self.source_z = source_z
        self.spatial_sigma = spatial_sigma

        # Time function network
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))

        self.time_net = nn.Sequential(*layers)

        # Amplitude scaling
        self.amplitude = nn.Parameter(torch.tensor(1.0))

    def time_function(self, t: Tensor) -> Tensor:
        """Learned time function."""
        # Normalize time to [-1, 1]
        t_norm = 2.0 * t / self.max_time - 1.0

        if t_norm.dim() == 0:
            t_norm = t_norm.unsqueeze(0)
        if t_norm.dim() == 1:
            t_norm = t_norm.unsqueeze(-1)

        return self.amplitude * self.time_net(t_norm).squeeze(-1)

    def spatial_function(self, x: Tensor, z: Tensor) -> Tensor:
        """Gaussian spatial distribution."""
        dx = x - self.source_x
        dz = z - self.source_z
        r2 = dx**2 + dz**2
        return torch.exp(-r2 / (2.0 * self.spatial_sigma**2))

    def evaluate(self, x: Tensor, z: Tensor, t: Tensor) -> Tensor:
        """Evaluate learned source."""
        return self.time_function(t) * self.spatial_function(x, z)

    def forward(self, x: Tensor, z: Tensor, t: Tensor) -> Tensor:
        """Forward method for nn.Module compatibility."""
        return self.evaluate(x, z, t)
