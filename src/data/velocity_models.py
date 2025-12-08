"""
Pre-defined Velocity Models for Seismic Inversion

This module provides commonly used velocity models in geophysics:
- Marmousi: Industry standard benchmark
- SEG/EAGE Salt Model: Complex salt structures
- Overthrust Model: Thrust fault structures

These models are used for benchmarking inversion algorithms.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter, zoom


class VelocityModel(ABC):
    """Abstract base class for velocity models."""

    @abstractmethod
    def generate(self) -> np.ndarray:
        """Generate the velocity model."""
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        """Model dimensions (nz, nx)."""
        pass

    @property
    @abstractmethod
    def spacing(self) -> Tuple[float, float]:
        """Grid spacing (dz, dx) in meters."""
        pass


class LayeredVelocityModel(VelocityModel):
    """
    Simple horizontally layered velocity model.

    Useful for basic testing and understanding wave propagation
    through layer interfaces.

    Args:
        nx: Number of grid points in x
        nz: Number of grid points in z
        dx: Grid spacing in x (meters)
        dz: Grid spacing in z (meters)
        velocities: List of layer velocities (m/s)
        interfaces: List of interface depths (normalized 0-1)
    """

    def __init__(
        self,
        nx: int = 200,
        nz: int = 100,
        dx: float = 10.0,
        dz: float = 10.0,
        velocities: Optional[list] = None,
        interfaces: Optional[list] = None,
    ) -> None:
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz

        self.velocities = velocities or [1500, 2000, 2500, 3000, 3500]
        self.interfaces = interfaces or [0.2, 0.4, 0.6, 0.8]

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.nz, self.nx)

    @property
    def spacing(self) -> Tuple[float, float]:
        return (self.dz, self.dx)

    def generate(self) -> np.ndarray:
        """Generate layered velocity model."""
        model = np.zeros((self.nz, self.nx), dtype=np.float32)

        interfaces = [0.0] + self.interfaces + [1.0]

        for i, v in enumerate(self.velocities):
            z_start = int(interfaces[i] * self.nz)
            z_end = int(interfaces[i + 1] * self.nz)
            model[z_start:z_end, :] = v

        return model


class MarmousiModel(VelocityModel):
    """
    Marmousi Velocity Model (Synthetic Version).

    The Marmousi model is an industry-standard benchmark based on
    a profile through the North Quenguela trough in the Cuanza basin
    of Angola. It features:
    - Complex fold and fault structures
    - Strong lateral velocity variations
    - Multiple reflectors

    This generates a synthetic approximation since the original
    requires licensed data.

    Args:
        nx: Number of grid points in x (default: 461)
        nz: Number of grid points in z (default: 151)
        dx: Grid spacing (default: 25 m)
        dz: Grid spacing (default: 25 m)
        version: 'simple' for basic structure, 'complex' for detailed
    """

    def __init__(
        self,
        nx: int = 461,
        nz: int = 151,
        dx: float = 25.0,
        dz: float = 25.0,
        version: str = "complex",
    ) -> None:
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.version = version

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.nz, self.nx)

    @property
    def spacing(self) -> Tuple[float, float]:
        return (self.dz, self.dx)

    def generate(self) -> np.ndarray:
        """Generate Marmousi-like velocity model."""
        np.random.seed(42)  # Reproducibility

        x = np.linspace(0, 1, self.nx)
        z = np.linspace(0, 1, self.nz)
        xx, zz = np.meshgrid(x, z)

        # Base gradient (velocity increases with depth)
        base = 1500 + 2500 * zz

        # Add folded layers
        n_layers = 8
        layers = np.zeros_like(base)

        for i in range(n_layers):
            # Layer with lateral variation (folds)
            depth = 0.1 + 0.1 * i
            amplitude = 0.03 * (1 + 0.5 * np.sin(2 * np.pi * x * (2 + i * 0.3)))
            wavelength = 0.3 + 0.1 * np.random.rand()

            fold = amplitude * np.sin(2 * np.pi * x / wavelength + i * 0.5)
            layer_boundary = depth + fold

            velocity_contrast = 200 + 100 * i
            layers += velocity_contrast * (zz > np.tile(layer_boundary, (self.nz, 1)))

        model = base + layers

        if self.version == "complex":
            # Add fault
            fault_x = 0.3
            fault_dip = 70  # degrees
            dip_rad = np.radians(fault_dip)

            for iz in range(self.nz):
                fault_at_z = fault_x + z[iz] / np.tan(dip_rad)
                for ix in range(self.nx):
                    if x[ix] < fault_at_z and z[iz] > 0.3:
                        # Offset layers on left side
                        model[iz, ix] = model[min(iz + 10, self.nz - 1), ix]

            # Add anticline (oil trap structure)
            anticline_center = 0.7
            anticline_width = 0.15
            anticline_height = 0.08

            anticline_mask = (np.abs(xx - anticline_center) < anticline_width) & (zz > 0.4)
            uplift = anticline_height * np.cos(
                np.pi * (xx - anticline_center) / (2 * anticline_width)
            ) ** 2
            uplift = np.clip(uplift, 0, anticline_height)

            for iz in range(self.nz):
                for ix in range(self.nx):
                    if anticline_mask[iz, ix]:
                        new_z = int(iz + uplift[iz, ix] * self.nz)
                        if new_z < self.nz:
                            model[iz, ix] = model[new_z, ix]

            # Add small-scale heterogeneity
            noise = np.random.randn(self.nz, self.nx) * 50
            noise = gaussian_filter(noise, sigma=3)
            model += noise

        # Smooth model slightly
        model = gaussian_filter(model, sigma=1)

        # Ensure physical bounds
        model = np.clip(model, 1500, 5500)

        return model.astype(np.float32)


class RandomGaussianModel(VelocityModel):
    """
    Random Gaussian Velocity Model.

    Generates smooth random velocity variations useful for
    testing inversion algorithms with unknown ground truth.

    Args:
        nx: Number of grid points in x
        nz: Number of grid points in z
        dx: Grid spacing in x
        dz: Grid spacing in z
        v_mean: Mean velocity (m/s)
        v_std: Velocity standard deviation (m/s)
        correlation_length: Spatial correlation length (grid points)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        nx: int = 200,
        nz: int = 100,
        dx: float = 10.0,
        dz: float = 10.0,
        v_mean: float = 2500.0,
        v_std: float = 500.0,
        correlation_length: float = 20.0,
        seed: Optional[int] = None,
    ) -> None:
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.v_mean = v_mean
        self.v_std = v_std
        self.correlation_length = correlation_length
        self.seed = seed

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.nz, self.nx)

    @property
    def spacing(self) -> Tuple[float, float]:
        return (self.dz, self.dx)

    def generate(self) -> np.ndarray:
        """Generate smooth random velocity model."""
        if self.seed is not None:
            np.random.seed(self.seed)

        # Generate white noise
        noise = np.random.randn(self.nz, self.nx)

        # Smooth to create correlated field
        smooth = gaussian_filter(noise, sigma=self.correlation_length)

        # Normalize
        smooth = (smooth - smooth.mean()) / smooth.std()

        # Scale to velocity
        model = self.v_mean + self.v_std * smooth

        # Add depth-dependent trend
        z = np.linspace(0, 1, self.nz)
        gradient = 500 * z[:, np.newaxis]
        model += gradient

        # Ensure physical bounds
        model = np.clip(model, 1500, 5000)

        return model.astype(np.float32)


class SaltDomeModel(VelocityModel):
    """
    Salt Dome Velocity Model.

    Salt structures are challenging for seismic imaging due to:
    - High velocity contrast with surrounding sediments
    - Complex geometry
    - Strong multiple reflections

    Args:
        nx: Number of grid points in x
        nz: Number of grid points in z
        dx: Grid spacing in x
        dz: Grid spacing in z
        salt_velocity: Velocity of salt (m/s)
        background_velocity: Background sediment velocity (m/s)
        dome_center: Horizontal position of dome center (normalized)
        dome_width: Width of dome at base (normalized)
        dome_height: Height of dome (normalized)
    """

    def __init__(
        self,
        nx: int = 200,
        nz: int = 150,
        dx: float = 10.0,
        dz: float = 10.0,
        salt_velocity: float = 4500.0,
        background_velocity: float = 2500.0,
        dome_center: float = 0.5,
        dome_width: float = 0.25,
        dome_height: float = 0.6,
    ) -> None:
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.salt_velocity = salt_velocity
        self.background_velocity = background_velocity
        self.dome_center = dome_center
        self.dome_width = dome_width
        self.dome_height = dome_height

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.nz, self.nx)

    @property
    def spacing(self) -> Tuple[float, float]:
        return (self.dz, self.dx)

    def generate(self) -> np.ndarray:
        """Generate salt dome velocity model."""
        # Background with gradient
        model = np.zeros((self.nz, self.nx), dtype=np.float32)

        x = np.linspace(0, 1, self.nx)
        z = np.linspace(0, 1, self.nz)
        xx, zz = np.meshgrid(x, z)

        # Background velocity with depth gradient
        model = self.background_velocity + 1000 * zz

        # Create salt dome geometry
        dome_base = 1.0 - self.dome_height  # Top of salt

        for iz in range(self.nz):
            z_pos = z[iz]

            if z_pos > dome_base:
                # Dome profile (mushroom shape)
                depth_in_dome = (z_pos - dome_base) / self.dome_height
                stem_width = self.dome_width * 0.4 * (1 - depth_in_dome * 0.5)
                cap_width = self.dome_width * (0.8 + 0.2 * np.sin(np.pi * depth_in_dome))

                for ix in range(self.nx):
                    dist_from_center = abs(x[ix] - self.dome_center)

                    # Stem region
                    if dist_from_center < stem_width:
                        model[iz, ix] = self.salt_velocity
                    # Cap region (top 30% of dome)
                    elif depth_in_dome < 0.3 and dist_from_center < cap_width:
                        model[iz, ix] = self.salt_velocity

        # Add flanking structures (upturned beds)
        for side in [-1, 1]:
            flank_x = self.dome_center + side * self.dome_width * 0.6
            for iz in range(self.nz):
                for ix in range(self.nx):
                    dist = abs(x[ix] - flank_x)
                    if dist < 0.05 and z[iz] > dome_base:
                        # Upturn near salt
                        upturn = 200 * (1 - dist / 0.05)
                        model[iz, ix] = min(
                            model[iz, ix] + upturn,
                            self.salt_velocity - 500,
                        )

        # Smooth slightly
        model = gaussian_filter(model, sigma=1)

        return model.astype(np.float32)


class FaultModel(VelocityModel):
    """
    Faulted Velocity Model.

    Creates models with fault structures including:
    - Normal faults
    - Reverse faults
    - Multiple fault systems

    Args:
        nx: Number of grid points in x
        nz: Number of grid points in z
        dx: Grid spacing in x
        dz: Grid spacing in z
        fault_type: 'normal', 'reverse', or 'strike_slip'
        fault_dip: Dip angle in degrees
        throw: Fault throw (vertical offset) normalized
        n_layers: Number of velocity layers
    """

    def __init__(
        self,
        nx: int = 200,
        nz: int = 100,
        dx: float = 10.0,
        dz: float = 10.0,
        fault_type: str = "normal",
        fault_dip: float = 60.0,
        throw: float = 0.15,
        n_layers: int = 5,
    ) -> None:
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.fault_type = fault_type
        self.fault_dip = fault_dip
        self.throw = throw
        self.n_layers = n_layers

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.nz, self.nx)

    @property
    def spacing(self) -> Tuple[float, float]:
        return (self.dz, self.dx)

    def generate(self) -> np.ndarray:
        """Generate faulted velocity model."""
        x = np.linspace(0, 1, self.nx)
        z = np.linspace(0, 1, self.nz)
        xx, zz = np.meshgrid(x, z)

        # Create layered model
        velocities = np.linspace(2000, 4000, self.n_layers)
        layer_thickness = 1.0 / self.n_layers

        model = np.zeros((self.nz, self.nx), dtype=np.float32)

        for i, v in enumerate(velocities):
            mask = (zz >= i * layer_thickness) & (zz < (i + 1) * layer_thickness)
            model[mask] = v

        # Apply fault offset
        fault_x = 0.5
        dip_rad = np.radians(self.fault_dip)

        throw_pixels = int(self.throw * self.nz)

        for iz in range(self.nz):
            fault_at_z = fault_x + z[iz] / np.tan(dip_rad) * 0.3

            for ix in range(self.nx):
                if x[ix] > fault_at_z:
                    if self.fault_type == "normal":
                        # Hanging wall drops down
                        source_iz = min(iz + throw_pixels, self.nz - 1)
                    else:  # reverse
                        # Hanging wall moves up
                        source_iz = max(iz - throw_pixels, 0)

                    model[iz, ix] = model[source_iz, ix]

        # Add fault zone (lower velocity)
        for iz in range(self.nz):
            fault_at_z = fault_x + z[iz] / np.tan(dip_rad) * 0.3

            for ix in range(self.nx):
                dist = abs(x[ix] - fault_at_z)
                if dist < 0.02:
                    # Fault damage zone
                    model[iz, ix] *= 0.9

        # Smooth slightly
        model = gaussian_filter(model, sigma=1)

        return model.astype(np.float32)


def load_segy_velocity(filepath: str, nx: int, nz: int) -> np.ndarray:
    """
    Load velocity model from SEG-Y file.

    Args:
        filepath: Path to SEG-Y file
        nx: Expected number of traces
        nz: Expected samples per trace

    Returns:
        Velocity model array
    """
    try:
        import segyio

        with segyio.open(filepath, "r") as f:
            data = np.array([f.trace[i] for i in range(f.tracecount)])

        # Reshape to 2D
        if data.shape[0] == nx and data.shape[1] == nz:
            return data.T.astype(np.float32)
        else:
            # Interpolate to target size
            return zoom(data.T, (nz / data.shape[1], nx / data.shape[0])).astype(
                np.float32
            )
    except ImportError:
        raise ImportError("segyio required for SEG-Y support: pip install segyio")
