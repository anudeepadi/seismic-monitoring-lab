"""
Data Generation for Physics-Informed Neural Networks

This module provides utilities for generating:
- Synthetic seismic data using finite difference modeling
- Collocation points for physics loss computation
- Training/validation data splits

The generated data is used to train PINNs for waveform inversion.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor
import math


class CollocationPointSampler:
    """
    Samples collocation points for PINN training.

    Collocation points are locations where we enforce the PDE constraint.
    Different sampling strategies can improve training convergence:
    - Uniform: Simple random sampling
    - Latin Hypercube: Better space coverage
    - Residual Adaptive: Focus on high-error regions
    - Importance Sampling: Based on wave energy

    Args:
        x_range: (min, max) for x coordinate
        z_range: (min, max) for z coordinate
        t_range: (min, max) for time
        strategy: Sampling strategy name
    """

    def __init__(
        self,
        x_range: Tuple[float, float],
        z_range: Tuple[float, float],
        t_range: Tuple[float, float],
        strategy: str = "uniform",
    ) -> None:
        self.x_range = x_range
        self.z_range = z_range
        self.t_range = t_range
        self.strategy = strategy

        self.ranges = [x_range, z_range, t_range]
        self.n_dims = 3

    def sample(
        self,
        n_points: int,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Sample collocation points.

        Args:
            n_points: Number of points to sample
            device: Device for tensor

        Returns:
            Tensor of shape (n_points, 3) with (x, z, t) coordinates
        """
        if self.strategy == "uniform":
            return self._uniform_sample(n_points, device)
        elif self.strategy == "latin_hypercube":
            return self._latin_hypercube_sample(n_points, device)
        elif self.strategy == "sobol":
            return self._sobol_sample(n_points, device)
        elif self.strategy == "grid":
            return self._grid_sample(n_points, device)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")

    def _uniform_sample(
        self,
        n_points: int,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Uniform random sampling."""
        samples = torch.rand(n_points, self.n_dims, device=device)

        for i, (lo, hi) in enumerate(self.ranges):
            samples[:, i] = samples[:, i] * (hi - lo) + lo

        return samples

    def _latin_hypercube_sample(
        self,
        n_points: int,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Latin Hypercube Sampling (LHS).

        Divides each dimension into n_points intervals and ensures
        exactly one sample per interval, providing better coverage.
        """
        samples = torch.zeros(n_points, self.n_dims, device=device)

        for i, (lo, hi) in enumerate(self.ranges):
            # Create intervals
            intervals = torch.linspace(0, 1, n_points + 1, device=device)

            # Sample one point per interval
            lower = intervals[:-1]
            upper = intervals[1:]
            points = lower + (upper - lower) * torch.rand(n_points, device=device)

            # Shuffle
            perm = torch.randperm(n_points, device=device)
            samples[:, i] = points[perm] * (hi - lo) + lo

        return samples

    def _sobol_sample(
        self,
        n_points: int,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Sobol sequence sampling (quasi-random).

        Provides very uniform coverage of the domain.
        """
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.n_dims, scramble=True)
            samples_np = sampler.random(n_points)

            samples = torch.tensor(samples_np, dtype=torch.float32, device=device)

            for i, (lo, hi) in enumerate(self.ranges):
                samples[:, i] = samples[:, i] * (hi - lo) + lo

            return samples
        except ImportError:
            # Fall back to Latin Hypercube if scipy not available
            return self._latin_hypercube_sample(n_points, device)

    def _grid_sample(
        self,
        n_points: int,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Regular grid sampling.

        Creates a regular grid, useful for evaluation but not training.
        """
        # Approximate grid size
        n_per_dim = int(n_points ** (1/3))

        x = torch.linspace(self.x_range[0], self.x_range[1], n_per_dim, device=device)
        z = torch.linspace(self.z_range[0], self.z_range[1], n_per_dim, device=device)
        t = torch.linspace(self.t_range[0], self.t_range[1], n_per_dim, device=device)

        xx, zz, tt = torch.meshgrid(x, z, t, indexing="ij")
        samples = torch.stack([xx.flatten(), zz.flatten(), tt.flatten()], dim=-1)

        return samples

    def sample_boundary(
        self,
        n_points: int,
        boundary: str = "all",
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Sample points on domain boundaries.

        Args:
            n_points: Points per boundary
            boundary: Which boundary ('left', 'right', 'top', 'bottom', 'all')
            device: Device for tensor

        Returns:
            Boundary point coordinates
        """
        points_list = []

        x_min, x_max = self.x_range
        z_min, z_max = self.z_range
        t_min, t_max = self.t_range

        if boundary in ["left", "all"]:
            # x = x_min
            z = torch.rand(n_points, device=device) * (z_max - z_min) + z_min
            t = torch.rand(n_points, device=device) * (t_max - t_min) + t_min
            x = torch.full((n_points,), x_min, device=device)
            points_list.append(torch.stack([x, z, t], dim=-1))

        if boundary in ["right", "all"]:
            # x = x_max
            z = torch.rand(n_points, device=device) * (z_max - z_min) + z_min
            t = torch.rand(n_points, device=device) * (t_max - t_min) + t_min
            x = torch.full((n_points,), x_max, device=device)
            points_list.append(torch.stack([x, z, t], dim=-1))

        if boundary in ["top", "all"]:
            # z = z_min
            x = torch.rand(n_points, device=device) * (x_max - x_min) + x_min
            t = torch.rand(n_points, device=device) * (t_max - t_min) + t_min
            z = torch.full((n_points,), z_min, device=device)
            points_list.append(torch.stack([x, z, t], dim=-1))

        if boundary in ["bottom", "all"]:
            # z = z_max
            x = torch.rand(n_points, device=device) * (x_max - x_min) + x_min
            t = torch.rand(n_points, device=device) * (t_max - t_min) + t_min
            z = torch.full((n_points,), z_max, device=device)
            points_list.append(torch.stack([x, z, t], dim=-1))

        return torch.cat(points_list, dim=0)

    def sample_initial(
        self,
        n_points: int,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Sample points at initial time (t = t_min).

        Args:
            n_points: Number of points
            device: Device for tensor

        Returns:
            Initial condition point coordinates
        """
        x = torch.rand(n_points, device=device) * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        z = torch.rand(n_points, device=device) * (self.z_range[1] - self.z_range[0]) + self.z_range[0]
        t = torch.full((n_points,), self.t_range[0], device=device)

        return torch.stack([x, z, t], dim=-1)


class SeismicDataGenerator:
    """
    Generates synthetic seismic data using finite difference modeling.

    This class provides ground truth seismic data for training PINNs.
    It uses the acoustic wave equation solved by finite differences.

    Args:
        velocity_model: 2D velocity model array
        dx: Grid spacing in x (meters)
        dz: Grid spacing in z (meters)
        dt: Time step (seconds)
        nt: Number of time steps
        source_type: Type of source wavelet
        peak_frequency: Peak frequency of source (Hz)
    """

    def __init__(
        self,
        velocity_model: np.ndarray,
        dx: float = 10.0,
        dz: float = 10.0,
        dt: float = 0.001,
        nt: int = 1000,
        source_type: str = "ricker",
        peak_frequency: float = 20.0,
    ) -> None:
        self.velocity = velocity_model
        self.nz, self.nx = velocity_model.shape
        self.dx = dx
        self.dz = dz
        self.dt = dt
        self.nt = nt
        self.source_type = source_type
        self.peak_frequency = peak_frequency

        # Check stability condition (CFL)
        v_max = velocity_model.max()
        cfl = v_max * dt * np.sqrt(1/dx**2 + 1/dz**2)
        if cfl > 1.0:
            raise ValueError(
                f"CFL condition violated: {cfl:.3f} > 1.0. "
                f"Reduce dt or increase dx/dz."
            )

    def _ricker_wavelet(self, t: np.ndarray, t0: float) -> np.ndarray:
        """Generate Ricker wavelet."""
        tau = t - t0
        arg = (np.pi * self.peak_frequency * tau) ** 2
        return (1 - 2 * arg) * np.exp(-arg)

    def generate_shot(
        self,
        source_x: int,
        source_z: int,
        receiver_x: np.ndarray,
        receiver_z: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single shot gather using finite differences.

        Args:
            source_x: Source x-index
            source_z: Source z-index
            receiver_x: Array of receiver x-indices
            receiver_z: Array of receiver z-indices

        Returns:
            Tuple of (wavefield_snapshots, seismograms)
        """
        # Initialize wavefields
        p = np.zeros((self.nz, self.nx), dtype=np.float64)
        p_old = np.zeros_like(p)
        p_new = np.zeros_like(p)

        # Source wavelet
        t = np.arange(self.nt) * self.dt
        t0 = 1.5 / self.peak_frequency  # Delay
        source = self._ricker_wavelet(t, t0)

        # Receivers
        n_receivers = len(receiver_x)
        seismograms = np.zeros((n_receivers, self.nt))

        # Snapshots (store every 10 steps)
        snapshot_interval = 10
        n_snapshots = self.nt // snapshot_interval
        snapshots = np.zeros((n_snapshots, self.nz, self.nx))

        # Precompute coefficient
        v2 = self.velocity ** 2
        coeff_x = (self.dt / self.dx) ** 2
        coeff_z = (self.dt / self.dz) ** 2

        # Time stepping
        for it in range(self.nt):
            # Finite difference stencil (4th order in space)
            for iz in range(2, self.nz - 2):
                for ix in range(2, self.nx - 2):
                    laplacian_x = (
                        -p[iz, ix-2] / 12 +
                        4 * p[iz, ix-1] / 3 -
                        5 * p[iz, ix] / 2 +
                        4 * p[iz, ix+1] / 3 -
                        p[iz, ix+2] / 12
                    )

                    laplacian_z = (
                        -p[iz-2, ix] / 12 +
                        4 * p[iz-1, ix] / 3 -
                        5 * p[iz, ix] / 2 +
                        4 * p[iz+1, ix] / 3 -
                        p[iz+2, ix] / 12
                    )

                    p_new[iz, ix] = (
                        2 * p[iz, ix] - p_old[iz, ix] +
                        v2[iz, ix] * (
                            coeff_x * laplacian_x +
                            coeff_z * laplacian_z
                        )
                    )

            # Add source
            p_new[source_z, source_x] += source[it]

            # Record seismograms
            for ir, (rx, rz) in enumerate(zip(receiver_x, receiver_z)):
                seismograms[ir, it] = p[rz, rx]

            # Store snapshot
            if it % snapshot_interval == 0:
                snapshots[it // snapshot_interval] = p.copy()

            # Swap arrays
            p_old, p, p_new = p, p_new, p_old

        return snapshots, seismograms

    def generate_dataset(
        self,
        n_shots: int,
        n_receivers: int,
        source_depth: int = 5,
        receiver_depth: int = 5,
    ) -> Dict[str, np.ndarray]:
        """
        Generate a complete seismic dataset with multiple shots.

        Args:
            n_shots: Number of shots
            n_receivers: Number of receivers per shot
            source_depth: Depth of sources (z-index)
            receiver_depth: Depth of receivers (z-index)

        Returns:
            Dictionary with seismic data and metadata
        """
        # Source and receiver positions
        source_x_positions = np.linspace(
            self.nx // 10, 9 * self.nx // 10, n_shots, dtype=int
        )

        receiver_x_positions = np.linspace(
            0, self.nx - 1, n_receivers, dtype=int
        )

        all_seismograms = []
        all_snapshots = []

        for i, sx in enumerate(source_x_positions):
            print(f"Generating shot {i+1}/{n_shots}...")

            receiver_z = np.full(n_receivers, receiver_depth, dtype=int)

            snapshots, seismograms = self.generate_shot(
                source_x=sx,
                source_z=source_depth,
                receiver_x=receiver_x_positions,
                receiver_z=receiver_z,
            )

            all_seismograms.append(seismograms)
            all_snapshots.append(snapshots)

        return {
            "seismograms": np.stack(all_seismograms),  # (n_shots, n_receivers, nt)
            "snapshots": np.stack(all_snapshots),  # (n_shots, n_snapshots, nz, nx)
            "velocity_model": self.velocity,
            "source_x": source_x_positions,
            "source_z": np.full(n_shots, source_depth),
            "receiver_x": receiver_x_positions,
            "receiver_z": np.full(n_receivers, receiver_depth),
            "dt": self.dt,
            "dx": self.dx,
            "dz": self.dz,
            "nt": self.nt,
            "peak_frequency": self.peak_frequency,
        }


class VelocityModelGenerator:
    """
    Generates synthetic velocity models for training.

    Provides various geological velocity structures:
    - Layered models
    - Gradient models
    - Random heterogeneous models
    - Realistic structures (salt domes, faults, etc.)

    Args:
        nx: Number of grid points in x
        nz: Number of grid points in z
        dx: Grid spacing in x (meters)
        dz: Grid spacing in z (meters)
    """

    def __init__(
        self,
        nx: int = 200,
        nz: int = 100,
        dx: float = 10.0,
        dz: float = 10.0,
    ) -> None:
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz

        # Physical dimensions
        self.width = nx * dx
        self.depth = nz * dz

    def constant(self, velocity: float = 2000.0) -> np.ndarray:
        """Generate constant velocity model."""
        return np.full((self.nz, self.nx), velocity, dtype=np.float32)

    def linear_gradient(
        self,
        v_top: float = 1500.0,
        v_bottom: float = 4000.0,
    ) -> np.ndarray:
        """
        Generate linear velocity gradient.

        Common in sedimentary basins where velocity increases with depth.
        """
        z = np.linspace(0, 1, self.nz)
        v = v_top + (v_bottom - v_top) * z
        return np.tile(v[:, np.newaxis], (1, self.nx)).astype(np.float32)

    def layered(
        self,
        layer_depths: List[float],
        layer_velocities: List[float],
    ) -> np.ndarray:
        """
        Generate horizontally layered model.

        Args:
            layer_depths: Depth of layer interfaces (normalized 0-1)
            layer_velocities: Velocity in each layer

        Returns:
            Layered velocity model
        """
        model = np.zeros((self.nz, self.nx), dtype=np.float32)

        depths = [0.0] + layer_depths + [1.0]

        for i in range(len(layer_velocities)):
            z_start = int(depths[i] * self.nz)
            z_end = int(depths[i + 1] * self.nz)
            model[z_start:z_end, :] = layer_velocities[i]

        return model

    def gaussian_anomaly(
        self,
        background: float = 2500.0,
        anomaly_velocity: float = 3500.0,
        center_x: float = 0.5,
        center_z: float = 0.5,
        sigma_x: float = 0.1,
        sigma_z: float = 0.1,
    ) -> np.ndarray:
        """
        Generate model with Gaussian velocity anomaly.

        Useful for simulating localized velocity variations.
        """
        x = np.linspace(0, 1, self.nx)
        z = np.linspace(0, 1, self.nz)
        xx, zz = np.meshgrid(x, z)

        gaussian = np.exp(
            -((xx - center_x)**2 / (2 * sigma_x**2)) -
            ((zz - center_z)**2 / (2 * sigma_z**2))
        )

        model = background + (anomaly_velocity - background) * gaussian

        return model.astype(np.float32)

    def random_smooth(
        self,
        v_min: float = 2000.0,
        v_max: float = 4000.0,
        smoothness: float = 20.0,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate smooth random velocity model.

        Uses Gaussian filtering on random noise to create
        smooth, realistic-looking velocity variations.
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate random noise
        noise = np.random.randn(self.nz, self.nx)

        # Smooth with Gaussian filter
        from scipy.ndimage import gaussian_filter
        smooth_noise = gaussian_filter(noise, sigma=smoothness)

        # Normalize and scale to velocity range
        smooth_noise = (smooth_noise - smooth_noise.min()) / (
            smooth_noise.max() - smooth_noise.min()
        )
        model = v_min + (v_max - v_min) * smooth_noise

        return model.astype(np.float32)

    def salt_dome(
        self,
        background: float = 2500.0,
        salt_velocity: float = 4500.0,
        dome_center_x: float = 0.5,
        dome_top: float = 0.3,
        dome_width: float = 0.2,
        dome_height: float = 0.5,
    ) -> np.ndarray:
        """
        Generate salt dome model.

        Salt domes are important geological structures in petroleum exploration,
        characterized by high velocity contrast with surrounding sediments.
        """
        model = np.full((self.nz, self.nx), background, dtype=np.float32)

        x = np.linspace(0, 1, self.nx)
        z = np.linspace(0, 1, self.nz)

        for iz in range(self.nz):
            # Salt dome shape (parabolic)
            z_rel = (z[iz] - dome_top) / dome_height
            if z_rel > 0 and z_rel < 1:
                # Width varies with depth
                width_at_z = dome_width * (1 - np.sqrt(1 - z_rel))

                for ix in range(self.nx):
                    if abs(x[ix] - dome_center_x) < width_at_z:
                        model[iz, ix] = salt_velocity

        return model

    def fault(
        self,
        v_left: float = 2500.0,
        v_right: float = 3000.0,
        fault_x: float = 0.5,
        fault_dip: float = 60.0,
        throw: float = 0.1,
    ) -> np.ndarray:
        """
        Generate faulted velocity model.

        Creates a normal fault with specified dip angle and throw.
        """
        model = np.zeros((self.nz, self.nx), dtype=np.float32)

        x = np.linspace(0, 1, self.nx)
        z = np.linspace(0, 1, self.nz)

        dip_rad = np.radians(fault_dip)

        for iz in range(self.nz):
            for ix in range(self.nx):
                # Fault plane position
                fault_at_z = fault_x + z[iz] / np.tan(dip_rad)

                if x[ix] < fault_at_z:
                    model[iz, ix] = v_left
                else:
                    # Shifted layers on right side
                    model[iz, ix] = v_right

        return model

    def checkerboard(
        self,
        v_background: float = 2500.0,
        v_perturbation: float = 500.0,
        n_cells_x: int = 4,
        n_cells_z: int = 4,
    ) -> np.ndarray:
        """
        Generate checkerboard velocity model.

        Useful for resolution analysis in tomography.
        """
        x = np.arange(self.nx)
        z = np.arange(self.nz)
        xx, zz = np.meshgrid(x, z)

        cell_size_x = self.nx / n_cells_x
        cell_size_z = self.nz / n_cells_z

        pattern = np.sin(np.pi * xx / cell_size_x) * np.sin(np.pi * zz / cell_size_z)

        model = v_background + v_perturbation * pattern

        return model.astype(np.float32)

    def to_torch(
        self,
        model: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Convert numpy velocity model to torch tensor."""
        return torch.tensor(model, dtype=torch.float32, device=device)

    def get_coordinates(
        self,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get coordinate grids for the velocity model.

        Returns:
            Tuple of (x_coords, z_coords) tensors
        """
        x = torch.linspace(0, self.width, self.nx, device=device)
        z = torch.linspace(0, self.depth, self.nz, device=device)

        return x, z
