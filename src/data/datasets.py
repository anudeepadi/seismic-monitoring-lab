"""
PyTorch Datasets for PINN Training

Provides efficient data loading for:
- Collocation points (physics loss)
- Observation data (data loss)
- Initial/boundary conditions
- Velocity models

Supports streaming and on-the-fly generation for large datasets.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset
import h5py


class SeismicDataset(Dataset):
    """
    Dataset for seismic observation data.

    Loads pre-computed seismic data (shot gathers) for training.
    Each sample contains receiver coordinates and corresponding
    seismic traces.

    Args:
        data_path: Path to HDF5 file or numpy archive
        transform: Optional transform to apply to data
        normalize: Whether to normalize seismograms
    """

    def __init__(
        self,
        data_path: str,
        transform: Optional[Callable] = None,
        normalize: bool = True,
    ) -> None:
        self.data_path = data_path
        self.transform = transform
        self.normalize = normalize

        # Load data
        self._load_data()

    def _load_data(self) -> None:
        """Load data from file."""
        if self.data_path.endswith(".h5") or self.data_path.endswith(".hdf5"):
            with h5py.File(self.data_path, "r") as f:
                self.seismograms = torch.tensor(f["seismograms"][:], dtype=torch.float32)
                self.receiver_coords = torch.tensor(
                    f["receiver_coords"][:], dtype=torch.float32
                )
                self.source_coords = torch.tensor(
                    f["source_coords"][:], dtype=torch.float32
                )
                self.time_samples = torch.tensor(
                    f["time_samples"][:], dtype=torch.float32
                )
                if "velocity_model" in f:
                    self.velocity_model = torch.tensor(
                        f["velocity_model"][:], dtype=torch.float32
                    )
        else:
            # Numpy archive
            data = np.load(self.data_path, allow_pickle=True)
            self.seismograms = torch.tensor(data["seismograms"], dtype=torch.float32)
            self.receiver_coords = torch.tensor(
                data["receiver_coords"], dtype=torch.float32
            )
            self.source_coords = torch.tensor(data["source_coords"], dtype=torch.float32)
            self.time_samples = torch.tensor(data["time_samples"], dtype=torch.float32)
            if "velocity_model" in data:
                self.velocity_model = torch.tensor(
                    data["velocity_model"], dtype=torch.float32
                )

        if self.normalize:
            # Normalize seismograms
            self.seismograms = self._normalize_seismograms(self.seismograms)

    def _normalize_seismograms(self, data: Tensor) -> Tensor:
        """Normalize seismograms by trace RMS."""
        rms = torch.sqrt(torch.mean(data ** 2, dim=-1, keepdim=True))
        rms = torch.clamp(rms, min=1e-8)
        return data / rms

    def __len__(self) -> int:
        return len(self.seismograms)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        sample = {
            "seismogram": self.seismograms[idx],
            "receiver_coords": self.receiver_coords[idx],
            "source_coords": self.source_coords[idx],
            "time_samples": self.time_samples,
        }

        if hasattr(self, "velocity_model"):
            sample["velocity_model"] = self.velocity_model

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class PINNDataset(Dataset):
    """
    Dataset for Physics-Informed Neural Network training.

    Provides batches containing:
    - Collocation points for PDE loss
    - Observation points with ground truth data
    - Initial condition points
    - Boundary condition points

    Args:
        domain_bounds: Dictionary with 'x', 'z', 't' ranges
        n_collocation: Number of collocation points per batch
        n_observation: Number of observation points per batch
        n_initial: Number of initial condition points per batch
        n_boundary: Number of boundary points per batch
        observation_data: Optional ground truth observation data
        velocity_model: Optional velocity model tensor
        sampling_strategy: Point sampling strategy
    """

    def __init__(
        self,
        domain_bounds: Dict[str, Tuple[float, float]],
        n_collocation: int = 10000,
        n_observation: int = 1000,
        n_initial: int = 2000,
        n_boundary: int = 1000,
        observation_data: Optional[Dict[str, Tensor]] = None,
        velocity_model: Optional[Tensor] = None,
        sampling_strategy: str = "uniform",
        n_batches: int = 100,
    ) -> None:
        self.domain_bounds = domain_bounds
        self.n_collocation = n_collocation
        self.n_observation = n_observation
        self.n_initial = n_initial
        self.n_boundary = n_boundary
        self.observation_data = observation_data
        self.velocity_model = velocity_model
        self.sampling_strategy = sampling_strategy
        self.n_batches = n_batches

        # Extract ranges
        self.x_range = domain_bounds["x"]
        self.z_range = domain_bounds["z"]
        self.t_range = domain_bounds["t"]

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Generate a batch of training data."""
        batch = {}

        # Collocation points for PDE loss
        batch["collocation"] = self._sample_collocation()

        # Initial condition points (t = t_min)
        batch["initial"] = self._sample_initial()

        # Boundary points
        batch["boundary"] = self._sample_boundary()

        # Observation points (if available)
        if self.observation_data is not None:
            batch["observation"] = self._sample_observations()

        # Add velocity model if available
        if self.velocity_model is not None:
            batch["velocity_model"] = self.velocity_model

        return batch

    def _sample_collocation(self) -> Dict[str, Tensor]:
        """Sample collocation points for PDE loss."""
        if self.sampling_strategy == "latin_hypercube":
            coords = self._latin_hypercube_sample(self.n_collocation)
        else:
            coords = self._uniform_sample(self.n_collocation)

        return {"coords": coords}

    def _sample_initial(self) -> Dict[str, Tensor]:
        """Sample initial condition points (t = 0)."""
        n = self.n_initial

        x = torch.rand(n) * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        z = torch.rand(n) * (self.z_range[1] - self.z_range[0]) + self.z_range[0]
        t = torch.full((n,), self.t_range[0])

        coords = torch.stack([x, z, t], dim=-1)

        # Initial conditions: u(x, z, 0) = 0, u_t(x, z, 0) = 0
        u_target = torch.zeros(n, 1)
        u_t_target = torch.zeros(n, 1)

        return {
            "coords": coords,
            "u_target": u_target,
            "u_t_target": u_t_target,
        }

    def _sample_boundary(self) -> Dict[str, Tensor]:
        """Sample boundary points."""
        n_per_boundary = self.n_boundary // 4

        boundaries = []

        # Left boundary (x = x_min)
        z = torch.rand(n_per_boundary) * (self.z_range[1] - self.z_range[0]) + self.z_range[0]
        t = torch.rand(n_per_boundary) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
        x = torch.full((n_per_boundary,), self.x_range[0])
        boundaries.append(torch.stack([x, z, t], dim=-1))

        # Right boundary (x = x_max)
        z = torch.rand(n_per_boundary) * (self.z_range[1] - self.z_range[0]) + self.z_range[0]
        t = torch.rand(n_per_boundary) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
        x = torch.full((n_per_boundary,), self.x_range[1])
        boundaries.append(torch.stack([x, z, t], dim=-1))

        # Top boundary (z = z_min) - often free surface
        x = torch.rand(n_per_boundary) * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        t = torch.rand(n_per_boundary) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
        z = torch.full((n_per_boundary,), self.z_range[0])
        boundaries.append(torch.stack([x, z, t], dim=-1))

        # Bottom boundary (z = z_max)
        x = torch.rand(n_per_boundary) * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        t = torch.rand(n_per_boundary) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
        z = torch.full((n_per_boundary,), self.z_range[1])
        boundaries.append(torch.stack([x, z, t], dim=-1))

        coords = torch.cat(boundaries, dim=0)

        return {"coords": coords}

    def _sample_observations(self) -> Dict[str, Tensor]:
        """Sample observation points with ground truth values."""
        obs = self.observation_data

        # Random subset of observation points
        n_total = len(obs["coords"])
        indices = torch.randperm(n_total)[: self.n_observation]

        return {
            "coords": obs["coords"][indices],
            "values": obs["values"][indices],
        }

    def _uniform_sample(self, n: int) -> Tensor:
        """Uniform random sampling."""
        x = torch.rand(n) * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        z = torch.rand(n) * (self.z_range[1] - self.z_range[0]) + self.z_range[0]
        t = torch.rand(n) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]

        return torch.stack([x, z, t], dim=-1)

    def _latin_hypercube_sample(self, n: int) -> Tensor:
        """Latin Hypercube sampling for better coverage."""
        coords = torch.zeros(n, 3)

        for i, (lo, hi) in enumerate(
            [self.x_range, self.z_range, self.t_range]
        ):
            intervals = torch.linspace(0, 1, n + 1)
            lower = intervals[:-1]
            upper = intervals[1:]
            points = lower + (upper - lower) * torch.rand(n)
            perm = torch.randperm(n)
            coords[:, i] = points[perm] * (hi - lo) + lo

        return coords


class StreamingDataset(IterableDataset):
    """
    Streaming dataset for infinite training data.

    Generates collocation points on-the-fly without storing
    all data in memory. Useful for large-scale training.

    Args:
        domain_bounds: Domain extent
        batch_size: Number of points per iteration
        sampling_strategy: Point sampling method
        velocity_model_fn: Optional function to sample velocity
    """

    def __init__(
        self,
        domain_bounds: Dict[str, Tuple[float, float]],
        batch_size: int = 10000,
        sampling_strategy: str = "uniform",
        velocity_model_fn: Optional[Callable] = None,
        include_source: bool = True,
        source_params: Optional[Dict] = None,
    ) -> None:
        self.domain_bounds = domain_bounds
        self.batch_size = batch_size
        self.sampling_strategy = sampling_strategy
        self.velocity_model_fn = velocity_model_fn
        self.include_source = include_source
        self.source_params = source_params or {
            "x": 0.5,
            "z": 0.05,
            "frequency": 20.0,
        }

        self.x_range = domain_bounds["x"]
        self.z_range = domain_bounds["z"]
        self.t_range = domain_bounds["t"]

    def __iter__(self):
        while True:
            yield self._generate_batch()

    def _generate_batch(self) -> Dict[str, Tensor]:
        """Generate a single batch of training data."""
        batch = {}

        # Collocation points
        coords = self._sample_points(self.batch_size)
        batch["coords"] = coords

        # Velocity at collocation points
        if self.velocity_model_fn is not None:
            batch["velocity"] = self.velocity_model_fn(coords[:, :2])
        else:
            # Default constant velocity
            batch["velocity"] = torch.full((self.batch_size, 1), 2000.0)

        # Source term
        if self.include_source:
            batch["source"] = self._compute_source(coords)

        # Initial condition points
        n_ic = self.batch_size // 10
        ic_coords = self._sample_initial(n_ic)
        batch["initial_coords"] = ic_coords

        # Boundary points
        n_bc = self.batch_size // 20
        bc_coords = self._sample_boundary(n_bc)
        batch["boundary_coords"] = bc_coords

        return batch

    def _sample_points(self, n: int) -> Tensor:
        """Sample collocation points."""
        x = torch.rand(n) * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        z = torch.rand(n) * (self.z_range[1] - self.z_range[0]) + self.z_range[0]
        t = torch.rand(n) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
        return torch.stack([x, z, t], dim=-1)

    def _sample_initial(self, n: int) -> Tensor:
        """Sample initial condition points."""
        x = torch.rand(n) * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        z = torch.rand(n) * (self.z_range[1] - self.z_range[0]) + self.z_range[0]
        t = torch.full((n,), self.t_range[0])
        return torch.stack([x, z, t], dim=-1)

    def _sample_boundary(self, n: int) -> Tensor:
        """Sample boundary points."""
        n_per_side = n // 4

        coords = []
        for _ in range(n_per_side):
            # Random boundary selection
            side = torch.randint(0, 4, (1,)).item()

            if side == 0:  # Left
                x, z = self.x_range[0], torch.rand(1).item() * (
                    self.z_range[1] - self.z_range[0]
                ) + self.z_range[0]
            elif side == 1:  # Right
                x, z = self.x_range[1], torch.rand(1).item() * (
                    self.z_range[1] - self.z_range[0]
                ) + self.z_range[0]
            elif side == 2:  # Top
                x, z = torch.rand(1).item() * (
                    self.x_range[1] - self.x_range[0]
                ) + self.x_range[0], self.z_range[0]
            else:  # Bottom
                x, z = torch.rand(1).item() * (
                    self.x_range[1] - self.x_range[0]
                ) + self.x_range[0], self.z_range[1]

            t = torch.rand(1).item() * (
                self.t_range[1] - self.t_range[0]
            ) + self.t_range[0]
            coords.append([x, z, t])

        return torch.tensor(coords, dtype=torch.float32)

    def _compute_source(self, coords: Tensor) -> Tensor:
        """Compute source term at given coordinates."""
        x = coords[:, 0]
        z = coords[:, 1]
        t = coords[:, 2]

        sx = self.source_params["x"] * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        sz = self.source_params["z"] * (self.z_range[1] - self.z_range[0]) + self.z_range[0]
        f0 = self.source_params["frequency"]

        # Spatial Gaussian
        sigma_s = 0.01 * (self.x_range[1] - self.x_range[0])
        spatial = torch.exp(-((x - sx) ** 2 + (z - sz) ** 2) / (2 * sigma_s ** 2))

        # Ricker wavelet in time
        t0 = 1.5 / f0
        tau = t - t0
        arg = (np.pi * f0 * tau) ** 2
        temporal = (1 - 2 * arg) * torch.exp(-arg)

        return (spatial * temporal).unsqueeze(-1)


class AdaptiveDataset(Dataset):
    """
    Adaptive sampling dataset that focuses on high-residual regions.

    After initial training, samples are concentrated in regions
    where the PDE residual is highest, accelerating convergence.

    Args:
        base_dataset: Underlying dataset
        model: PINN model for computing residuals
        adaptation_frequency: How often to update sampling distribution
    """

    def __init__(
        self,
        base_dataset: PINNDataset,
        model: Optional[torch.nn.Module] = None,
        adaptation_frequency: int = 100,
        residual_power: float = 1.0,
    ) -> None:
        self.base_dataset = base_dataset
        self.model = model
        self.adaptation_frequency = adaptation_frequency
        self.residual_power = residual_power

        # Importance weights for sampling
        self.importance_weights = None
        self.sample_count = 0

    def update_importance(self, coords: Tensor, residuals: Tensor) -> None:
        """
        Update importance sampling weights based on residuals.

        Args:
            coords: Coordinates where residuals were computed
            residuals: PDE residual values
        """
        # Use residual magnitude as importance weight
        weights = torch.abs(residuals).pow(self.residual_power)
        weights = weights / weights.sum()

        self.importance_weights = {
            "coords": coords,
            "weights": weights.squeeze(),
        }

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Get batch with adaptive sampling."""
        self.sample_count += 1

        # Get base batch
        batch = self.base_dataset[idx]

        # If we have importance weights, resample collocation points
        if self.importance_weights is not None and self.sample_count % 2 == 0:
            n = len(batch["collocation"]["coords"])

            # Sample indices according to importance
            indices = torch.multinomial(
                self.importance_weights["weights"],
                n,
                replacement=True,
            )

            # Add small noise for exploration
            sampled_coords = self.importance_weights["coords"][indices]
            noise = torch.randn_like(sampled_coords) * 0.01
            batch["collocation"]["coords"] = sampled_coords + noise

        return batch
