"""
Model Checkpointing and Experiment Tracking Utilities.

This module provides comprehensive checkpoint management, model versioning,
and experiment tracking capabilities for PINN training.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
from dataclasses import dataclass, field, asdict
import hashlib

import torch
import torch.nn as nn


@dataclass
class CheckpointMetadata:
    """Metadata for a model checkpoint."""

    checkpoint_id: str
    model_name: str
    model_type: str
    epoch: int
    step: int

    # Training metrics
    total_loss: float
    physics_loss: float
    data_loss: float
    boundary_loss: float
    learning_rate: float

    # Model architecture
    hidden_dim: int
    num_layers: int
    total_params: int

    # Training configuration
    batch_size: int
    optimizer: str
    scheduler: str
    velocity_model: str
    wave_equation: str

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    training_time_seconds: float = 0.0

    # Additional info
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointMetadata":
        """Create from dictionary."""
        return cls(**data)


class CheckpointManager:
    """Manages model checkpoints with versioning and cleanup."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_checkpoints: int = 10,
        keep_best: int = 3,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            keep_best: Number of best checkpoints to always keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best

        # Track checkpoints
        self.checkpoints: list[dict] = []
        self._load_checkpoint_index()

    def _load_checkpoint_index(self) -> None:
        """Load existing checkpoint index."""
        index_path = self.checkpoint_dir / "checkpoint_index.json"
        if index_path.exists():
            with open(index_path, "r") as f:
                self.checkpoints = json.load(f)
        else:
            self.checkpoints = []

    def _save_checkpoint_index(self) -> None:
        """Save checkpoint index to disk."""
        index_path = self.checkpoint_dir / "checkpoint_index.json"
        with open(index_path, "w") as f:
            json.dump(self.checkpoints, f, indent=2)

    def _generate_checkpoint_id(self, model_name: str, epoch: int) -> str:
        """Generate unique checkpoint ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{model_name}_{epoch}_{timestamp}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"{model_name}_{epoch}_{hash_suffix}"

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        metadata: CheckpointMetadata,
        is_best: bool = False,
    ) -> Path:
        """
        Save model checkpoint.

        Args:
            model: PyTorch model
            optimizer: Optimizer state
            scheduler: Learning rate scheduler (optional)
            metadata: Checkpoint metadata
            is_best: Whether this is the best checkpoint so far

        Returns:
            Path to saved checkpoint
        """
        checkpoint_id = metadata.checkpoint_id
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"

        # Prepare checkpoint data
        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metadata": metadata.to_dict(),
        }

        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)

        # Save metadata separately as JSON for easy inspection
        metadata_path = self.checkpoint_dir / f"{checkpoint_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Update index
        checkpoint_info = {
            "id": checkpoint_id,
            "path": str(checkpoint_path),
            "metadata_path": str(metadata_path),
            "epoch": metadata.epoch,
            "loss": metadata.total_loss,
            "is_best": is_best,
            "created_at": metadata.created_at,
        }
        self.checkpoints.append(checkpoint_info)

        # Save best checkpoint copy
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            shutil.copy(checkpoint_path, best_path)

        # Save latest checkpoint copy
        latest_path = self.checkpoint_dir / "latest_model.pt"
        shutil.copy(checkpoint_path, latest_path)

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

        # Save index
        self._save_checkpoint_index()

        return checkpoint_path

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints, keeping best and most recent."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return

        # Sort by loss to find best
        sorted_by_loss = sorted(self.checkpoints, key=lambda x: x["loss"])
        best_ids = {c["id"] for c in sorted_by_loss[:self.keep_best]}

        # Sort by time to find most recent
        sorted_by_time = sorted(
            self.checkpoints,
            key=lambda x: x["created_at"],
            reverse=True
        )
        recent_ids = {c["id"] for c in sorted_by_time[:self.max_checkpoints - self.keep_best]}

        # IDs to keep
        keep_ids = best_ids | recent_ids

        # Remove old checkpoints
        checkpoints_to_remove = [
            c for c in self.checkpoints if c["id"] not in keep_ids
        ]

        for checkpoint in checkpoints_to_remove:
            # Remove files
            checkpoint_path = Path(checkpoint["path"])
            metadata_path = Path(checkpoint["metadata_path"])

            if checkpoint_path.exists():
                checkpoint_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()

            # Remove from index
            self.checkpoints.remove(checkpoint)

    def load_checkpoint(
        self,
        checkpoint_id: Optional[str] = None,
        load_best: bool = False,
        load_latest: bool = False,
    ) -> dict:
        """
        Load a checkpoint.

        Args:
            checkpoint_id: Specific checkpoint ID to load
            load_best: Load the best checkpoint
            load_latest: Load the latest checkpoint

        Returns:
            Checkpoint data dictionary
        """
        if load_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
        elif load_latest:
            checkpoint_path = self.checkpoint_dir / "latest_model.pt"
        elif checkpoint_id:
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
        else:
            raise ValueError("Must specify checkpoint_id, load_best, or load_latest")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        return torch.load(checkpoint_path, map_location="cpu")

    def list_checkpoints(self) -> list[dict]:
        """List all available checkpoints."""
        return self.checkpoints.copy()

    def get_best_checkpoint(self) -> Optional[dict]:
        """Get info about the best checkpoint."""
        if not self.checkpoints:
            return None
        return min(self.checkpoints, key=lambda x: x["loss"])

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint."""
        for checkpoint in self.checkpoints:
            if checkpoint["id"] == checkpoint_id:
                # Remove files
                Path(checkpoint["path"]).unlink(missing_ok=True)
                Path(checkpoint["metadata_path"]).unlink(missing_ok=True)

                # Remove from index
                self.checkpoints.remove(checkpoint)
                self._save_checkpoint_index()
                return True
        return False


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    name: str
    description: str = ""

    # Model config
    model_type: str = "siren"
    hidden_dim: int = 256
    num_layers: int = 6

    # Training config
    learning_rate: float = 1e-4
    batch_size: int = 4096
    num_epochs: int = 1000

    # Physics config
    physics_weight: float = 1.0
    data_weight: float = 1.0
    boundary_weight: float = 0.1

    # Domain config
    velocity_model: str = "marmousi"
    wave_equation: str = "acoustic"

    # Tags for organization
    tags: list[str] = field(default_factory=list)


@dataclass
class ExperimentMetrics:
    """Metrics tracked during an experiment."""

    epoch: int
    step: int
    total_loss: float
    physics_loss: float
    data_loss: float
    boundary_loss: float
    learning_rate: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    additional_metrics: dict = field(default_factory=dict)


class ExperimentTracker:
    """Tracks experiments and their metrics."""

    def __init__(self, experiments_dir: str | Path):
        """
        Initialize experiment tracker.

        Args:
            experiments_dir: Directory to store experiment data
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        self.current_experiment: Optional[str] = None
        self.metrics_buffer: list[dict] = []
        self._load_experiments_index()

    def _load_experiments_index(self) -> None:
        """Load experiments index."""
        index_path = self.experiments_dir / "experiments_index.json"
        if index_path.exists():
            with open(index_path, "r") as f:
                self.experiments = json.load(f)
        else:
            self.experiments = {}

    def _save_experiments_index(self) -> None:
        """Save experiments index."""
        index_path = self.experiments_dir / "experiments_index.json"
        with open(index_path, "w") as f:
            json.dump(self.experiments, f, indent=2)

    def create_experiment(self, config: ExperimentConfig) -> str:
        """
        Create a new experiment.

        Args:
            config: Experiment configuration

        Returns:
            Experiment ID
        """
        # Generate experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"{config.name}_{timestamp}"

        # Create experiment directory
        exp_dir = self.experiments_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = exp_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(config), f, indent=2)

        # Initialize metrics file
        metrics_path = exp_dir / "metrics.jsonl"
        metrics_path.touch()

        # Update index
        self.experiments[exp_id] = {
            "id": exp_id,
            "name": config.name,
            "description": config.description,
            "status": "running",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "config_path": str(config_path),
            "metrics_path": str(metrics_path),
            "tags": config.tags,
        }
        self._save_experiments_index()

        self.current_experiment = exp_id
        return exp_id

    def log_metrics(self, metrics: ExperimentMetrics) -> None:
        """
        Log metrics for current experiment.

        Args:
            metrics: Metrics to log
        """
        if not self.current_experiment:
            raise RuntimeError("No active experiment. Call create_experiment first.")

        exp_info = self.experiments[self.current_experiment]
        metrics_path = Path(exp_info["metrics_path"])

        # Append to metrics file
        with open(metrics_path, "a") as f:
            f.write(json.dumps(asdict(metrics)) + "\n")

        # Update experiment timestamp
        exp_info["updated_at"] = datetime.now().isoformat()
        self._save_experiments_index()

    def log_metrics_batch(self, metrics_list: list[ExperimentMetrics]) -> None:
        """Log multiple metrics at once."""
        if not self.current_experiment:
            raise RuntimeError("No active experiment. Call create_experiment first.")

        exp_info = self.experiments[self.current_experiment]
        metrics_path = Path(exp_info["metrics_path"])

        with open(metrics_path, "a") as f:
            for metrics in metrics_list:
                f.write(json.dumps(asdict(metrics)) + "\n")

        exp_info["updated_at"] = datetime.now().isoformat()
        self._save_experiments_index()

    def finish_experiment(
        self,
        status: str = "completed",
        final_metrics: Optional[dict] = None,
    ) -> None:
        """
        Mark experiment as finished.

        Args:
            status: Final status (completed, failed, cancelled)
            final_metrics: Optional final metrics summary
        """
        if not self.current_experiment:
            return

        exp_info = self.experiments[self.current_experiment]
        exp_info["status"] = status
        exp_info["finished_at"] = datetime.now().isoformat()

        if final_metrics:
            exp_info["final_metrics"] = final_metrics

        self._save_experiments_index()
        self.current_experiment = None

    def get_experiment(self, exp_id: str) -> Optional[dict]:
        """Get experiment info by ID."""
        return self.experiments.get(exp_id)

    def list_experiments(
        self,
        status: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        List experiments with optional filters.

        Args:
            status: Filter by status
            tags: Filter by tags (experiments must have all tags)

        Returns:
            List of experiment info dictionaries
        """
        results = list(self.experiments.values())

        if status:
            results = [e for e in results if e["status"] == status]

        if tags:
            results = [
                e for e in results
                if all(tag in e.get("tags", []) for tag in tags)
            ]

        return sorted(results, key=lambda x: x["created_at"], reverse=True)

    def load_metrics(self, exp_id: str) -> list[dict]:
        """
        Load all metrics for an experiment.

        Args:
            exp_id: Experiment ID

        Returns:
            List of metrics dictionaries
        """
        exp_info = self.experiments.get(exp_id)
        if not exp_info:
            raise ValueError(f"Experiment not found: {exp_id}")

        metrics_path = Path(exp_info["metrics_path"])
        if not metrics_path.exists():
            return []

        metrics = []
        with open(metrics_path, "r") as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))

        return metrics

    def compare_experiments(
        self,
        exp_ids: list[str],
        metric_name: str = "total_loss",
    ) -> dict:
        """
        Compare metrics across experiments.

        Args:
            exp_ids: List of experiment IDs to compare
            metric_name: Name of metric to compare

        Returns:
            Dictionary with comparison data
        """
        comparison = {}

        for exp_id in exp_ids:
            metrics = self.load_metrics(exp_id)
            if metrics:
                comparison[exp_id] = {
                    "values": [m.get(metric_name, m.get("additional_metrics", {}).get(metric_name)) for m in metrics],
                    "epochs": [m["epoch"] for m in metrics],
                    "final_value": metrics[-1].get(metric_name),
                    "best_value": min(
                        m.get(metric_name, float("inf"))
                        for m in metrics
                    ),
                }

        return comparison

    def delete_experiment(self, exp_id: str) -> bool:
        """Delete an experiment and all its data."""
        if exp_id not in self.experiments:
            return False

        # Remove experiment directory
        exp_dir = self.experiments_dir / exp_id
        if exp_dir.exists():
            shutil.rmtree(exp_dir)

        # Remove from index
        del self.experiments[exp_id]
        self._save_experiments_index()

        return True
