"""
Main Training Loop for Physics-Informed Neural Networks

This module provides the complete training pipeline including:
- Mixed precision training for efficiency
- Gradient accumulation for large effective batch sizes
- Learning rate scheduling with warmup
- Checkpointing and experiment tracking
- Real-time monitoring via WebSocket
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging

import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for PINN training."""

    # Basic settings
    max_epochs: int = 1000
    batch_size: int = 10000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Optimizer settings
    optimizer: str = "adam"  # 'adam', 'adamw', 'lbfgs'
    betas: Tuple[float, float] = (0.9, 0.999)

    # Scheduler settings
    scheduler: str = "cosine"  # 'cosine', 'step', 'plateau', 'none'
    warmup_epochs: int = 100
    min_lr: float = 1e-7

    # Training tricks
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    use_mixed_precision: bool = True

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_frequency: int = 100
    keep_n_checkpoints: int = 5

    # Logging
    log_frequency: int = 10
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Early stopping
    patience: int = 200
    min_delta: float = 1e-6

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Validation
    val_frequency: int = 50
    val_batch_size: int = 5000

    # WebSocket updates
    websocket_host: Optional[str] = None
    websocket_port: Optional[int] = None


@dataclass
class TrainingState:
    """Tracks training state for checkpointing and monitoring."""

    epoch: int = 0
    global_step: int = 0
    best_loss: float = float("inf")
    best_epoch: int = 0
    patience_counter: int = 0

    # History
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)

    # Component losses
    data_losses: List[float] = field(default_factory=list)
    physics_losses: List[float] = field(default_factory=list)
    boundary_losses: List[float] = field(default_factory=list)
    initial_losses: List[float] = field(default_factory=list)


class PINNTrainer:
    """
    Complete training pipeline for Physics-Informed Neural Networks.

    Features:
    - Flexible loss composition
    - Adaptive loss weighting
    - Mixed precision training
    - Comprehensive logging
    - WebSocket updates for real-time monitoring

    Args:
        model: PINN model to train
        loss_fn: Combined loss function
        config: Training configuration
        train_loader: Training data loader
        val_loader: Optional validation data loader
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> None:
        self.model = model.to(config.device)
        self.loss_fn = loss_fn
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks = callbacks or []

        self.device = torch.device(config.device)
        self.state = TrainingState()

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision else None

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # WebSocket connection
        self.websocket = None

        # Logging
        self._setup_logging()

    def _create_optimizer(self) -> Optimizer:
        """Create optimizer based on config."""
        if self.config.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "lbfgs":
            return torch.optim.LBFGS(
                self.model.parameters(),
                lr=self.config.learning_rate,
                max_iter=20,
                history_size=50,
                line_search_fn="strong_wolfe",
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self) -> Optional[_LRScheduler]:
        """Create learning rate scheduler."""
        if self.config.scheduler == "none":
            return None

        warmup_steps = self.config.warmup_epochs * len(self.train_loader)
        total_steps = self.config.max_epochs * len(self.train_loader)

        if self.config.scheduler == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=total_steps // 4,
                eta_min=self.config.min_lr,
            )
        elif self.config.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.max_epochs // 3,
                gamma=0.5,
            )
        elif self.config.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=50,
                min_lr=self.config.min_lr,
            )
        elif self.config.scheduler == "warmup_cosine":
            # Custom warmup + cosine
            return self._create_warmup_cosine_scheduler(warmup_steps, total_steps)

        return None

    def _create_warmup_cosine_scheduler(
        self,
        warmup_steps: int,
        total_steps: int,
    ):
        """Create warmup + cosine annealing scheduler."""
        from torch.optim.lr_scheduler import LambdaLR
        import math

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

        return LambdaLR(self.optimizer, lr_lambda)

    def _setup_logging(self) -> None:
        """Setup logging and optional W&B."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        if self.config.wandb_project:
            try:
                import wandb

                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name,
                    config=vars(self.config),
                )
                self.use_wandb = True
            except ImportError:
                logger.warning("wandb not installed, skipping")
                self.use_wandb = False
        else:
            self.use_wandb = False

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        Returns:
            Training history and final metrics
        """
        logger.info("Starting training...")
        logger.info(f"Config: {self.config}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        start_time = time.time()

        try:
            for epoch in range(self.state.epoch, self.config.max_epochs):
                self.state.epoch = epoch

                # Training epoch
                train_metrics = self._train_epoch()
                self.state.train_losses.append(train_metrics["total_loss"])

                # Track component losses
                self._track_component_losses(train_metrics)

                # Validation
                if self.val_loader and epoch % self.config.val_frequency == 0:
                    val_metrics = self._validate()
                    self.state.val_losses.append(val_metrics["total_loss"])
                else:
                    val_metrics = {}

                # Learning rate
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.state.learning_rates.append(current_lr)

                # Logging
                if epoch % self.config.log_frequency == 0:
                    self._log_metrics(epoch, train_metrics, val_metrics)

                # Scheduler step
                if self.scheduler is not None:
                    if isinstance(
                        self.scheduler,
                        torch.optim.lr_scheduler.ReduceLROnPlateau,
                    ):
                        self.scheduler.step(train_metrics["total_loss"])
                    else:
                        self.scheduler.step()

                # Checkpointing
                if epoch % self.config.checkpoint_frequency == 0:
                    self._save_checkpoint(epoch, train_metrics)

                # Best model tracking
                if train_metrics["total_loss"] < self.state.best_loss - self.config.min_delta:
                    self.state.best_loss = train_metrics["total_loss"]
                    self.state.best_epoch = epoch
                    self.state.patience_counter = 0
                    self._save_checkpoint(epoch, train_metrics, is_best=True)
                else:
                    self.state.patience_counter += 1

                # Early stopping
                if self.state.patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

                # WebSocket update
                self._send_websocket_update(epoch, train_metrics, val_metrics)

                # Callbacks
                for callback in self.callbacks:
                    callback(epoch, train_metrics, val_metrics, self.model)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self._save_checkpoint(self.state.epoch, {}, is_interrupted=True)

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")

        return {
            "train_losses": self.state.train_losses,
            "val_losses": self.state.val_losses,
            "best_loss": self.state.best_loss,
            "best_epoch": self.state.best_epoch,
            "total_time": total_time,
        }

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_metrics: Dict[str, List[float]] = {}

        accumulation_counter = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._move_to_device(batch)

            # Forward pass with optional mixed precision
            if self.config.use_mixed_precision and self.scaler is not None:
                with autocast():
                    loss, metrics = self.loss_fn(
                        self.model, batch, return_components=True
                    )
            else:
                loss, metrics = self.loss_fn(
                    self.model, batch, return_components=True
                )

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.config.use_mixed_precision and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulation_counter += 1

            # Update weights
            if accumulation_counter >= self.config.gradient_accumulation_steps:
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                accumulation_counter = 0
                self.state.global_step += 1

            # Track metrics
            epoch_losses.append(loss.item() * self.config.gradient_accumulation_steps)
            for key, value in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)

        # Average metrics
        avg_metrics = {
            key: sum(values) / len(values)
            for key, values in epoch_metrics.items()
        }
        avg_metrics["total_loss"] = sum(epoch_losses) / len(epoch_losses)

        return avg_metrics

    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        val_losses = []
        val_metrics: Dict[str, List[float]] = {}

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_to_device(batch)
                loss, metrics = self.loss_fn(self.model, batch)

                val_losses.append(loss.item())
                for key, value in metrics.items():
                    if key not in val_metrics:
                        val_metrics[key] = []
                    val_metrics[key].append(value)

        avg_metrics = {
            key: sum(values) / len(values)
            for key, values in val_metrics.items()
        }
        avg_metrics["total_loss"] = sum(val_losses) / len(val_losses)

        return avg_metrics

    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch data to device."""

        def to_device(x):
            if isinstance(x, Tensor):
                return x.to(self.device)
            elif isinstance(x, dict):
                return {k: to_device(v) for k, v in x.items()}
            elif isinstance(x, list):
                return [to_device(item) for item in x]
            return x

        return to_device(batch)

    def _track_component_losses(self, metrics: Dict[str, float]) -> None:
        """Track individual loss components."""
        if "data_loss" in metrics:
            self.state.data_losses.append(metrics["data_loss"])
        if "physics_loss" in metrics:
            self.state.physics_losses.append(metrics["physics_loss"])
        if "boundary_loss" in metrics:
            self.state.boundary_losses.append(metrics["boundary_loss"])
        if "initial_loss" in metrics:
            self.state.initial_losses.append(metrics["initial_loss"])

    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        """Log metrics to console and W&B."""
        lr = self.optimizer.param_groups[0]["lr"]

        msg = f"Epoch {epoch:5d} | Loss: {train_metrics['total_loss']:.6f}"
        if "physics_loss" in train_metrics:
            msg += f" | Physics: {train_metrics['physics_loss']:.6f}"
        if val_metrics:
            msg += f" | Val: {val_metrics['total_loss']:.6f}"
        msg += f" | LR: {lr:.2e}"

        logger.info(msg)

        if self.use_wandb:
            import wandb

            log_dict = {
                "train/" + k: v for k, v in train_metrics.items()
            }
            log_dict["learning_rate"] = lr
            log_dict["epoch"] = epoch

            if val_metrics:
                log_dict.update({
                    "val/" + k: v for k, v in val_metrics.items()
                })

            wandb.log(log_dict)

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        is_interrupted: bool = False,
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.state.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": vars(self.config),
            "state": {
                "best_loss": self.state.best_loss,
                "best_epoch": self.state.best_epoch,
                "train_losses": self.state.train_losses[-100:],  # Keep last 100
                "val_losses": self.state.val_losses[-100:],
            },
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        elif is_interrupted:
            path = self.checkpoint_dir / "interrupted_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda x: x.stat().st_mtime,
        )

        while len(checkpoints) > self.config.keep_n_checkpoints:
            oldest = checkpoints.pop(0)
            oldest.unlink()

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load model from checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.state.epoch = checkpoint["epoch"]
        self.state.global_step = checkpoint["global_step"]

        if "state" in checkpoint:
            self.state.best_loss = checkpoint["state"]["best_loss"]
            self.state.best_epoch = checkpoint["state"]["best_epoch"]

        logger.info(f"Loaded checkpoint from {path}")

    def _send_websocket_update(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        """Send training update via WebSocket."""
        if self.config.websocket_host is None:
            return

        try:
            message = {
                "type": "training_update",
                "epoch": epoch,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "best_loss": self.state.best_loss,
            }

            # This would be an async send in production
            # await self.websocket.send(json.dumps(message))
        except Exception as e:
            logger.warning(f"Failed to send WebSocket update: {e}")

    def get_training_history(self) -> Dict[str, List[float]]:
        """Get full training history."""
        return {
            "train_losses": self.state.train_losses,
            "val_losses": self.state.val_losses,
            "data_losses": self.state.data_losses,
            "physics_losses": self.state.physics_losses,
            "boundary_losses": self.state.boundary_losses,
            "initial_losses": self.state.initial_losses,
            "learning_rates": self.state.learning_rates,
        }
