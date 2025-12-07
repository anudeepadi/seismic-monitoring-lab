"""
Learning Rate Schedulers for PINN Training

Specialized schedulers for physics-informed learning:
- Curriculum schedulers that gradually increase physics weight
- Multi-stage training for different phases
- Warmup strategies for stable initialization
"""

from typing import Callable, Dict, List, Optional, Tuple
import math

import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Warmup followed by cosine annealing.

    Linearly increases LR during warmup, then follows cosine decay.
    This is effective for transformer-like architectures.

    Args:
        optimizer: Torch optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        min_lr: Minimum learning rate
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            return [
                self.min_lr
                + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class CyclicCosineScheduler(_LRScheduler):
    """
    Cyclic cosine annealing with restarts.

    Periodically resets learning rate to help escape local minima.
    Each cycle can optionally decrease the maximum LR.

    Args:
        optimizer: Torch optimizer
        cycle_length: Epochs per cycle
        cycle_mult: Multiply cycle length after each restart
        lr_decay: Decay factor for max LR after each cycle
        min_lr: Minimum learning rate
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        cycle_length: int = 100,
        cycle_mult: float = 1.0,
        lr_decay: float = 0.9,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ) -> None:
        self.cycle_length = cycle_length
        self.cycle_mult = cycle_mult
        self.lr_decay = lr_decay
        self.min_lr = min_lr
        self.current_cycle = 0
        self.cycle_epoch = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        # Determine which cycle we're in
        epoch = self.last_epoch
        cycle = 0
        cycle_start = 0
        current_cycle_length = self.cycle_length

        while epoch >= cycle_start + current_cycle_length:
            cycle_start += current_cycle_length
            current_cycle_length = int(current_cycle_length * self.cycle_mult)
            cycle += 1

        # Position within current cycle
        cycle_progress = (epoch - cycle_start) / current_cycle_length

        # Decay factor for max LR
        max_lr_decay = self.lr_decay ** cycle

        return [
            self.min_lr
            + (base_lr * max_lr_decay - self.min_lr)
            * 0.5
            * (1 + math.cos(math.pi * cycle_progress))
            for base_lr in self.base_lrs
        ]


class CurriculumScheduler:
    """
    Curriculum learning scheduler for loss weights.

    Gradually increases the weight of physics loss as training
    progresses. This helps the network first learn the data
    distribution before enforcing strict physics constraints.

    Stages:
    1. Data-focused: High data weight, low physics weight
    2. Balanced: Equal weights
    3. Physics-focused: High physics weight for refinement

    Args:
        initial_weights: Starting loss weights
        target_weights: Final loss weights
        warmup_epochs: Epochs for curriculum transition
        schedule_type: 'linear', 'exponential', or 'step'
    """

    def __init__(
        self,
        initial_weights: Dict[str, float],
        target_weights: Dict[str, float],
        warmup_epochs: int = 500,
        schedule_type: str = "exponential",
    ) -> None:
        self.initial_weights = initial_weights
        self.target_weights = target_weights
        self.warmup_epochs = warmup_epochs
        self.schedule_type = schedule_type

    def get_weights(self, epoch: int) -> Dict[str, float]:
        """Get loss weights for current epoch."""
        if epoch >= self.warmup_epochs:
            return self.target_weights.copy()

        progress = epoch / self.warmup_epochs

        if self.schedule_type == "linear":
            alpha = progress
        elif self.schedule_type == "exponential":
            alpha = 1 - math.exp(-5 * progress)
        elif self.schedule_type == "step":
            alpha = 0.0 if progress < 0.5 else 1.0
        elif self.schedule_type == "cosine":
            alpha = 0.5 * (1 - math.cos(math.pi * progress))
        else:
            alpha = progress

        weights = {}
        for key in self.initial_weights:
            initial = self.initial_weights[key]
            target = self.target_weights.get(key, initial)
            weights[key] = initial + alpha * (target - initial)

        return weights


class MultiStageScheduler:
    """
    Multi-stage training scheduler.

    Divides training into distinct phases, each with different:
    - Learning rates
    - Loss weights
    - Optimizer settings

    Example stages:
    1. Pretrain: High LR, data loss only
    2. Physics: Medium LR, add physics loss
    3. Refinement: Low LR, balanced losses

    Args:
        stages: List of stage configurations
    """

    def __init__(
        self,
        stages: List[Dict],
    ) -> None:
        """
        Each stage dict should contain:
        - 'epochs': Number of epochs for this stage
        - 'lr': Learning rate for this stage
        - 'weights': Loss weights dict
        - 'optimizer': Optional optimizer type
        """
        self.stages = stages
        self.stage_boundaries = []

        # Compute stage boundaries
        total = 0
        for stage in stages:
            total += stage.get("epochs", 100)
            self.stage_boundaries.append(total)

    def get_current_stage(self, epoch: int) -> int:
        """Get current training stage index."""
        for i, boundary in enumerate(self.stage_boundaries):
            if epoch < boundary:
                return i
        return len(self.stages) - 1

    def get_stage_config(self, epoch: int) -> Dict:
        """Get configuration for current stage."""
        stage_idx = self.get_current_stage(epoch)
        return self.stages[stage_idx]

    def get_learning_rate(self, epoch: int) -> float:
        """Get learning rate for current epoch."""
        config = self.get_stage_config(epoch)
        return config.get("lr", 1e-4)

    def get_weights(self, epoch: int) -> Dict[str, float]:
        """Get loss weights for current epoch."""
        config = self.get_stage_config(epoch)
        return config.get("weights", {"data": 1.0, "physics": 1.0})

    def apply_to_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> None:
        """Apply stage-specific learning rate to optimizer."""
        lr = self.get_learning_rate(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


class PolynomialDecayScheduler(_LRScheduler):
    """
    Polynomial learning rate decay.

    LR decays polynomially from initial to minimum:
    lr = (lr_init - lr_min) * (1 - progress)^power + lr_min

    Args:
        optimizer: Torch optimizer
        total_epochs: Total training epochs
        power: Polynomial power (1 = linear, 2 = quadratic)
        min_lr: Minimum learning rate
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        power: float = 1.0,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ) -> None:
        self.total_epochs = total_epochs
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        progress = min(self.last_epoch / self.total_epochs, 1.0)
        decay = (1 - progress) ** self.power

        return [
            self.min_lr + (base_lr - self.min_lr) * decay
            for base_lr in self.base_lrs
        ]


class OneCycleLRWithWarmdown(_LRScheduler):
    """
    One-cycle learning rate with warmdown phase.

    Similar to PyTorch's OneCycleLR but with a gentler warmdown
    phase at the end of training.

    Phases:
    1. Warmup: Linear increase to max LR
    2. Annealing: Cosine decay to initial LR
    3. Warmdown: Linear decay to min LR

    Args:
        optimizer: Torch optimizer
        max_lr: Maximum learning rate
        total_epochs: Total training epochs
        warmup_pct: Fraction of training for warmup
        warmdown_pct: Fraction of training for warmdown
        min_lr: Minimum learning rate
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        total_epochs: int,
        warmup_pct: float = 0.1,
        warmdown_pct: float = 0.2,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ) -> None:
        self.max_lr = max_lr
        self.total_epochs = total_epochs
        self.warmup_pct = warmup_pct
        self.warmdown_pct = warmdown_pct
        self.min_lr = min_lr

        self.warmup_epochs = int(total_epochs * warmup_pct)
        self.warmdown_epochs = int(total_epochs * warmdown_pct)
        self.anneal_epochs = total_epochs - self.warmup_epochs - self.warmdown_epochs

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            # Warmup phase
            progress = epoch / self.warmup_epochs
            return [
                base_lr + progress * (self.max_lr - base_lr)
                for base_lr in self.base_lrs
            ]

        elif epoch < self.warmup_epochs + self.anneal_epochs:
            # Annealing phase
            progress = (epoch - self.warmup_epochs) / self.anneal_epochs
            return [
                base_lr
                + (self.max_lr - base_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]

        else:
            # Warmdown phase
            progress = (epoch - self.warmup_epochs - self.anneal_epochs) / max(
                self.warmdown_epochs, 1
            )
            return [
                self.min_lr + (base_lr - self.min_lr) * (1 - progress)
                for base_lr in self.base_lrs
            ]


def create_scheduler(
    scheduler_type: str,
    optimizer: torch.optim.Optimizer,
    config: Dict,
) -> Optional[_LRScheduler]:
    """
    Factory function to create schedulers.

    Args:
        scheduler_type: Type of scheduler
        optimizer: Torch optimizer
        config: Scheduler configuration dict

    Returns:
        Configured scheduler or None
    """
    if scheduler_type == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=config.get("warmup_epochs", 100),
            total_epochs=config.get("total_epochs", 1000),
            min_lr=config.get("min_lr", 1e-7),
        )

    elif scheduler_type == "cyclic_cosine":
        return CyclicCosineScheduler(
            optimizer,
            cycle_length=config.get("cycle_length", 100),
            cycle_mult=config.get("cycle_mult", 1.0),
            lr_decay=config.get("lr_decay", 0.9),
            min_lr=config.get("min_lr", 1e-7),
        )

    elif scheduler_type == "polynomial":
        return PolynomialDecayScheduler(
            optimizer,
            total_epochs=config.get("total_epochs", 1000),
            power=config.get("power", 1.0),
            min_lr=config.get("min_lr", 1e-7),
        )

    elif scheduler_type == "one_cycle":
        return OneCycleLRWithWarmdown(
            optimizer,
            max_lr=config.get("max_lr", 1e-3),
            total_epochs=config.get("total_epochs", 1000),
            warmup_pct=config.get("warmup_pct", 0.1),
            warmdown_pct=config.get("warmdown_pct", 0.2),
            min_lr=config.get("min_lr", 1e-7),
        )

    elif scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get("total_epochs", 1000),
            eta_min=config.get("min_lr", 1e-7),
        )

    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 100),
            gamma=config.get("gamma", 0.5),
        )

    elif scheduler_type == "none" or scheduler_type is None:
        return None

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
