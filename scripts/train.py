#!/usr/bin/env python3
"""
PINN Seismic Inversion - Training Script

This script provides a command-line interface for training PINN models
for seismic waveform inversion.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --model siren --epochs 1000 --velocity-model marmousi
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from datetime import datetime

from src.models.pinn import SeismicPINN
from src.training.trainer import PINNTrainer, TrainingConfig
from src.data.velocity_models import (
    MarmousiModel,
    LayeredVelocityModel,
    SaltDomeModel,
    FaultModel,
)
from src.data.generators import SeismicDataGenerator, CollocationPointSampler
from src.physics.wave_equation import AcousticWaveEquation, ElasticWaveEquation
from src.physics.sources import RickerWavelet
from src.utils.checkpointing import (
    CheckpointManager,
    ExperimentTracker,
    ExperimentConfig,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PINN models for seismic waveform inversion"
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )

    # Model architecture
    parser.add_argument(
        "--model",
        type=str,
        default="siren",
        choices=["siren", "fourier", "modulated"],
        help="Model architecture type",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension of the network",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of hidden layers",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )

    # Physics configuration
    parser.add_argument(
        "--velocity-model",
        type=str,
        default="marmousi",
        choices=["marmousi", "layered", "salt_dome", "fault"],
        help="Velocity model to use",
    )
    parser.add_argument(
        "--wave-equation",
        type=str,
        default="acoustic",
        choices=["acoustic", "elastic"],
        help="Wave equation type",
    )
    parser.add_argument(
        "--source-frequency",
        type=float,
        default=15.0,
        help="Source wavelet frequency in Hz",
    )

    # Loss weights
    parser.add_argument(
        "--physics-weight",
        type=float,
        default=1.0,
        help="Weight for physics loss",
    )
    parser.add_argument(
        "--data-weight",
        type=float,
        default=1.0,
        help="Weight for data loss",
    )
    parser.add_argument(
        "--boundary-weight",
        type=float,
        default=0.1,
        help="Weight for boundary loss",
    )

    # Adaptive weighting
    parser.add_argument(
        "--adaptive-weights",
        action="store_true",
        help="Use adaptive loss weighting",
    )
    parser.add_argument(
        "--adaptive-method",
        type=str,
        default="gradnorm",
        choices=["gradnorm", "uncertainty", "softadapt"],
        help="Adaptive weighting method",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save checkpoint every N epochs",
    )

    # Device settings
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda, cpu, or auto)",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use mixed precision training",
    )

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_velocity_model(name: str, nx: int = 200, nz: int = 100):
    """Get velocity model by name."""
    models = {
        "marmousi": lambda: MarmousiModel(nx=nx, nz=nz),
        "layered": lambda: LayeredVelocityModel(
            nx=nx, nz=nz,
            velocities=[1500, 2000, 2500, 3000, 3500],
            depths=[0, 200, 400, 600, 800],
        ),
        "salt_dome": lambda: SaltDomeModel(nx=nx, nz=nz),
        "fault": lambda: FaultModel(nx=nx, nz=nz),
    }
    return models[name]()


def main():
    """Main training function."""
    args = parse_args()

    # Load config file if provided
    config = {}
    if args.config:
        config = load_config(args.config)

    # Merge command line args with config (command line takes precedence)
    for key, value in vars(args).items():
        if value is not None and key != "config":
            config[key.replace("-", "_")] = value

    # Set device
    if config.get("device", "auto") == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config["device"]

    print("=" * 60)
    print("  PINN Seismic Inversion - Training")
    print("=" * 60)
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Create experiment name if not provided
    if not config.get("experiment_name"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config["experiment_name"] = f"{config['model']}_{config['velocity_model']}_{timestamp}"

    # Setup directories
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize experiment tracker
    tracker = ExperimentTracker(output_dir / "experiments")
    exp_config = ExperimentConfig(
        name=config["experiment_name"],
        description=f"Training {config['model']} on {config['velocity_model']}",
        model_type=config["model"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        num_epochs=config["epochs"],
        physics_weight=config["physics_weight"],
        data_weight=config["data_weight"],
        boundary_weight=config["boundary_weight"],
        velocity_model=config["velocity_model"],
        wave_equation=config["wave_equation"],
    )
    exp_id = tracker.create_experiment(exp_config)
    print(f"Experiment ID: {exp_id}")

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        output_dir / "checkpoints" / exp_id,
        max_checkpoints=10,
        keep_best=3,
    )

    # Create velocity model
    print(f"\nCreating velocity model: {config['velocity_model']}")
    velocity_model = get_velocity_model(config["velocity_model"])

    # Create wave equation
    if config["wave_equation"] == "acoustic":
        wave_equation = AcousticWaveEquation()
    else:
        wave_equation = ElasticWaveEquation()

    # Create source
    source = RickerWavelet(
        frequency=config["source_frequency"],
        amplitude=1.0,
        delay=1.0 / config["source_frequency"],
    )

    # Create PINN model
    print(f"Creating PINN model: {config['model']}")
    model = SeismicPINN(
        input_dim=3,  # x, z, t
        output_dim=1,  # pressure/displacement
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        backbone_type=config["model"],
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create data generator
    data_generator = SeismicDataGenerator(
        velocity_model=velocity_model,
        source=source,
        wave_equation=wave_equation,
    )

    # Create training config
    training_config = TrainingConfig(
        num_epochs=config["epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        physics_weight=config["physics_weight"],
        data_weight=config["data_weight"],
        boundary_weight=config["boundary_weight"],
        use_adaptive_weights=config.get("adaptive_weights", False),
        adaptive_method=config.get("adaptive_method", "gradnorm"),
        checkpoint_interval=config["checkpoint_interval"],
        device=device,
        mixed_precision=config.get("mixed_precision", False),
    )

    # Create trainer
    trainer = PINNTrainer(
        model=model,
        config=training_config,
        checkpoint_manager=checkpoint_manager,
        experiment_tracker=tracker,
    )

    # Resume from checkpoint if specified
    if config.get("resume"):
        print(f"\nResuming from checkpoint: {config['resume']}")
        trainer.load_checkpoint(config["resume"])

    # Start training
    print("\n" + "=" * 60)
    print("  Starting Training")
    print("=" * 60 + "\n")

    try:
        trainer.train(
            data_generator=data_generator,
            velocity_model=velocity_model,
            wave_equation=wave_equation,
        )
        tracker.finish_experiment(status="completed")
        print("\n✓ Training completed successfully!")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        tracker.finish_experiment(status="cancelled")

    except Exception as e:
        print(f"\n\nError during training: {e}")
        tracker.finish_experiment(status="failed")
        raise

    # Print final summary
    print("\n" + "=" * 60)
    print("  Training Summary")
    print("=" * 60)
    print(f"Experiment: {exp_id}")
    print(f"Checkpoints saved to: {output_dir / 'checkpoints' / exp_id}")
    print(f"Logs saved to: {output_dir / 'experiments'}")


if __name__ == "__main__":
    main()
