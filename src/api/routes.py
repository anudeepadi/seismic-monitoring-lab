"""
API Routes for PINN Operations

Additional routes for specific PINN operations including:
- Seismic data generation
- Model comparison
- Experiment management
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel, Field
import numpy as np

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================


class SourceConfig(BaseModel):
    """Seismic source configuration."""

    source_type: str = Field(default="ricker", description="Source wavelet type")
    peak_frequency: float = Field(default=20.0, description="Peak frequency (Hz)")
    position_x: float = Field(default=0.5, description="Normalized x position")
    position_z: float = Field(default=0.05, description="Normalized z position")


class ReceiverConfig(BaseModel):
    """Receiver configuration."""

    n_receivers: int = Field(default=101, description="Number of receivers")
    depth: float = Field(default=0.05, description="Receiver depth (normalized)")
    spread: str = Field(default="linear", description="Receiver spread type")


class SyntheticDataRequest(BaseModel):
    """Request for synthetic seismic data generation."""

    velocity_model_type: str = Field(default="marmousi")
    nx: int = Field(default=200)
    nz: int = Field(default=100)
    nt: int = Field(default=1000)
    dt: float = Field(default=0.001)
    dx: float = Field(default=10.0)
    dz: float = Field(default=10.0)
    source: SourceConfig = Field(default_factory=SourceConfig)
    receivers: ReceiverConfig = Field(default_factory=ReceiverConfig)


class ExperimentConfig(BaseModel):
    """Experiment configuration for tracking."""

    name: str
    description: Optional[str] = None
    model_configuration: Dict[str, Any]
    training_configuration: Dict[str, Any]
    velocity_model: str = "marmousi"
    tags: List[str] = Field(default_factory=list)


class ComparisonRequest(BaseModel):
    """Request for model comparison."""

    model_ids: List[str]
    metric: str = "total_loss"
    coordinates: Optional[List[List[float]]] = None


# ============================================================================
# Routes
# ============================================================================


@router.post("/synthetic-data/generate")
async def generate_synthetic_data(request: SyntheticDataRequest):
    """
    Generate synthetic seismic data using finite differences.

    This creates ground truth data for training PINNs by solving
    the wave equation numerically.
    """
    try:
        from ..data import SeismicDataGenerator, VelocityModelGenerator
        from ..data.velocity_models import MarmousiModel, SaltDomeModel

        # Generate velocity model
        if request.velocity_model_type == "marmousi":
            model = MarmousiModel(
                request.nx, request.nz, request.dx, request.dz
            )
        elif request.velocity_model_type == "salt_dome":
            model = SaltDomeModel(
                request.nx, request.nz, request.dx, request.dz
            )
        else:
            gen = VelocityModelGenerator(
                request.nx, request.nz, request.dx, request.dz
            )
            velocity = gen.linear_gradient()
            model = type("Model", (), {"generate": lambda: velocity})()

        velocity = model.generate()

        # Generate seismic data
        generator = SeismicDataGenerator(
            velocity_model=velocity,
            dx=request.dx,
            dz=request.dz,
            dt=request.dt,
            nt=request.nt,
            peak_frequency=request.source.peak_frequency,
        )

        # Source position
        source_x = int(request.source.position_x * request.nx)
        source_z = int(request.source.position_z * request.nz)

        # Receiver positions
        receiver_x = np.linspace(
            0, request.nx - 1, request.receivers.n_receivers
        ).astype(int)
        receiver_z = np.full(
            request.receivers.n_receivers,
            int(request.receivers.depth * request.nz),
            dtype=int,
        )

        # Generate shot
        snapshots, seismograms = generator.generate_shot(
            source_x=source_x,
            source_z=source_z,
            receiver_x=receiver_x,
            receiver_z=receiver_z,
        )

        return {
            "status": "success",
            "velocity_model": {
                "shape": velocity.shape,
                "min": float(velocity.min()),
                "max": float(velocity.max()),
                "data": velocity.tolist(),
            },
            "seismograms": {
                "shape": seismograms.shape,
                "data": seismograms.tolist(),
            },
            "n_snapshots": len(snapshots),
            "source_position": [source_x, source_z],
            "receiver_positions": {
                "x": receiver_x.tolist(),
                "z": receiver_z.tolist(),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/create")
async def create_experiment(config: ExperimentConfig):
    """
    Create a new tracked experiment.

    Experiments allow organizing multiple training runs with
    different configurations for comparison.
    """
    from uuid import uuid4
    from datetime import datetime

    experiment_id = str(uuid4())

    # Create experiment directory
    exp_dir = Path("experiments") / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    import json

    config_path = exp_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "id": experiment_id,
            "name": config.name,
            "description": config.description,
            "model_config": config.model_config,
            "training_config": config.training_config,
            "velocity_model": config.velocity_model,
            "tags": config.tags,
            "created_at": datetime.utcnow().isoformat(),
        }, f, indent=2)

    return {
        "experiment_id": experiment_id,
        "name": config.name,
        "path": str(exp_dir),
    }


@router.get("/experiments")
async def list_experiments():
    """List all experiments."""
    import json

    experiments = []
    exp_base = Path("experiments")

    if exp_base.exists():
        for exp_dir in exp_base.iterdir():
            if exp_dir.is_dir():
                config_path = exp_dir / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                        experiments.append({
                            "id": config.get("id"),
                            "name": config.get("name"),
                            "created_at": config.get("created_at"),
                            "tags": config.get("tags", []),
                        })

    return {"experiments": experiments}


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Get experiment details."""
    import json

    config_path = Path("experiments") / experiment_id / "config.json"

    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")

    with open(config_path) as f:
        config = json.load(f)

    # Get associated runs
    runs_path = Path("experiments") / experiment_id / "runs"
    runs = []
    if runs_path.exists():
        for run_dir in runs_path.iterdir():
            if run_dir.is_dir():
                runs.append(run_dir.name)

    return {
        **config,
        "runs": runs,
    }


@router.post("/compare")
async def compare_models(request: ComparisonRequest):
    """
    Compare multiple trained models.

    Provides metrics comparison and optional prediction differences.
    """
    import torch

    results = []

    for model_id in request.model_ids:
        model_path = Path("checkpoints") / model_id / "best_model.pt"

        if not model_path.exists():
            results.append({
                "model_id": model_id,
                "error": "Model not found",
            })
            continue

        checkpoint = torch.load(model_path, map_location="cpu")

        metrics = checkpoint.get("metrics", {})
        epoch = checkpoint.get("epoch", 0)

        results.append({
            "model_id": model_id,
            "epoch": epoch,
            "metrics": metrics,
        })

    # Sort by requested metric
    results.sort(
        key=lambda x: x.get("metrics", {}).get(request.metric, float("inf"))
    )

    return {
        "comparison_metric": request.metric,
        "results": results,
        "best_model": results[0]["model_id"] if results else None,
    }


@router.post("/velocity-model/upload")
async def upload_velocity_model(
    file: UploadFile = File(...),
    name: str = "uploaded_model",
):
    """
    Upload a custom velocity model.

    Supports numpy (.npy) and SEG-Y formats.
    """
    import tempfile
    import shutil

    # Save uploaded file
    upload_dir = Path("uploads") / "velocity_models"
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / f"{name}_{file.filename}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Load and validate
    try:
        if file.filename.endswith(".npy"):
            velocity = np.load(file_path)
        elif file.filename.endswith((".segy", ".sgy")):
            from ..data.velocity_models import load_segy_velocity
            velocity = load_segy_velocity(str(file_path), 200, 100)
        else:
            raise ValueError("Unsupported file format")

        return {
            "name": name,
            "path": str(file_path),
            "shape": velocity.shape,
            "stats": {
                "min": float(velocity.min()),
                "max": float(velocity.max()),
                "mean": float(velocity.mean()),
            },
        }

    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/physics/wave-equation")
async def get_wave_equation_info():
    """
    Get information about supported wave equations.

    Returns mathematical formulation and available options.
    """
    return {
        "equations": {
            "acoustic": {
                "name": "Acoustic Wave Equation",
                "formula": "∂²u/∂t² = v²(∂²u/∂x² + ∂²u/∂z²)",
                "variables": {
                    "u": "Pressure field",
                    "v": "Acoustic velocity",
                    "t": "Time",
                    "x, z": "Spatial coordinates",
                },
                "parameters": ["velocity"],
            },
            "elastic": {
                "name": "Elastic Wave Equation",
                "formula": "ρ∂²u/∂t² = ∇·σ",
                "variables": {
                    "u": "Displacement vector",
                    "σ": "Stress tensor",
                    "ρ": "Density",
                },
                "parameters": ["vp", "vs", "density"],
            },
            "viscoacoustic": {
                "name": "Viscoacoustic Wave Equation",
                "formula": "∂²u/∂t² + (ω₀/Q)∂u/∂t = v²∇²u",
                "variables": {
                    "u": "Pressure field",
                    "Q": "Quality factor",
                    "ω₀": "Reference frequency",
                },
                "parameters": ["velocity", "Q", "reference_frequency"],
            },
        },
        "boundary_conditions": [
            "absorbing",
            "free_surface",
            "pml",
            "sponge",
        ],
        "source_types": [
            "ricker",
            "gaussian",
            "gaussian_derivative",
            "plane_wave",
        ],
    }


@router.get("/architectures")
async def get_available_architectures():
    """
    Get information about available neural network architectures.
    """
    return {
        "architectures": {
            "siren": {
                "name": "SIREN (Sinusoidal Representation Networks)",
                "description": "Uses periodic sine activations for accurate derivative computation",
                "paper": "Sitzmann et al., 2020",
                "best_for": "Smooth solutions with high-frequency components",
                "parameters": ["omega_0", "hidden_layers", "hidden_features"],
            },
            "fourier": {
                "name": "Fourier Feature Networks",
                "description": "Uses random Fourier features for positional encoding",
                "paper": "Tancik et al., 2020",
                "best_for": "Learning high-frequency patterns",
                "parameters": ["num_frequencies", "frequency_scale", "multi_scale"],
            },
            "modulated": {
                "name": "Modulated SIREN",
                "description": "SIREN with velocity-conditional modulation",
                "best_for": "Velocity-dependent wave propagation",
                "parameters": ["modulation_type", "base_omega"],
            },
            "gradient_siren": {
                "name": "Gradient-Scaling SIREN",
                "description": "SIREN with learnable gradient scaling",
                "best_for": "Multi-scale physics with varying magnitudes",
                "parameters": ["num_scales"],
            },
        },
        "recommended": {
            "forward_modeling": "siren",
            "velocity_inversion": "modulated",
            "high_frequency": "fourier",
        },
    }


@router.post("/inference/batch")
async def batch_inference(
    model_id: str,
    coordinates: List[List[float]],
    batch_size: int = 10000,
):
    """
    Run batch inference for large coordinate sets.

    Processes coordinates in batches to manage memory.
    """
    import torch

    model_path = Path("checkpoints") / model_id / "best_model.pt"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu")

    # Process in batches
    coords = torch.tensor(coordinates, dtype=torch.float32)
    predictions = []

    for i in range(0, len(coords), batch_size):
        batch = coords[i:i + batch_size]
        # Run inference (placeholder)
        pred = torch.zeros(len(batch), 1)
        predictions.extend(pred.numpy().tolist())

    return {
        "model_id": model_id,
        "n_points": len(coordinates),
        "predictions": predictions,
    }
