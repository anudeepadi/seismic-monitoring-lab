"""
FastAPI Application for Physics-Informed Neural Networks

Production-ready API server providing:
- RESTful endpoints for model management
- WebSocket for real-time training updates
- Background task processing for training jobs
- Model inference and visualization endpoints
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .routes import router
from .seismic_routes import router as seismic_router
from .websocket_manager import WebSocketManager
from .training_manager import TrainingManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Application state
class AppState:
    """Singleton for application state."""

    def __init__(self):
        self.training_manager: Optional[TrainingManager] = None
        self.websocket_manager = WebSocketManager()
        self.active_jobs: Dict[str, Dict] = {}
        self.models: Dict[str, Any] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting PINN API server...")
    logger.info(f"Device: {app_state.device}")

    # Initialize training manager
    app_state.training_manager = TrainingManager(
        websocket_manager=app_state.websocket_manager,
        device=app_state.device,
    )

    # Create directories
    Path("checkpoints").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    yield

    # Shutdown
    logger.info("Shutting down PINN API server...")

    # Cancel active training jobs
    for job_id in list(app_state.active_jobs.keys()):
        if app_state.training_manager:
            await app_state.training_manager.cancel_job(job_id)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Physics-Informed Neural Networks API",
        description="API for training and inference of PINNs for seismic waveform inversion",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:3002", "http://localhost:5173", "*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(router, prefix="/api/v1")
    app.include_router(seismic_router, prefix="/api/v1")

    # Mount static files for frontend (if built)
    static_path = Path(__file__).parent.parent.parent / "frontend" / "dist"
    if static_path.exists():
        app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")

    return app


app = create_app()


# ============================================================================
# Pydantic Models
# ============================================================================


class TrainingConfig(BaseModel):
    """Training configuration."""

    model_type: str = Field(default="siren", description="Model backbone type")
    hidden_dim: int = Field(default=256, description="Hidden layer dimension")
    hidden_layers: int = Field(default=6, description="Number of hidden layers")
    learning_rate: float = Field(default=1e-4, description="Learning rate")
    max_epochs: int = Field(default=1000, description="Maximum training epochs")
    batch_size: int = Field(default=10000, description="Batch size")
    velocity_model: str = Field(default="marmousi", description="Velocity model type")
    use_adaptive_weights: bool = Field(default=True, description="Use adaptive loss weighting")


class VelocityModelConfig(BaseModel):
    """Velocity model configuration."""

    model_type: str = Field(default="marmousi", description="Model type")
    nx: int = Field(default=200, description="Grid points in x")
    nz: int = Field(default=100, description="Grid points in z")
    dx: float = Field(default=10.0, description="Grid spacing x (m)")
    dz: float = Field(default=10.0, description="Grid spacing z (m)")
    parameters: Dict[str, Any] = Field(default_factory=dict)


class InferenceRequest(BaseModel):
    """Model inference request."""

    model_id: str
    coordinates: List[List[float]]  # [[x, z, t], ...]
    velocity: Optional[List[float]] = None


class VisualizationRequest(BaseModel):
    """Wavefield visualization request."""

    model_id: str
    time: float
    x_range: List[float] = [0, 1]
    z_range: List[float] = [0, 1]
    resolution: List[int] = [100, 100]


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Physics-Informed Neural Networks API",
        "version": "1.0.0",
        "status": "running",
        "device": str(app_state.device),
    }


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "device": str(app_state.device),
        "cuda_available": torch.cuda.is_available(),
        "active_jobs": len(app_state.active_jobs),
    }


@app.post("/api/v1/training/start")
async def start_training(
    config: TrainingConfig,
    background_tasks: BackgroundTasks,
):
    """Start a new training job."""
    job_id = str(uuid4())

    try:
        # Create and start training job
        job_info = await app_state.training_manager.create_job(
            job_id=job_id,
            config=config.model_dump(),
        )

        # Start training in background
        background_tasks.add_task(
            app_state.training_manager.run_training,
            job_id,
        )

        app_state.active_jobs[job_id] = job_info

        return {
            "job_id": job_id,
            "status": "started",
            "config": config.model_dump(),
        }

    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/training/{job_id}")
async def get_training_status(job_id: str):
    """Get training job status."""
    if job_id not in app_state.active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return app_state.training_manager.get_job_status(job_id)


@app.post("/api/v1/training/{job_id}/stop")
async def stop_training(job_id: str):
    """Stop a training job."""
    if job_id not in app_state.active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    await app_state.training_manager.cancel_job(job_id)

    return {"job_id": job_id, "status": "stopped"}


@app.get("/api/v1/training/{job_id}/history")
async def get_training_history(job_id: str):
    """Get training history for a job."""
    if job_id not in app_state.active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return app_state.training_manager.get_training_history(job_id)


@app.get("/api/v1/models")
async def list_models():
    """List available trained models."""
    checkpoints_dir = Path("checkpoints")
    models = []

    for model_dir in checkpoints_dir.iterdir():
        if model_dir.is_dir():
            best_model = model_dir / "best_model.pt"
            if best_model.exists():
                checkpoint = torch.load(best_model, map_location="cpu")
                models.append({
                    "id": model_dir.name,
                    "epoch": checkpoint.get("epoch", 0),
                    "best_loss": checkpoint.get("metrics", {}).get("total_loss", None),
                    "config": checkpoint.get("config", {}),
                })

    return {"models": models}


@app.post("/api/v1/models/{model_id}/load")
async def load_model(model_id: str):
    """Load a trained model for inference."""
    model_path = Path("checkpoints") / model_id / "best_model.pt"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        checkpoint = torch.load(model_path, map_location=app_state.device)

        # Reconstruct model from checkpoint
        # This would need the actual model class import
        app_state.models[model_id] = {
            "checkpoint": checkpoint,
            "loaded_at": datetime.utcnow().isoformat(),
        }

        return {
            "model_id": model_id,
            "status": "loaded",
            "epoch": checkpoint.get("epoch", 0),
        }

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/inference")
async def run_inference(request: InferenceRequest):
    """Run model inference."""
    if request.model_id not in app_state.models:
        raise HTTPException(status_code=404, detail="Model not loaded")

    try:
        # Convert coordinates to tensor
        coords = torch.tensor(request.coordinates, device=app_state.device)

        # Get model and run inference
        model_info = app_state.models[request.model_id]
        # This would use the actual loaded model

        # Placeholder response
        predictions = torch.zeros(len(request.coordinates), 1)

        return {
            "model_id": request.model_id,
            "predictions": predictions.cpu().numpy().tolist(),
        }

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/visualize/wavefield")
async def visualize_wavefield(request: VisualizationRequest):
    """Generate wavefield snapshot for visualization."""
    if request.model_id not in app_state.models:
        raise HTTPException(status_code=404, detail="Model not loaded")

    try:
        # Generate wavefield snapshot
        # This would use the actual model

        nx, nz = request.resolution
        wavefield = torch.zeros(nz, nx)

        return {
            "model_id": request.model_id,
            "time": request.time,
            "shape": [nz, nx],
            "data": wavefield.cpu().numpy().tolist(),
            "x_range": request.x_range,
            "z_range": request.z_range,
        }

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/velocity-model/generate")
async def generate_velocity_model(config: VelocityModelConfig):
    """Generate a synthetic velocity model."""
    from ..data.velocity_models import (
        MarmousiModel,
        LayeredVelocityModel,
        SaltDomeModel,
        RandomGaussianModel,
    )

    try:
        if config.model_type == "marmousi":
            model = MarmousiModel(config.nx, config.nz, config.dx, config.dz)
        elif config.model_type == "layered":
            model = LayeredVelocityModel(config.nx, config.nz, config.dx, config.dz)
        elif config.model_type == "salt_dome":
            model = SaltDomeModel(config.nx, config.nz, config.dx, config.dz)
        elif config.model_type == "random":
            model = RandomGaussianModel(config.nx, config.nz, config.dx, config.dz)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

        velocity = model.generate()

        return {
            "model_type": config.model_type,
            "shape": velocity.shape,
            "data": velocity.tolist(),
            "stats": {
                "min": float(velocity.min()),
                "max": float(velocity.max()),
                "mean": float(velocity.mean()),
            },
        }

    except Exception as e:
        logger.error(f"Failed to generate velocity model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WebSocket Endpoints
# ============================================================================


@app.websocket("/ws/training/{job_id}")
async def websocket_training(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time training updates."""
    await app_state.websocket_manager.connect(websocket, job_id)

    try:
        while True:
            # Wait for messages from client (e.g., control commands)
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0,
                )
                message = json.loads(data)

                # Handle control messages
                if message.get("type") == "pause":
                    await app_state.training_manager.pause_job(job_id)
                elif message.get("type") == "resume":
                    await app_state.training_manager.resume_job(job_id)
                elif message.get("type") == "stop":
                    await app_state.training_manager.cancel_job(job_id)

            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})

    except WebSocketDisconnect:
        app_state.websocket_manager.disconnect(websocket, job_id)
        logger.info(f"WebSocket disconnected for job {job_id}")


@app.websocket("/ws/visualization")
async def websocket_visualization(websocket: WebSocket):
    """WebSocket endpoint for real-time visualization updates."""
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "request_frame":
                # Generate visualization frame
                model_id = message.get("model_id")
                time_val = message.get("time", 0.0)

                # Generate and send frame
                frame_data = {
                    "type": "frame",
                    "time": time_val,
                    "data": [],  # Would be actual wavefield data
                }
                await websocket.send_json(frame_data)

    except WebSocketDisconnect:
        logger.info("Visualization WebSocket disconnected")


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
