"""
API Routes for Real Seismic Data Access

Provides endpoints for:
- Downloading seismic waveforms from IRIS
- Accessing major earthquake events
- Generating regional velocity models
- Working with 2004 Indian Ocean Tsunami data
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import numpy as np

router = APIRouter(prefix="/seismic", tags=["seismic"])


# ============================================================================
# Request/Response Models
# ============================================================================


class EventInfo(BaseModel):
    """Information about a seismic event."""

    event_id: str
    name: str
    magnitude: float
    latitude: float
    longitude: float
    depth_km: float
    origin_time: str
    event_type: str
    description: str


class DownloadRequest(BaseModel):
    """Request to download seismic data."""

    event_name: str = Field(
        default="sumatra_2004",
        description="Event name (sumatra_2004, tohoku_2011, chile_2010, nepal_2015)"
    )
    duration_minutes: int = Field(
        default=60,
        ge=1,
        le=180,
        description="Duration of waveforms to download"
    )
    channels: str = Field(
        default="BHZ",
        description="Channel code (BHZ=broadband vertical)"
    )


class VelocityModelRequest(BaseModel):
    """Request for regional velocity model."""

    region: str = Field(
        default="sumatra",
        description="Region name (sumatra, japan, chile)"
    )
    nx: int = Field(default=200, ge=50, le=1000)
    nz: int = Field(default=100, ge=50, le=500)
    x_extent_km: float = Field(default=500.0, ge=100, le=2000)
    z_extent_km: float = Field(default=200.0, ge=50, le=700)


class TrainingDataRequest(BaseModel):
    """Request for PINN training data from real seismic data."""

    event_name: str = Field(default="sumatra_2004")
    duration_minutes: int = Field(default=30, ge=1, le=120)
    downsample_factor: int = Field(default=10, ge=1, le=100)
    filter_band: Optional[List[float]] = Field(
        default=[0.01, 1.0],
        description="Bandpass filter [low_hz, high_hz]"
    )


# ============================================================================
# Routes
# ============================================================================


@router.get("/events")
async def list_available_events() -> Dict[str, Any]:
    """
    List all available major seismic events.

    These events have pre-configured parameters for easy data download.
    """
    from ..data.seismic_data_loader import MAJOR_EVENTS

    events = []
    for name, event in MAJOR_EVENTS.items():
        events.append({
            "event_key": name,
            "event_id": event.event_id,
            "name": event.name,
            "magnitude": event.magnitude,
            "latitude": event.latitude,
            "longitude": event.longitude,
            "depth_km": event.depth_km,
            "origin_time": event.origin_time.isoformat(),
            "event_type": event.event_type,
            "description": event.description,
        })

    return {
        "events": events,
        "note": "Use event_key (e.g., 'sumatra_2004') in API calls",
    }


@router.get("/events/{event_name}")
async def get_event_details(event_name: str) -> EventInfo:
    """Get detailed information about a specific event."""
    from ..data.seismic_data_loader import MAJOR_EVENTS

    if event_name not in MAJOR_EVENTS:
        available = list(MAJOR_EVENTS.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Event not found. Available: {available}"
        )

    event = MAJOR_EVENTS[event_name]

    return EventInfo(
        event_id=event.event_id,
        name=event.name,
        magnitude=event.magnitude,
        latitude=event.latitude,
        longitude=event.longitude,
        depth_km=event.depth_km,
        origin_time=event.origin_time.isoformat(),
        event_type=event.event_type,
        description=event.description,
    )


@router.post("/download")
async def download_seismic_data(
    request: DownloadRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """
    Download seismic waveform data from IRIS.

    This downloads real seismograms recorded during major earthquakes
    from the IRIS Data Management Center.

    Note: First download may take several minutes depending on network speed.
    Data is cached locally for subsequent requests.
    """
    from ..data.seismic_data_loader import IRISDataLoader, MAJOR_EVENTS

    if request.event_name not in MAJOR_EVENTS:
        available = list(MAJOR_EVENTS.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown event. Available: {available}"
        )

    try:
        loader = IRISDataLoader()
        event = loader.get_event(request.event_name)

        # Check cache first - if cached, load and return the station data
        cache_file = Path("data/seismic_cache") / f"{request.event_name}_{request.duration_minutes}min.npz"

        if cache_file.exists():
            # Load cached waveforms to get station info
            waveforms = loader.download_event_data(
                request.event_name,
                duration_minutes=request.duration_minutes,
                use_cache=True,
            )
            return {
                "status": "cached",
                "event": {
                    "name": event.name,
                    "magnitude": event.magnitude,
                    "origin_time": event.origin_time.isoformat(),
                },
                "waveforms_downloaded": len(waveforms),
                "stations": [
                    {
                        "network": wf.station.network,
                        "station": wf.station.station,
                        "latitude": wf.station.latitude,
                        "longitude": wf.station.longitude,
                        "samples": len(wf.data),
                    }
                    for wf in waveforms
                ],
                "cache_file": str(cache_file),
                "message": "Loaded from cache.",
            }

        # Start download
        waveforms = loader.download_event_data(
            request.event_name,
            duration_minutes=request.duration_minutes,
            channels=request.channels,
        )

        return {
            "status": "success",
            "event": {
                "name": event.name,
                "magnitude": event.magnitude,
                "origin_time": event.origin_time.isoformat(),
            },
            "waveforms_downloaded": len(waveforms),
            "stations": [
                {
                    "network": wf.station.network,
                    "station": wf.station.station,
                    "latitude": wf.station.latitude,
                    "longitude": wf.station.longitude,
                    "samples": len(wf.data),
                }
                for wf in waveforms
            ],
            "cache_file": str(cache_file),
        }

    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"ObsPy not installed: {e}. Install with: pip install obspy"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/training-data")
async def get_training_data(request: TrainingDataRequest) -> Dict[str, Any]:
    """
    Get PINN-ready training data from real seismic waveforms.

    This downloads (or loads from cache), preprocesses, and formats
    the data for PINN training.

    Returns:
        Coordinates, observations, and event metadata
    """
    from ..data.seismic_data_loader import IRISDataLoader, MAJOR_EVENTS

    if request.event_name not in MAJOR_EVENTS:
        available = list(MAJOR_EVENTS.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown event. Available: {available}"
        )

    try:
        loader = IRISDataLoader()
        event = loader.get_event(request.event_name)

        # Download/load waveforms
        waveforms = loader.download_event_data(
            request.event_name,
            duration_minutes=request.duration_minutes,
        )

        if not waveforms:
            raise HTTPException(
                status_code=404,
                detail="No waveforms available. Try downloading first."
            )

        # Preprocess
        filter_band = tuple(request.filter_band) if request.filter_band else None
        processed = loader.preprocess_waveforms(
            waveforms,
            filter_band=filter_band,
        )

        # Convert to training format
        training_data = loader.to_training_data(
            processed,
            event,
            downsample_factor=request.downsample_factor,
        )

        return {
            "status": "success",
            "event": {
                "name": event.name,
                "magnitude": event.magnitude,
                "latitude": event.latitude,
                "longitude": event.longitude,
                "depth_km": event.depth_km,
            },
            "training_data": {
                "n_samples": training_data["coordinates"].shape[0],
                "coordinate_dim": training_data["coordinates"].shape[1],
                "coordinates": training_data["coordinates"].tolist(),
                "observations": training_data["observations"].tolist(),
            },
            "preprocessing": {
                "filter_band_hz": request.filter_band,
                "downsample_factor": request.downsample_factor,
                "n_stations": training_data["n_stations"],
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/velocity-model")
async def generate_regional_velocity_model(
    request: VelocityModelRequest
) -> Dict[str, Any]:
    """
    Generate a regional crustal velocity model.

    Currently supports:
    - sumatra: Sumatra-Andaman subduction zone
    """
    from ..data.seismic_data_loader import SumatraVelocityModel

    try:
        if request.region.lower() == "sumatra":
            model = SumatraVelocityModel(
                nx=request.nx,
                nz=request.nz,
                x_extent_km=request.x_extent_km,
                z_extent_km=request.z_extent_km,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown region: {request.region}. Available: sumatra"
            )

        velocity_data = model.to_pinn_format()

        return {
            "status": "success",
            "region": velocity_data["region"],
            "shape": velocity_data["shape"],
            "dx_km": velocity_data["dx_km"],
            "dz_km": velocity_data["dz_km"],
            "x_extent_km": velocity_data["x_extent_km"],
            "z_extent_km": velocity_data["z_extent_km"],
            "stats": velocity_data["stats"],
            "velocity": velocity_data["velocity"].tolist(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sumatra-2004/info")
async def get_sumatra_2004_info() -> Dict[str, Any]:
    """
    Get comprehensive information about the 2004 Sumatra-Andaman earthquake.

    This was the third-largest earthquake ever recorded and generated
    the deadliest tsunami in recorded history.
    """
    from ..data.seismic_data_loader import MAJOR_EVENTS

    event = MAJOR_EVENTS["sumatra_2004"]

    return {
        "event": {
            "name": event.name,
            "date": "December 26, 2004",
            "time_utc": "00:58:53",
            "magnitude": event.magnitude,
            "magnitude_type": "Mw",
            "location": {
                "latitude": event.latitude,
                "longitude": event.longitude,
                "depth_km": event.depth_km,
                "region": "Off the west coast of northern Sumatra, Indonesia",
            },
            "rupture": {
                "length_km": event.rupture_length_km,
                "width_km": event.rupture_width_km,
                "duration_seconds": 500,
                "slip_m": 15,
            },
            "tsunami": {
                "max_wave_height_m": 30,
                "affected_countries": 14,
                "casualties": "~230,000",
            },
        },
        "data_sources": {
            "iris": {
                "description": "Seismic waveforms from global stations",
                "url": "https://ds.iris.edu/seismon/swaves/index.php",
            },
            "noaa": {
                "description": "Tide gauge and DART buoy data",
                "url": "https://nctr.pmel.noaa.gov/indo_1204.html",
            },
            "usgs": {
                "description": "Earthquake parameters and rupture models",
                "url": "https://earthquake.usgs.gov/earthquakes/eventpage/official20041226005853450_30",
            },
        },
        "available_data": {
            "seismic_waveforms": True,
            "velocity_model": True,
            "tide_gauge": "via NOAA download",
            "dart_buoy": "via NOAA download",
        },
        "usage": {
            "download_waveforms": "POST /api/v1/seismic/download with event_name='sumatra_2004'",
            "get_training_data": "POST /api/v1/seismic/training-data with event_name='sumatra_2004'",
            "velocity_model": "POST /api/v1/seismic/velocity-model with region='sumatra'",
        },
    }


@router.get("/training-results/images")
async def list_training_result_images() -> Dict[str, Any]:
    """
    List available pre-generated training result images.
    """
    outputs_dir = Path("outputs/sumatra_2004")

    if not outputs_dir.exists():
        return {
            "status": "not_found",
            "images": [],
            "message": "No training results available. Run training first.",
        }

    images = []
    image_info = {
        "sumatra_waveforms.png": {
            "title": "Real Seismograms",
            "description": "Waveforms from 12 global seismic stations",
        },
        "sumatra_station_map.png": {
            "title": "Station Network",
            "description": "Global distribution of recording stations",
        },
        "sumatra_velocity_model.png": {
            "title": "Velocity Model",
            "description": "Sumatra-Andaman subduction zone structure",
        },
        "sumatra_training_results.png": {
            "title": "Training Results",
            "description": "Loss curves and waveform predictions",
        },
        "sumatra_wavefield_snapshots.png": {
            "title": "Wavefield Snapshots",
            "description": "PINN predictions at different times",
        },
        "sumatra_summary.png": {
            "title": "Complete Analysis",
            "description": "Overview of all results",
        },
    }

    for img_file in outputs_dir.glob("*.png"):
        info = image_info.get(img_file.name, {
            "title": img_file.stem.replace("_", " ").title(),
            "description": "",
        })
        images.append({
            "filename": img_file.name,
            "path": f"/seismic/training-results/image/{img_file.name}",
            "title": info["title"],
            "description": info["description"],
            "size_bytes": img_file.stat().st_size,
        })

    return {
        "status": "success",
        "images": images,
        "event": "sumatra_2004",
    }


@router.get("/training-results/image/{filename}")
async def get_training_result_image(filename: str):
    """
    Serve a training result image.
    """
    from fastapi.responses import FileResponse

    # Security: only allow specific filenames
    allowed_files = [
        "sumatra_waveforms.png",
        "sumatra_station_map.png",
        "sumatra_velocity_model.png",
        "sumatra_training_results.png",
        "sumatra_wavefield_snapshots.png",
        "sumatra_summary.png",
    ]

    if filename not in allowed_files:
        raise HTTPException(status_code=404, detail="Image not found")

    file_path = Path("outputs/sumatra_2004") / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(file_path, media_type="image/png")


@router.get("/stations")
async def list_available_stations() -> Dict[str, Any]:
    """
    List seismic stations available for data download.

    These are stations from the Global Seismographic Network (GSN)
    that recorded major earthquakes.
    """
    from ..data.seismic_data_loader import IRISDataLoader

    stations = []
    for network, station in IRISDataLoader.GSN_STATIONS:
        stations.append({
            "network": network,
            "station": station,
            "network_name": "IU=IRIS/USGS, II=IRIS/IDA",
        })

    return {
        "stations": stations,
        "note": "These stations are automatically queried when downloading event data",
        "networks": {
            "IU": "IRIS/USGS Global Seismographic Network",
            "II": "IRIS/IDA Global Seismographic Network",
        },
    }
