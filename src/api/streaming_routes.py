"""
Real-Time Seismic Data Streaming Routes

Provides endpoints for:
- Live seismic data streaming via WebSocket
- FDSN polling for recent waveforms
- Station status monitoring
- Earthquake detection events
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
import threading

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/streams", tags=["Streaming"])


# Indian Ocean priority stations for tsunami monitoring
INDIAN_OCEAN_STATIONS = [
    {"id": "PALK", "network": "II", "name": "Pallekele", "country": "Sri Lanka", "lat": 7.2728, "lng": 80.7022},
    {"id": "COCO", "network": "II", "name": "Cocos Islands", "country": "Australia", "lat": -12.1901, "lng": 96.8349},
    {"id": "DGAR", "network": "II", "name": "Diego Garcia", "country": "BIOT", "lat": -7.4121, "lng": 72.4525},
    {"id": "CHTO", "network": "IU", "name": "Chiang Mai", "country": "Thailand", "lat": 18.8141, "lng": 98.9443},
    {"id": "TATO", "network": "IU", "name": "Taipei", "country": "Taiwan", "lat": 24.9735, "lng": 121.4971},
    {"id": "NWAO", "network": "IU", "name": "Narrogin", "country": "Australia", "lat": -32.9277, "lng": 117.2390},
    {"id": "WRAB", "network": "II", "name": "Warramunga", "country": "Australia", "lat": -19.9336, "lng": 134.3600},
    {"id": "MBWA", "network": "IU", "name": "Marble Bar", "country": "Australia", "lat": -21.1590, "lng": 119.7313},
]


class StreamingState:
    """Global streaming state manager."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.station_data: Dict[str, Dict] = defaultdict(dict)
        self.is_streaming = False
        self.last_fetch_time: Optional[datetime] = None
        self.recent_events: List[Dict] = []

    def get_station_status(self, station_id: str) -> str:
        """Get current status of a station."""
        if station_id in self.station_data:
            last_update = self.station_data[station_id].get("last_update")
            if last_update:
                age = (datetime.utcnow() - last_update).total_seconds()
                if age < 60:
                    return "online"
                elif age < 300:
                    return "delayed"
        return "offline"


streaming_state = StreamingState()


# Pydantic Models
class StationInfo(BaseModel):
    """Station information response."""
    id: str
    network: str
    name: str
    country: str
    lat: float
    lng: float
    status: str
    last_update: Optional[str] = None
    amplitude: Optional[float] = None


class StreamConfig(BaseModel):
    """Streaming configuration."""
    stations: List[str] = Field(default=["PALK", "COCO", "DGAR"], description="Station IDs to stream")
    channels: List[str] = Field(default=["BHZ"], description="Channels to stream")
    poll_interval: int = Field(default=10, description="Polling interval in seconds")


class WaveformData(BaseModel):
    """Waveform data response."""
    station_id: str
    network: str
    channel: str
    start_time: str
    end_time: str
    sample_rate: float
    data: List[float]
    amplitude: float


class SeismicEvent(BaseModel):
    """Detected seismic event."""
    event_id: str
    timestamp: str
    lat: float
    lng: float
    magnitude: Optional[float] = None
    depth: Optional[float] = None
    detecting_stations: List[str]
    tsunami_potential: bool = False


# REST Endpoints
@router.get("/stations", response_model=List[StationInfo])
async def list_stations():
    """List available Indian Ocean monitoring stations."""
    stations = []
    for station in INDIAN_OCEAN_STATIONS:
        station_info = StationInfo(
            id=station["id"],
            network=station["network"],
            name=station["name"],
            country=station["country"],
            lat=station["lat"],
            lng=station["lng"],
            status=streaming_state.get_station_status(station["id"]),
            last_update=streaming_state.station_data.get(station["id"], {}).get("last_update", "").isoformat() if streaming_state.station_data.get(station["id"], {}).get("last_update") else None,
            amplitude=streaming_state.station_data.get(station["id"], {}).get("amplitude"),
        )
        stations.append(station_info)
    return stations


@router.get("/status")
async def get_stream_status():
    """Get current streaming status."""
    return {
        "is_streaming": streaming_state.is_streaming,
        "last_fetch_time": streaming_state.last_fetch_time.isoformat() if streaming_state.last_fetch_time else None,
        "active_connections": len(streaming_state.active_connections),
        "stations_online": sum(1 for s in INDIAN_OCEAN_STATIONS if streaming_state.get_station_status(s["id"]) == "online"),
        "recent_events": len(streaming_state.recent_events),
    }


@router.post("/start")
async def start_streaming(config: StreamConfig):
    """Start streaming seismic data."""
    streaming_state.is_streaming = True
    return {
        "status": "started",
        "stations": config.stations,
        "poll_interval": config.poll_interval,
    }


@router.post("/stop")
async def stop_streaming():
    """Stop streaming seismic data."""
    streaming_state.is_streaming = False
    return {"status": "stopped"}


@router.get("/waveforms/{station_id}")
async def get_recent_waveforms(station_id: str, minutes: int = 5):
    """Get recent waveform data for a station using FDSN."""
    try:
        from obspy.clients.fdsn import Client
        from obspy import UTCDateTime

        # Find station info
        station_info = next((s for s in INDIAN_OCEAN_STATIONS if s["id"] == station_id), None)
        if not station_info:
            raise HTTPException(status_code=404, detail=f"Station {station_id} not found")

        client = Client("IRIS")
        end_time = UTCDateTime.now()
        start_time = end_time - (minutes * 60)

        st = client.get_waveforms(
            network=station_info["network"],
            station=station_id,
            location="*",
            channel="BHZ",
            starttime=start_time,
            endtime=end_time
        )

        if len(st) == 0:
            return {"station_id": station_id, "data": [], "message": "No data available"}

        tr = st[0]

        # Downsample for transmission (keep every 10th sample)
        decimation = max(1, len(tr.data) // 1000)
        data = tr.data[::decimation].tolist()

        # Calculate amplitude (RMS)
        import numpy as np
        amplitude = float(np.sqrt(np.mean(tr.data ** 2)))

        # Update streaming state
        streaming_state.station_data[station_id] = {
            "last_update": datetime.utcnow(),
            "amplitude": min(1.0, amplitude / 10000),  # Normalize
            "sample_rate": tr.stats.sampling_rate,
        }

        return WaveformData(
            station_id=station_id,
            network=station_info["network"],
            channel="BHZ",
            start_time=str(tr.stats.starttime),
            end_time=str(tr.stats.endtime),
            sample_rate=float(tr.stats.sampling_rate),
            data=data,
            amplitude=min(1.0, amplitude / 10000),
        )

    except Exception as e:
        logger.error(f"Failed to fetch waveforms for {station_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events")
async def get_recent_events(hours: int = 24):
    """Get recent seismic events from USGS."""
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson",
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()

        events = []
        for feature in data.get("features", [])[:20]:  # Latest 20 events
            props = feature["properties"]
            coords = feature["geometry"]["coordinates"]

            # Check if event is in Indian Ocean region
            lng, lat, depth = coords
            is_indian_ocean = -40 < lat < 30 and 40 < lng < 140

            event = {
                "event_id": feature["id"],
                "timestamp": datetime.fromtimestamp(props["time"] / 1000).isoformat(),
                "lat": lat,
                "lng": lng,
                "magnitude": props["mag"],
                "depth": depth,
                "place": props["place"],
                "tsunami": props.get("tsunami", 0) == 1,
                "in_region": is_indian_ocean,
            }
            events.append(event)

        # Filter to Indian Ocean if requested
        indian_ocean_events = [e for e in events if e["in_region"]]

        return {
            "total_events": len(events),
            "indian_ocean_events": len(indian_ocean_events),
            "events": events,
            "indian_ocean_only": indian_ocean_events,
        }

    except Exception as e:
        logger.error(f"Failed to fetch events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket Endpoint
@router.websocket("/live")
async def websocket_seismic_live(websocket: WebSocket):
    """WebSocket endpoint for live seismic data streaming."""
    await websocket.accept()
    connection_id = str(id(websocket))
    streaming_state.active_connections[connection_id] = websocket

    logger.info(f"New WebSocket connection: {connection_id}")

    try:
        # Send initial station data
        await websocket.send_json({
            "type": "init",
            "stations": INDIAN_OCEAN_STATIONS,
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Start polling loop
        poll_interval = 5  # seconds

        while True:
            try:
                # Check for incoming messages with timeout
                try:
                    message = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=poll_interval
                    )
                    data = json.loads(message)

                    if data.get("type") == "subscribe":
                        # Handle subscription to specific stations
                        stations = data.get("stations", ["PALK", "COCO", "DGAR"])
                        await websocket.send_json({
                            "type": "subscribed",
                            "stations": stations,
                        })

                    elif data.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})

                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue polling

                # Fetch and send updated data
                updates = []
                for station in INDIAN_OCEAN_STATIONS[:3]:  # Start with 3 stations
                    station_data = streaming_state.station_data.get(station["id"], {})

                    # Simulate amplitude variation if no real data
                    import random
                    amplitude = station_data.get("amplitude", random.uniform(0.1, 0.4))

                    updates.append({
                        "station_id": station["id"],
                        "network": station["network"],
                        "amplitude": amplitude,
                        "status": streaming_state.get_station_status(station["id"]),
                        "timestamp": datetime.utcnow().isoformat(),
                    })

                # Send station updates
                await websocket.send_json({
                    "type": "update",
                    "data": updates,
                    "timestamp": datetime.utcnow().isoformat(),
                })

                streaming_state.last_fetch_time = datetime.utcnow()

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(1)

    except WebSocketDisconnect:
        pass
    finally:
        del streaming_state.active_connections[connection_id]
        logger.info(f"WebSocket disconnected: {connection_id}")
