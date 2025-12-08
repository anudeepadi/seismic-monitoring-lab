"""
Real Seismic Data Loader Module

This module provides functionality to download and process real seismic data
from various sources including:
- IRIS (Incorporated Research Institutions for Seismology)
- NOAA (National Oceanic and Atmospheric Administration)
- USGS (United States Geological Survey)

Specifically designed to work with the 2004 Indian Ocean Tsunami data
and similar major seismic events.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SeismicEvent:
    """Represents a seismic event (earthquake)."""

    event_id: str
    name: str
    magnitude: float
    latitude: float
    longitude: float
    depth_km: float
    origin_time: datetime
    event_type: str = "earthquake"
    description: str = ""

    # Additional metadata
    moment_tensor: Optional[Dict[str, float]] = None
    focal_mechanism: Optional[Dict[str, float]] = None
    rupture_length_km: Optional[float] = None
    rupture_width_km: Optional[float] = None


@dataclass
class SeismicStation:
    """Represents a seismic recording station."""

    network: str
    station: str
    latitude: float
    longitude: float
    elevation_m: float
    channels: List[str] = field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


@dataclass
class SeismicWaveform:
    """Container for seismic waveform data."""

    station: SeismicStation
    event: SeismicEvent
    channel: str
    data: np.ndarray
    times: np.ndarray
    sampling_rate: float
    start_time: datetime
    units: str = "counts"  # or "m/s", "m/s^2"

    # Processing info
    filtered: bool = False
    detrended: bool = False
    instrument_corrected: bool = False


# Predefined major seismic events
MAJOR_EVENTS = {
    "sumatra_2004": SeismicEvent(
        event_id="usp000d0mp",
        name="2004 Sumatra-Andaman Earthquake",
        magnitude=9.1,
        latitude=3.295,
        longitude=95.982,
        depth_km=30.0,
        origin_time=datetime(2004, 12, 26, 0, 58, 53),
        event_type="megathrust",
        description="The 2004 Indian Ocean earthquake and tsunami",
        rupture_length_km=1300,
        rupture_width_km=150,
    ),
    "tohoku_2011": SeismicEvent(
        event_id="usp000hvpg",
        name="2011 Tohoku Earthquake",
        magnitude=9.1,
        latitude=38.297,
        longitude=142.373,
        depth_km=29.0,
        origin_time=datetime(2011, 3, 11, 5, 46, 24),
        event_type="megathrust",
        description="The 2011 Japan earthquake and tsunami",
        rupture_length_km=500,
        rupture_width_km=200,
    ),
    "chile_2010": SeismicEvent(
        event_id="usp000h7rf",
        name="2010 Chile Earthquake",
        magnitude=8.8,
        latitude=-36.122,
        longitude=-72.898,
        depth_km=22.9,
        origin_time=datetime(2010, 2, 27, 6, 34, 14),
        event_type="megathrust",
        description="The 2010 Maule earthquake",
    ),
    "nepal_2015": SeismicEvent(
        event_id="us20002926",
        name="2015 Nepal Earthquake",
        magnitude=7.8,
        latitude=28.231,
        longitude=84.731,
        depth_km=8.2,
        origin_time=datetime(2015, 4, 25, 6, 11, 26),
        event_type="thrust",
        description="The 2015 Gorkha earthquake",
    ),
}


class IRISDataLoader:
    """
    Data loader for IRIS (Incorporated Research Institutions for Seismology).

    Downloads seismic waveform data from the IRIS Data Management Center
    using ObsPy and FDSN web services.

    Example:
        >>> loader = IRISDataLoader()
        >>> waveforms = loader.download_event_data("sumatra_2004", duration_minutes=60)
        >>> print(f"Downloaded {len(waveforms)} waveforms")
    """

    # Global Seismographic Network stations good for teleseismic events
    GSN_STATIONS = [
        ("II", "PALK"),   # Sri Lanka - closest to Sumatra
        ("II", "COCO"),   # Cocos Islands
        ("II", "DGAR"),   # Diego Garcia
        ("IU", "CHTO"),   # Chiang Mai, Thailand
        ("IU", "TATO"),   # Taipei, Taiwan
        ("IU", "CTAO"),   # Australia
        ("IU", "NWAO"),   # Australia
        ("IU", "SNZO"),   # New Zealand
        ("II", "AAK"),    # Kyrgyzstan
        ("IU", "HNR"),    # Solomon Islands
        ("IU", "GUMO"),   # Guam
        ("II", "KURK"),   # Kazakhstan
        ("IU", "COLA"),   # Alaska
        ("IU", "ANMO"),   # New Mexico
        ("II", "BFO"),    # Germany
    ]

    def __init__(self, cache_dir: str = "data/seismic_cache"):
        """
        Initialize the IRIS data loader.

        Args:
            cache_dir: Directory for caching downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = None

    @property
    def client(self):
        """Lazy initialization of ObsPy FDSN client."""
        if self._client is None:
            try:
                from obspy.clients.fdsn import Client
                self._client = Client("IRIS")
                logger.info("Connected to IRIS FDSN web service")
            except ImportError:
                raise ImportError(
                    "ObsPy is required for IRIS data access. "
                    "Install with: pip install obspy"
                )
        return self._client

    def get_event(self, event_name: str) -> SeismicEvent:
        """Get a predefined seismic event by name."""
        if event_name not in MAJOR_EVENTS:
            available = ", ".join(MAJOR_EVENTS.keys())
            raise ValueError(
                f"Unknown event: {event_name}. Available: {available}"
            )
        return MAJOR_EVENTS[event_name]

    def download_event_data(
        self,
        event_name: str,
        duration_minutes: int = 60,
        stations: Optional[List[Tuple[str, str]]] = None,
        channels: str = "BHZ",
        use_cache: bool = True,
    ) -> List[SeismicWaveform]:
        """
        Download seismic waveform data for a major event.

        Args:
            event_name: Name of the event (e.g., "sumatra_2004")
            duration_minutes: Duration of data to download after event
            stations: List of (network, station) tuples, or None for defaults
            channels: Channel codes (e.g., "BHZ" for broadband vertical)
            use_cache: Whether to use cached data if available

        Returns:
            List of SeismicWaveform objects
        """
        event = self.get_event(event_name)

        # Check cache BEFORE importing obspy (allows working without obspy if cached)
        cache_file = self.cache_dir / f"{event_name}_{duration_minutes}min.npz"
        if use_cache and cache_file.exists():
            logger.info(f"Loading cached data from {cache_file}")
            return self._load_from_cache(cache_file, event)

        # Only import obspy when actually needed for download
        from obspy import UTCDateTime

        stations = stations or self.GSN_STATIONS

        # Download data
        origin_time = UTCDateTime(event.origin_time)
        end_time = origin_time + duration_minutes * 60

        waveforms = []

        for network, station in stations:
            try:
                logger.info(f"Downloading {network}.{station}...")

                # Get waveforms
                st = self.client.get_waveforms(
                    network=network,
                    station=station,
                    location="*",
                    channel=channels,
                    starttime=origin_time,
                    endtime=end_time,
                )

                # Get station metadata
                inv = self.client.get_stations(
                    network=network,
                    station=station,
                    level="station",
                )

                sta_info = inv[0][0]
                station_obj = SeismicStation(
                    network=network,
                    station=station,
                    latitude=sta_info.latitude,
                    longitude=sta_info.longitude,
                    elevation_m=sta_info.elevation,
                    channels=[channels],
                )

                for tr in st:
                    times = np.arange(len(tr.data)) / tr.stats.sampling_rate

                    waveform = SeismicWaveform(
                        station=station_obj,
                        event=event,
                        channel=tr.stats.channel,
                        data=tr.data.astype(np.float32),
                        times=times.astype(np.float32),
                        sampling_rate=tr.stats.sampling_rate,
                        start_time=tr.stats.starttime.datetime,
                    )
                    waveforms.append(waveform)

            except Exception as e:
                logger.warning(f"Failed to download {network}.{station}: {e}")
                continue

        # Cache the data
        if waveforms:
            self._save_to_cache(cache_file, waveforms)

        logger.info(f"Downloaded {len(waveforms)} waveforms for {event_name}")
        return waveforms

    def _save_to_cache(
        self,
        cache_file: Path,
        waveforms: List[SeismicWaveform],
    ) -> None:
        """Save waveforms to cache file."""
        data_dict = {
            "n_waveforms": len(waveforms),
        }

        for i, wf in enumerate(waveforms):
            prefix = f"wf_{i}_"
            data_dict[f"{prefix}data"] = wf.data
            data_dict[f"{prefix}times"] = wf.times
            data_dict[f"{prefix}sampling_rate"] = wf.sampling_rate
            data_dict[f"{prefix}station_network"] = wf.station.network
            data_dict[f"{prefix}station_name"] = wf.station.station
            data_dict[f"{prefix}station_lat"] = wf.station.latitude
            data_dict[f"{prefix}station_lon"] = wf.station.longitude
            data_dict[f"{prefix}channel"] = wf.channel

        np.savez_compressed(cache_file, **data_dict)
        logger.info(f"Cached {len(waveforms)} waveforms to {cache_file}")

    def _load_from_cache(
        self,
        cache_file: Path,
        event: SeismicEvent,
    ) -> List[SeismicWaveform]:
        """Load waveforms from cache file."""
        data = np.load(cache_file, allow_pickle=True)
        n_waveforms = int(data["n_waveforms"])

        waveforms = []
        for i in range(n_waveforms):
            prefix = f"wf_{i}_"

            station = SeismicStation(
                network=str(data[f"{prefix}station_network"]),
                station=str(data[f"{prefix}station_name"]),
                latitude=float(data[f"{prefix}station_lat"]),
                longitude=float(data[f"{prefix}station_lon"]),
                elevation_m=0.0,
                channels=[str(data[f"{prefix}channel"])],
            )

            waveform = SeismicWaveform(
                station=station,
                event=event,
                channel=str(data[f"{prefix}channel"]),
                data=data[f"{prefix}data"],
                times=data[f"{prefix}times"],
                sampling_rate=float(data[f"{prefix}sampling_rate"]),
                start_time=event.origin_time,
            )
            waveforms.append(waveform)

        return waveforms

    def preprocess_waveforms(
        self,
        waveforms: List[SeismicWaveform],
        detrend: bool = True,
        filter_band: Optional[Tuple[float, float]] = (0.01, 1.0),
        normalize: bool = True,
    ) -> List[SeismicWaveform]:
        """
        Preprocess waveforms for PINN training.

        Args:
            waveforms: List of raw waveforms
            detrend: Remove linear trend
            filter_band: Bandpass filter frequencies (Hz)
            normalize: Normalize to [-1, 1]

        Returns:
            Preprocessed waveforms
        """
        from scipy import signal

        processed = []

        for wf in waveforms:
            data = wf.data.copy()

            # Detrend
            if detrend:
                data = signal.detrend(data)

            # Bandpass filter
            if filter_band is not None:
                nyq = wf.sampling_rate / 2
                low = filter_band[0] / nyq
                high = min(filter_band[1] / nyq, 0.99)

                if low < high:
                    b, a = signal.butter(4, [low, high], btype="band")
                    data = signal.filtfilt(b, a, data)

            # Normalize
            if normalize:
                max_val = np.abs(data).max()
                if max_val > 0:
                    data = data / max_val

            # Create new waveform
            new_wf = SeismicWaveform(
                station=wf.station,
                event=wf.event,
                channel=wf.channel,
                data=data.astype(np.float32),
                times=wf.times,
                sampling_rate=wf.sampling_rate,
                start_time=wf.start_time,
                filtered=filter_band is not None,
                detrended=detrend,
            )
            processed.append(new_wf)

        return processed

    def to_training_data(
        self,
        waveforms: List[SeismicWaveform],
        event: SeismicEvent,
        downsample_factor: int = 10,
    ) -> Dict[str, np.ndarray]:
        """
        Convert waveforms to PINN training format.

        Args:
            waveforms: Preprocessed waveforms
            event: Seismic event
            downsample_factor: Factor to reduce temporal resolution

        Returns:
            Dictionary with coordinates and observed data
        """
        all_coords = []
        all_data = []

        for wf in waveforms:
            # Calculate distance and azimuth from event
            dist_km = self._haversine_distance(
                event.latitude, event.longitude,
                wf.station.latitude, wf.station.longitude,
            )

            azimuth = self._azimuth(
                event.latitude, event.longitude,
                wf.station.latitude, wf.station.longitude,
            )

            # Downsample
            times = wf.times[::downsample_factor]
            data = wf.data[::downsample_factor]

            # Create coordinate array [distance, azimuth, time]
            n_samples = len(times)
            coords = np.zeros((n_samples, 3), dtype=np.float32)
            coords[:, 0] = dist_km / 1000.0  # Normalize to ~[0, 20] range
            coords[:, 1] = azimuth / 360.0   # Normalize to [0, 1]
            coords[:, 2] = times / times.max()  # Normalize to [0, 1]

            all_coords.append(coords)
            all_data.append(data.reshape(-1, 1))

        return {
            "coordinates": np.vstack(all_coords),
            "observations": np.vstack(all_data),
            "event": {
                "latitude": event.latitude,
                "longitude": event.longitude,
                "depth_km": event.depth_km,
                "magnitude": event.magnitude,
            },
            "n_stations": len(waveforms),
        }

    @staticmethod
    def _haversine_distance(
        lat1: float, lon1: float,
        lat2: float, lon2: float,
    ) -> float:
        """Calculate great circle distance in km."""
        R = 6371.0  # Earth radius in km

        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    @staticmethod
    def _azimuth(
        lat1: float, lon1: float,
        lat2: float, lon2: float,
    ) -> float:
        """Calculate azimuth from point 1 to point 2 in degrees."""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1

        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

        azimuth = np.degrees(np.arctan2(x, y))
        return (azimuth + 360) % 360


class NOAADataLoader:
    """
    Data loader for NOAA tide gauge and DART buoy data.

    Downloads tsunami water level data from NOAA's Center for
    Tsunami Research and National Data Buoy Center.
    """

    # DART buoys active during 2004 tsunami
    DART_BUOYS_2004 = [
        "23401",  # Near Sumatra
        "46405",  # Eastern Pacific (for comparison)
    ]

    # Tide gauge stations with 2004 data
    TIDE_GAUGES_2004 = [
        ("colombo", "Sri Lanka"),
        ("male", "Maldives"),
        ("cocos", "Cocos Islands"),
        ("chennai", "India"),
        ("vishakhapatnam", "India"),
        ("port_blair", "Andaman Islands"),
    ]

    NOAA_BASE_URL = "https://nctr.pmel.noaa.gov"

    def __init__(self, cache_dir: str = "data/noaa_cache"):
        """Initialize NOAA data loader."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_dart_data(
        self,
        buoy_id: str = "46405",
        event_name: str = "sumatra_2004",
    ) -> Dict[str, np.ndarray]:
        """
        Download DART buoy data.

        Note: Direct download requires accessing NOAA NDBC archives.
        This provides a structured interface for the data.

        Args:
            buoy_id: DART buoy identifier
            event_name: Event name for context

        Returns:
            Dictionary with time series data
        """
        import requests

        # The actual NOAA data URL
        url = f"{self.NOAA_BASE_URL}/indo_1204.html"

        logger.info(f"DART buoy data can be downloaded from: {url}")
        logger.info("Direct data files available:")
        logger.info("  - nemo_sumatra.txt")
        logger.info("  - 46405_sumatra.txt")

        # Return placeholder with download instructions
        return {
            "source": "NOAA NCTR",
            "url": url,
            "buoy_id": buoy_id,
            "event": event_name,
            "data_files": ["nemo_sumatra.txt", "46405_sumatra.txt"],
            "instructions": (
                "Download data from NOAA NCTR website. "
                "Place files in data/noaa_cache/ directory."
            ),
        }

    def load_local_dart_file(
        self,
        filename: str,
    ) -> Dict[str, np.ndarray]:
        """
        Load locally downloaded DART data file.

        Args:
            filename: Name of the data file

        Returns:
            Dictionary with parsed data
        """
        filepath = self.cache_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(
                f"Data file not found: {filepath}. "
                f"Download from NOAA NCTR first."
            )

        # Parse NOAA format (typically columns: time, water_level)
        data = np.loadtxt(filepath, comments="#")

        return {
            "time": data[:, 0] if data.ndim > 1 else np.arange(len(data)),
            "water_level": data[:, 1] if data.ndim > 1 else data,
            "filename": filename,
        }


class SumatraVelocityModel:
    """
    Crustal velocity model for the Sumatra-Andaman subduction zone.

    Based on published seismic tomography studies of the region.
    """

    # Simplified 1D velocity profile (depth in km, Vp in km/s)
    # Based on Kennett & Engdahl (1991) and regional studies
    VELOCITY_PROFILE = [
        (0, 5.8),      # Upper crust
        (15, 6.5),     # Middle crust
        (35, 8.0),     # Upper mantle (Moho)
        (80, 8.1),     # Lithospheric mantle
        (220, 8.6),    # Asthenosphere
        (400, 9.0),    # Transition zone
    ]

    # Subduction zone specific parameters
    SLAB_DIP = 15  # degrees (shallow near trench)
    TRENCH_DEPTH_KM = 5

    def __init__(
        self,
        nx: int = 500,
        nz: int = 200,
        x_extent_km: float = 1000.0,
        z_extent_km: float = 400.0,
    ):
        """
        Initialize Sumatra velocity model.

        Args:
            nx: Grid points in x (along-strike)
            nz: Grid points in z (depth)
            x_extent_km: Model extent in x direction
            z_extent_km: Maximum depth
        """
        self.nx = nx
        self.nz = nz
        self.x_extent_km = x_extent_km
        self.z_extent_km = z_extent_km

        self.dx = x_extent_km / nx
        self.dz = z_extent_km / nz

    def generate(self) -> np.ndarray:
        """
        Generate 2D velocity model.

        Returns:
            2D array of P-wave velocities (km/s)
        """
        velocity = np.zeros((self.nz, self.nx), dtype=np.float32)

        # Create depth array
        z = np.linspace(0, self.z_extent_km, self.nz)

        # Fill with 1D profile
        for i, depth in enumerate(z):
            velocity[i, :] = self._get_velocity_at_depth(depth)

        # Add subduction zone structure
        velocity = self._add_subduction_slab(velocity)

        # Add some lateral heterogeneity
        velocity = self._add_heterogeneity(velocity)

        return velocity

    def _get_velocity_at_depth(self, depth: float) -> float:
        """Interpolate velocity at given depth."""
        depths = [d[0] for d in self.VELOCITY_PROFILE]
        velocities = [d[1] for d in self.VELOCITY_PROFILE]

        return np.interp(depth, depths, velocities)

    def _add_subduction_slab(self, velocity: np.ndarray) -> np.ndarray:
        """Add subducting slab structure."""
        x = np.linspace(0, self.x_extent_km, self.nx)
        z = np.linspace(0, self.z_extent_km, self.nz)

        # Slab top geometry (simplified)
        trench_x = self.x_extent_km * 0.2

        for i, xi in enumerate(x):
            if xi > trench_x:
                # Distance from trench
                dx = xi - trench_x
                # Slab depth (increases with distance)
                slab_top = self.TRENCH_DEPTH_KM + dx * np.tan(np.radians(self.SLAB_DIP))
                slab_bottom = slab_top + 80  # 80 km thick slab

                for j, zj in enumerate(z):
                    if slab_top < zj < slab_bottom:
                        # Slab is faster (cold oceanic lithosphere)
                        velocity[j, i] *= 1.05

        return velocity

    def _add_heterogeneity(
        self,
        velocity: np.ndarray,
        amplitude: float = 0.03,
    ) -> np.ndarray:
        """Add random heterogeneity."""
        # Create smooth random field
        from scipy.ndimage import gaussian_filter

        noise = np.random.randn(self.nz, self.nx)
        smooth_noise = gaussian_filter(noise, sigma=10)
        smooth_noise = smooth_noise / np.abs(smooth_noise).max()

        return velocity * (1 + amplitude * smooth_noise)

    def to_pinn_format(self) -> Dict[str, Any]:
        """
        Convert to PINN training format.

        Returns:
            Dictionary with velocity model data
        """
        velocity = self.generate()

        return {
            "velocity": velocity,
            "shape": velocity.shape,
            "dx_km": self.dx,
            "dz_km": self.dz,
            "x_extent_km": self.x_extent_km,
            "z_extent_km": self.z_extent_km,
            "stats": {
                "min": float(velocity.min()),
                "max": float(velocity.max()),
                "mean": float(velocity.mean()),
            },
            "region": "Sumatra-Andaman Subduction Zone",
        }


def download_sumatra_2004_data(
    duration_minutes: int = 60,
    cache_dir: str = "data/seismic_cache",
) -> Dict[str, Any]:
    """
    Convenience function to download 2004 Sumatra earthquake data.

    Args:
        duration_minutes: Duration of waveforms to download
        cache_dir: Cache directory

    Returns:
        Dictionary with event info, waveforms, and velocity model
    """
    # Initialize loaders
    iris_loader = IRISDataLoader(cache_dir)

    # Get event
    event = iris_loader.get_event("sumatra_2004")

    # Download waveforms
    logger.info("Downloading seismic waveforms from IRIS...")
    waveforms = iris_loader.download_event_data(
        "sumatra_2004",
        duration_minutes=duration_minutes,
    )

    # Preprocess
    logger.info("Preprocessing waveforms...")
    processed = iris_loader.preprocess_waveforms(waveforms)

    # Convert to training format
    training_data = iris_loader.to_training_data(processed, event)

    # Generate velocity model
    logger.info("Generating Sumatra velocity model...")
    vel_model = SumatraVelocityModel()
    velocity_data = vel_model.to_pinn_format()

    return {
        "event": {
            "name": event.name,
            "magnitude": event.magnitude,
            "latitude": event.latitude,
            "longitude": event.longitude,
            "depth_km": event.depth_km,
            "origin_time": event.origin_time.isoformat(),
        },
        "waveforms": {
            "n_stations": len(waveforms),
            "duration_minutes": duration_minutes,
            "preprocessed": True,
        },
        "training_data": training_data,
        "velocity_model": velocity_data,
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Downloading 2004 Sumatra earthquake data...")
    data = download_sumatra_2004_data(duration_minutes=30)

    print(f"\nEvent: {data['event']['name']}")
    print(f"Magnitude: {data['event']['magnitude']}")
    print(f"Stations: {data['waveforms']['n_stations']}")
    print(f"Training samples: {data['training_data']['coordinates'].shape[0]}")
    print(f"Velocity model shape: {data['velocity_model']['shape']}")
