# PINN Seismic - Real-Time Tsunami Warning System

A physics-informed neural network platform combining deep learning with seismic wave physics for **real-time earthquake detection and tsunami early warning** in the Indian Ocean region.

---

## Tsunami Early Warning System

### Overview

This system provides **real-time seismic monitoring** focused on the Indian Ocean - the world's most tsunami-prone region. It integrates live data from global seismograph networks to detect potentially tsunamigenic earthquakes and provide early warnings.

### Key Tsunami Features

#### Real-Time Earthquake Detection
- **Live seismic streaming** from IRIS SeedLink network
- **8 monitoring stations** strategically positioned around the Indian Ocean:
  - PALK (Sri Lanka) - Central Indian Ocean coverage
  - COCO (Cocos Islands) - Eastern Indian Ocean
  - DGAR (Diego Garcia) - Central monitoring point
  - CHTO (Thailand) - Andaman Sea coverage
  - WRAB (Australia) - Southern detection
  - NWAO (Australia) - Southwest monitoring
  - TATO (Taiwan) - Northern boundary
  - MBWA (Australia) - Western Australia coast

#### Tsunami Potential Assessment
Earthquakes are automatically assessed for tsunami risk based on:
- **Magnitude threshold**: M7.0+ earthquakes flagged as high risk
- **Depth analysis**: Shallow events (<100km) pose greater tsunami risk
- **Location**: Submarine earthquakes in subduction zones prioritized
- **Historical patterns**: Comparison with known tsunamigenic events

#### Interactive 3D Globe Visualization
- **Mapbox GL** powered 3D globe with real-time updates
- **Station markers** showing live amplitude readings
- **Earthquake epicenters** with magnitude-scaled visualization
- **Impact radius zones**:
  - Severe (red) - Immediate danger zone
  - Moderate (orange) - Strong shaking expected
  - Light (yellow) - Felt but minimal damage
- **Tsunami wave propagation** animation for major events

#### Earthquake Simulation System
Test the warning system with historical scenarios:
1. **2004 Sumatra Earthquake** (M9.1) - The Boxing Day tsunami
2. **Andaman Mega Event** (M9.5) - Hypothetical worst-case scenario
3. **Bay of Bengal Event** (M8.2) - Regional impact simulation

#### Emergency Alert System
- **Visual alerts**: Red flashing warning banners
- **Audio alerts**: Multi-frequency emergency siren (Purge-style)
- **Browser notifications**: Push alerts for significant events
- **Real-time waveform display**: Live seismograph visualization

---

## Live Demo

- **Backend API**: https://web-production-09ae.up.railway.app
- **Frontend**: Deploy to Vercel (see deployment section)

---

## Screenshots

### Tsunami Warning Map
The main interface showing the Indian Ocean region with:
- Active seismic stations (green = online, red = alert)
- Recent earthquake markers
- Real-time waveform panel
- Event feed with tsunami assessments

### Simulation Mode
Test emergency protocols with historical earthquake scenarios including the devastating 2004 Sumatra event.

---

## Technical Architecture

### Real-Time Data Pipeline

```
IRIS SeedLink Server
        │
        ▼
┌─────────────────────┐
│  SeedLink Streamer  │ ◄── ObsPy client
│  (Python Backend)   │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  WebSocket Manager  │ ◄── FastAPI
│  /api/v1/streams    │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  React Frontend     │ ◄── Mapbox GL + Recharts
│  Real-time Updates  │
└─────────────────────┘
```

### API Endpoints

#### Streaming API (Tsunami System)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/streams/stations` | List monitored seismic stations |
| GET | `/api/v1/streams/events` | Recent earthquakes from USGS |
| GET | `/api/v1/streams/status` | Stream connection status |
| WS | `/api/v1/streams/live` | Real-time waveform WebSocket |

#### Seismic Data API
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/seismic/events` | Historical earthquake catalog |
| POST | `/api/v1/seismic/download` | Download event waveforms |
| GET | `/api/v1/seismic/velocity-model` | Regional velocity model |

---

## Physics-Informed Neural Networks

Beyond tsunami warnings, this platform includes advanced PINN capabilities for seismic research:

### Neural Network Architectures
- **SIREN**: Sinusoidal networks for wave representation
- **Fourier Features**: High-frequency function learning
- **Modulated SIREN**: Velocity-conditioned models
- **Attention-based**: For complex geological structures

### Wave Equation Physics
- Acoustic wave equation
- Elastic wave equation (P and S waves)
- Viscoacoustic with attenuation (Q factor)
- Multiple source types (Ricker wavelet, Gaussian)

### Training Features
- Adaptive loss weighting (GradNorm, Uncertainty)
- Multi-stage curriculum learning
- Real-time training visualization
- Mixed precision GPU support

---

## Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- CUDA 12.1+ (optional, for GPU)

### Quick Start

```bash
# Clone repository
git clone https://github.com/anudeepadi/bug-free-journey.git
cd bug-free-journey

# Setup Python environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start backend
python -m uvicorn src.api.main:app --reload --port 8000

# In new terminal - start frontend
cd frontend
npm install
npm run dev
```

Open http://localhost:5173/tsunami for the warning system.

---

## Deployment

### Backend (Railway)
Already deployed at: https://web-production-09ae.up.railway.app

### Frontend (Vercel)
1. Import repo to Vercel
2. Set root directory: `frontend`
3. Add environment variables:
   - `VITE_API_URL=https://web-production-09ae.up.railway.app`
   - `VITE_WS_URL=wss://web-production-09ae.up.railway.app`
4. Deploy

---

## Project Structure

```
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI app
│   │   ├── streaming_routes.py  # Tsunami/seismic streaming
│   │   └── seismic_routes.py    # Historical data API
│   ├── data/
│   │   └── seismic_data_loader.py  # IRIS/USGS data fetching
│   ├── models/                  # PINN architectures
│   ├── physics/                 # Wave equations
│   └── training/                # Training utilities
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   └── TsunamiMap.tsx   # Main warning interface
│   │   ├── hooks/
│   │   │   └── useSeismicStream.ts  # Real-time data hook
│   │   └── components/
│   └── ...
├── scripts/
│   └── test_seedlink.py         # SeedLink connection test
└── configs/
```

---

## Indian Ocean Tsunami Risk

### Why Focus on Indian Ocean?

The Indian Ocean hosts some of the world's most active subduction zones:

1. **Sunda Trench** (Indonesia) - Generated the 2004 M9.1 earthquake
2. **Andaman-Nicobar Subduction** - Continuous seismic activity
3. **Makran Subduction Zone** (Pakistan/Iran) - Historically tsunamigenic

### Historical Events
| Year | Location | Magnitude | Casualties |
|------|----------|-----------|------------|
| 2004 | Sumatra | 9.1 | 230,000+ |
| 1883 | Krakatoa | ~9.0 (volcanic) | 36,000+ |
| 1945 | Makran | 8.1 | 4,000+ |

This system aims to provide early warning for coastal communities in:
- Indonesia
- Sri Lanka
- India (Tamil Nadu, Andaman Islands)
- Thailand
- Malaysia
- Bangladesh
- Myanmar

---

## Configuration

### Tsunami Alert Thresholds (configurable)

```python
TSUNAMI_THRESHOLDS = {
    'magnitude_warning': 7.0,      # Yellow alert
    'magnitude_critical': 7.5,     # Orange alert
    'magnitude_emergency': 8.0,    # Red alert + siren
    'depth_shallow': 70,           # km - higher tsunami risk
    'depth_very_shallow': 30,      # km - critical tsunami risk
}
```

### Monitored Region
```python
INDIAN_OCEAN_BOUNDS = {
    'min_lat': -40,
    'max_lat': 30,
    'min_lon': 30,
    'max_lon': 130,
}
```

---

## Contributing

Contributions welcome! Priority areas:
1. Additional seismic station integration
2. Improved tsunami wave propagation models
3. Mobile app for alerts
4. SMS/email notification system
5. Historical event analysis tools

---

## License

MIT License - See LICENSE file

---

## Acknowledgments

- **IRIS** (Incorporated Research Institutions for Seismology) - SeedLink data
- **USGS** - Earthquake catalog and hazard data
- **NOAA** - Tsunami warning protocols
- **ObsPy** - Seismological data processing
- **Mapbox** - Globe visualization

---

## Emergency Contacts

In case of actual tsunami warning:
- **Indonesia**: BMKG - bmkg.go.id
- **India**: INCOIS - incois.gov.in
- **Sri Lanka**: DMC - www.dmc.gov.lk
- **Thailand**: TMD - www.tmd.go.th

**This is a research/educational tool. Always follow official government warnings.**
