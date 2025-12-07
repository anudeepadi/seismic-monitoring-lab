# PINN Seismic Inversion

Physics-Informed Neural Networks for Seismic Waveform Inversion and Velocity Model Prediction

## Overview

This project implements state-of-the-art Physics-Informed Neural Networks (PINNs) for seismic waveform inversion. It combines the power of deep learning with physical constraints from wave equations to predict subsurface velocity models from seismic data.

### Key Features

- **Multiple Neural Network Architectures**
  - SIREN (Sinusoidal Representation Networks)
  - Fourier Feature Networks
  - Modulated SIREN with velocity conditioning
  - Attention-based architectures

- **Comprehensive Physics**
  - Acoustic wave equation
  - Elastic wave equation
  - Viscoacoustic wave equation with attenuation
  - Multiple source types (Ricker wavelet, Gaussian, plane wave)
  - Boundary conditions (absorbing, PML, free surface)

- **Advanced Training**
  - Adaptive loss weighting (GradNorm, Uncertainty, SoftAdapt)
  - Curriculum learning
  - Multi-stage training
  - Mixed precision support

- **Production-Ready**
  - FastAPI backend with WebSocket support
  - React frontend with real-time visualizations
  - Docker deployment ready
  - Comprehensive checkpointing and experiment tracking

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- CUDA 12.1+ (optional, for GPU support)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pinn-seismic.git
cd pinn-seismic
```

2. Run the setup script:
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

Or manually:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (with CUDA support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..
```

### Running the Application

1. Start the backend API:
```bash
source venv/bin/activate
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

2. In a new terminal, start the frontend:
```bash
cd frontend
npm run dev
```

3. Open http://localhost:5173 in your browser

### Training from Command Line

```bash
# Basic training
python scripts/train.py --model siren --epochs 1000 --velocity-model marmousi

# With configuration file
python scripts/train.py --config configs/default.yaml

# With custom parameters
python scripts/train.py \
    --model fourier \
    --hidden-dim 512 \
    --num-layers 8 \
    --learning-rate 0.0001 \
    --batch-size 8192 \
    --epochs 2000 \
    --velocity-model marmousi \
    --adaptive-weights \
    --adaptive-method gradnorm
```

## Docker Deployment

### CPU-only:
```bash
docker-compose up --build
```

### With GPU support:
```bash
docker-compose -f docker-compose.gpu.yml up --build
```

Access the application at http://localhost:3000

## Project Structure

```
pinn-seismic/
├── src/
│   ├── models/           # Neural network architectures
│   │   ├── siren.py      # SIREN implementation
│   │   ├── fourier.py    # Fourier feature networks
│   │   ├── modulated.py  # Modulated/conditional networks
│   │   ├── attention.py  # Attention mechanisms
│   │   └── pinn.py       # Main PINN model
│   ├── physics/          # Physics constraints
│   │   ├── wave_equation.py
│   │   ├── sources.py
│   │   └── boundary.py
│   ├── data/             # Data generation and loading
│   │   ├── generators.py
│   │   ├── velocity_models.py
│   │   └── datasets.py
│   ├── training/         # Training utilities
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   ├── adaptive_weights.py
│   │   └── schedulers.py
│   ├── api/              # FastAPI backend
│   │   ├── main.py
│   │   ├── routes.py
│   │   ├── websocket_manager.py
│   │   └── training_manager.py
│   └── utils/            # Utilities
│       └── checkpointing.py
├── frontend/             # React frontend
│   ├── src/
│   │   ├── pages/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── stores/
│   └── ...
├── configs/              # Configuration files
├── scripts/              # Utility scripts
├── tests/                # Test suite
└── docker-compose.yml    # Docker configuration
```

## Model Architectures

### SIREN (Sinusoidal Representation Networks)

SIREN uses periodic sine activations which are ideal for representing signals with high-frequency content like seismic waves.

```python
from src.models.siren import SIRENNetwork

model = SIRENNetwork(
    input_dim=3,      # x, z, t
    output_dim=1,     # pressure
    hidden_dim=256,
    num_layers=6,
    omega_0=30.0,     # First layer frequency
)
```

### Fourier Feature Networks

Fourier features help the network learn high-frequency functions by mapping inputs to a higher-dimensional space.

```python
from src.models.fourier import FourierFeatureNetwork

model = FourierFeatureNetwork(
    input_dim=3,
    output_dim=1,
    hidden_dim=256,
    num_layers=6,
    num_frequencies=128,
    sigma=10.0,
)
```

### Modulated SIREN

Velocity-conditioned networks that can adapt to different velocity models.

```python
from src.models.modulated import VelocityConditionedPINN

model = VelocityConditionedPINN(
    coord_dim=3,
    output_dim=1,
    hidden_dim=256,
    num_layers=6,
    velocity_channels=64,
)
```

## Physics Configuration

### Wave Equations

The system supports multiple wave equation formulations:

- **Acoustic**: Simple pressure wave equation for fluids
- **Elastic**: Coupled P and S wave propagation for solids
- **Viscoacoustic**: Includes attenuation effects (Q factor)

### Velocity Models

Built-in velocity models:
- **Marmousi**: Industry-standard benchmark
- **Layered**: Horizontally layered medium
- **Salt Dome**: Salt body with complex geometry
- **Fault**: Faulted geological structure

## API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/training/start` | Start new training job |
| GET | `/api/training/jobs` | List all training jobs |
| GET | `/api/training/jobs/{id}` | Get job status |
| POST | `/api/training/jobs/{id}/pause` | Pause training |
| POST | `/api/training/jobs/{id}/resume` | Resume training |
| GET | `/api/models` | List saved models |
| POST | `/api/inference` | Run inference |

### WebSocket

Connect to `/ws/training/{job_id}` for real-time training updates.

## Configuration

See `configs/default.yaml` for all available options:

```yaml
model:
  type: siren
  hidden_dim: 256
  num_layers: 6

training:
  epochs: 1000
  batch_size: 4096
  learning_rate: 0.0001

physics:
  wave_equation: acoustic
  velocity_model: marmousi
```

## Performance Tips

1. **Use GPU**: Training is significantly faster on GPU
2. **Batch Size**: Larger batches improve throughput
3. **Mixed Precision**: Enable for faster training on modern GPUs
4. **Adaptive Weights**: Use GradNorm for balanced multi-task learning

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{pinn_seismic,
  title = {PINN Seismic Inversion},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/pinn-seismic}
}
```

## Acknowledgments

- SIREN: [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)
- Fourier Features: [Fourier Features Let Networks Learn High Frequency Functions](https://arxiv.org/abs/2006.10739)
- PINNs: [Physics-informed neural networks](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
