"""
Physics-Informed Neural Networks for Seismic Waveform Inversion

A production-ready system combining deep learning with wave equation physics
for subsurface velocity model prediction.

Key Features:
- Multiple PINN architectures (SIREN, FourierNet, ModulatedSIREN)
- Acoustic and elastic wave equation constraints
- Multi-scale training with adaptive loss weighting
- Real-time training visualization via WebSocket
- Interactive 3D velocity model exploration
"""

__version__ = "1.0.0"
__author__ = "Seismic AI Research"

from .models import PINN, SIRENNetwork, FourierFeatureNetwork, ModulatedSIREN
from .physics import WaveEquation, AcousticWaveEquation, ElasticWaveEquation
from .training import PINNTrainer, AdaptiveLossWeighting
from .data import SeismicDataGenerator, VelocityModelGenerator

__all__ = [
    "PINN",
    "SIRENNetwork",
    "FourierFeatureNetwork",
    "ModulatedSIREN",
    "WaveEquation",
    "AcousticWaveEquation",
    "ElasticWaveEquation",
    "PINNTrainer",
    "AdaptiveLossWeighting",
    "SeismicDataGenerator",
    "VelocityModelGenerator",
]
