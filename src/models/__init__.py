"""Neural network architectures for Physics-Informed Neural Networks."""

from .pinn import PINN, BackboneType
from .siren import SIRENNetwork, SineLayer
from .fourier import FourierFeatureNetwork, GaussianFourierFeatures
from .modulated import ModulatedSIREN, HyperNetwork
from .attention import SpatioTemporalAttention, CrossAttentionFusion

__all__ = [
    "PINN",
    "BackboneType",
    "SIRENNetwork",
    "SineLayer",
    "FourierFeatureNetwork",
    "GaussianFourierFeatures",
    "ModulatedSIREN",
    "HyperNetwork",
    "SpatioTemporalAttention",
    "CrossAttentionFusion",
]
