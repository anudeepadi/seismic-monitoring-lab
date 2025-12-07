"""Physics modules for wave equation constraints."""

from .wave_equation import (
    WaveEquation,
    AcousticWaveEquation,
    ElasticWaveEquation,
    ViscoacousticWaveEquation,
)
from .sources import (
    SourceFunction,
    RickerWavelet,
    GaussianSource,
    PointSource,
    PlaneWaveSource,
)
from .boundary import (
    BoundaryCondition,
    AbsorbingBoundary,
    FreeSurfaceBoundary,
    PMLBoundary,
)

__all__ = [
    "WaveEquation",
    "AcousticWaveEquation",
    "ElasticWaveEquation",
    "ViscoacousticWaveEquation",
    "SourceFunction",
    "RickerWavelet",
    "GaussianSource",
    "PointSource",
    "PlaneWaveSource",
    "BoundaryCondition",
    "AbsorbingBoundary",
    "FreeSurfaceBoundary",
    "PMLBoundary",
]
