"""Manifold Gen: three independent geometry-aware inference components."""

from .adapters import MODELS, load_pipeline
from .oracle import OracleNoise
from .pipeline import ManifoldPipeline, plug
from .samg import SAMG, register_guidance
from .schedulers import ManifoldDDIMScheduler
from .z2 import Z2Sampling

__version__ = "0.1.0"
__all__ = [
    "SAMG",
    "OracleNoise",
    "Z2Sampling",
    "ManifoldPipeline",
    "plug",
    "MODELS",
    "load_pipeline",
    "ManifoldDDIMScheduler",
    "register_guidance",
]
