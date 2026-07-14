"""Encoders: map real-valued inputs to spike trains."""
from .encoder import (
    PoissonRateEncoder,
    LatencyEncoder,
    ConstantEncoder,
    LearnedEncoder,
)
from .decoder import RateDecoder, LastStepDecoder, AttentionDecoder

__all__ = [
    "PoissonRateEncoder",
    "LatencyEncoder",
    "ConstantEncoder",
    "LearnedEncoder",
    "RateDecoder",
    "LastStepDecoder",
    "AttentionDecoder",
]
