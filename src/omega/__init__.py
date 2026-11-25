"""
Omega v3.3 — Zero-dependency constrained decision & safety learning framework
=============================================================================

Public API exports.
"""

from .core import (
    # Enums
    Decision,
    Hypothesis,
    View,
    # Data classes
    OwnerHandle,
    OmegaConfig,
    Prediction,
    Input,
    Episode,
    OmegaState,
    # Core classes
    Boundary,
    StaticEnv,
    Omega,
    # Constants
    EPSILON,
    # Utility functions
    clamp,
    finite_number,
    normalize_vector,
    vector_difference,
    vector_error,
    canonical_json,
    sha256_text,
)

__version__ = "3.3.0"
__all__ = [
    "Decision",
    "Hypothesis",
    "View",
    "OwnerHandle",
    "OmegaConfig",
    "Prediction",
    "Input",
    "Episode",
    "OmegaState",
    "Boundary",
    "StaticEnv",
    "Omega",
    "EPSILON",
    "clamp",
    "finite_number",
    "normalize_vector",
    "vector_difference",
    "vector_error",
    "canonical_json",
    "sha256_text",
]