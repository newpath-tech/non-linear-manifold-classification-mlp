"""Neural network architectural models for non-linear manifold classification."""

from src.models.architectures import (
    LinearPerceptron,
    ShallowMLP,
    DeepModularMLP,
    build_model,
)

__all__ = [
    "LinearPerceptron",
    "ShallowMLP",
    "DeepModularMLP",
    "build_model",
]
