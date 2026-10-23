"""Data ingestion, synthesis, stratification, and PyTorch dataset modules."""

from src.data.dataset import (
    generate_moons_data,
    prepare_stratified_splits,
    scale_features,
    MoonsDataset,
    build_dataloaders,
    DataSplits
)

__all__ = [
    "generate_moons_data",
    "prepare_stratified_splits",
    "scale_features",
    "MoonsDataset",
    "build_dataloaders",
    "DataSplits",
]
