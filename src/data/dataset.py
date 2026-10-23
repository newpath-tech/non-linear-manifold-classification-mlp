from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DataSplits:
    """Container holding raw/scaled feature arrays, targets, scaler, and loaders."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


class MoonsDataset(Dataset):
    """PyTorch Dataset wrapper for coordinate features and binary labels.
    
    Tensors are stored in float32 precision for fast GPU compute.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        if len(X) != len(y):
            raise ValueError(f"Feature count ({len(X)}) must match label count ({len(y)}).")
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def generate_moons_data(
    n_samples: int = 3000,
    noise: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthesize 2D interleaved crescent moons manifold.

    Args:
        n_samples: Total number of coordinate points generated.
        noise: Standard deviation of Gaussian noise added to coordinates.
        random_state: Seed for deterministic sampling.

    Returns:
        Tuple of (X, y) where X is shape (n_samples, 2) and y is shape (n_samples,).
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y


def prepare_stratified_splits(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Execute strict stratified partitioning into Train, Validation, and Test subsets.

    Args:
        X: Feature matrix of shape (N, 2).
        y: Binary target vector of shape (N,).
        train_ratio: Proportion allocated to training.
        val_ratio: Proportion allocated to validation.
        test_ratio: Proportion allocated to testing.
        random_state: Seed ensuring deterministic distribution.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0."

    # First split: Train vs (Val + Test)
    val_test_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=val_test_ratio,
        stratify=y,
        random_state=random_state
    )

    # Second split: Val vs Test
    test_relative_ratio = test_ratio / val_test_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_relative_ratio,
        stratify=y_temp,
        random_state=random_state
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Apply Z-score standardization strictly fitted on the training partition.

    Prevents any data leakage into validation and test sets.

    Args:
        X_train: Training coordinates.
        X_val: Validation coordinates.
        X_test: Testing coordinates.

    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, fitted_scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def build_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 64,
    pin_memory: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Construct PyTorch DataLoaders with memory pinning for asynchronous GPU transfers.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        X_test, y_test: Testing data.
        batch_size: Mini-batch size.
        pin_memory: Enable page-locked host memory for direct GPU DMA.
        num_workers: Multi-process loader count (0 minimizes CPU context-switch overhead).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_ds = MoonsDataset(X_train, y_train)
    val_ds = MoonsDataset(X_val, y_val)
    test_ds = MoonsDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False
    )

    return train_loader, val_loader, test_loader


def get_pipeline_data(config: Dict[str, Any]) -> DataSplits:
    """Convenience pipeline assembling generation, splitting, scaling, and loaders."""
    data_cfg = config["data"]
    train_cfg = config["training"]
    seed = config["experiment"]["seed"]

    # 1. Generate
    X, y = generate_moons_data(
        n_samples=data_cfg["n_samples"],
        noise=data_cfg["noise"],
        random_state=seed
    )

    # 2. Split
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_stratified_splits(
        X, y,
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        test_ratio=data_cfg["test_ratio"],
        random_state=seed
    )

    # 3. Scale without leakage
    X_tr_s, X_va_s, X_te_s, scaler = scale_features(X_train, X_val, X_test)

    # 4. DataLoaders
    train_loader, val_loader, test_loader = build_dataloaders(
        X_tr_s, y_train,
        X_va_s, y_val,
        X_te_s, y_test,
        batch_size=train_cfg["batch_size"],
        pin_memory=train_cfg["pin_memory"],
        num_workers=train_cfg["num_workers"]
    )

    return DataSplits(
        X_train=X_tr_s,
        y_train=y_train,
        X_val=X_va_s,
        y_val=y_val,
        X_test=X_te_s,
        y_test=y_test,
        scaler=scaler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )
