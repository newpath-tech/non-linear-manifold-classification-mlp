import numpy as np
import pytest
import torch
from src.data.dataset import (
    generate_moons_data,
    prepare_stratified_splits,
    scale_features,
    MoonsDataset,
    build_dataloaders
)


def test_generate_moons_data_shapes():
    n_samples = 1000
    noise = 0.2
    X, y = generate_moons_data(n_samples=n_samples, noise=noise, random_state=42)
    assert X.shape == (n_samples, 2)
    assert y.shape == (n_samples,)
    assert set(np.unique(y)) == {0, 1}


def test_stratified_split_ratios_and_balance():
    n_samples = 2000
    X, y = generate_moons_data(n_samples=n_samples, noise=0.15, random_state=42)
    X_tr, y_tr, X_va, y_va, X_te, y_te = prepare_stratified_splits(
        X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42
    )
    
    assert len(X_tr) == int(0.70 * n_samples)
    assert len(X_va) == int(0.15 * n_samples)
    assert len(X_te) == int(0.15 * n_samples)

    # Verify balance
    assert np.isclose(np.mean(y_tr), 0.5, atol=0.02)
    assert np.isclose(np.mean(y_va), 0.5, atol=0.02)
    assert np.isclose(np.mean(y_te), 0.5, atol=0.02)


def test_zero_leakage_scaling():
    n_samples = 1000
    X, y = generate_moons_data(n_samples=n_samples, noise=0.15, random_state=42)
    X_tr, y_tr, X_va, y_va, X_te, y_te = prepare_stratified_splits(
        X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42
    )
    X_tr_s, X_va_s, X_te_s, scaler = scale_features(X_tr, X_va, X_te)

    # Train set should have mean 0 and std 1
    assert np.allclose(np.mean(X_tr_s, axis=0), [0.0, 0.0], atol=1e-6)
    assert np.allclose(np.std(X_tr_s, axis=0), [1.0, 1.0], atol=1e-6)

    # Scaler mean and variance must match train set
    assert np.allclose(scaler.mean_, np.mean(X_tr, axis=0))


def test_moons_dataset_and_dataloaders():
    X = np.random.randn(100, 2).astype(np.float32)
    y = np.random.randint(0, 2, size=(100,)).astype(np.float32)
    ds = MoonsDataset(X, y)
    assert len(ds) == 100
    x_sample, y_sample = ds[0]
    assert x_sample.shape == (2,)
    assert y_sample.shape == (1,)
    assert isinstance(x_sample, torch.Tensor)
    assert isinstance(y_sample, torch.Tensor)
