import pytest
import torch
import numpy as np
from src.data.dataset import MoonsDataset, build_dataloaders
from src.models.architectures import DeepModularMLP
from src.training.trainer import ModelTrainer
from src.training.metrics import compute_classification_metrics
from src.utils.device import get_compute_device


def test_compute_classification_metrics_perfect():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.05, 0.95, 0.10, 0.90, 0.01, 0.99])
    metrics = compute_classification_metrics(y_true, y_prob)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["brier_score"] < 0.05


def test_trainer_smoke_run(tmp_path):
    device = get_compute_device("cuda")
    
    # Synthetic mini-batch
    X = np.random.randn(64, 2).astype(np.float32)
    y = np.random.randint(0, 2, size=(64,)).astype(np.float32)

    tr_loader, va_loader, te_loader = build_dataloaders(
        X, y, X, y, X, y, batch_size=32, pin_memory=(device.type == "cuda")
    )

    model = DeepModularMLP(input_dim=2, hidden_dims=[16, 8], output_dim=1)
    trainer = ModelTrainer(
        model=model,
        device=device,
        learning_rate=0.01,
        early_stopping_patience=3,
        checkpoints_dir=str(tmp_path)
    )

    history = trainer.fit(tr_loader, va_loader, epochs=2, verbose=False)
    assert len(history.train_loss) == 2
    assert len(history.val_loss) == 2

    test_metrics = trainer.evaluate_test_set(te_loader)
    assert "accuracy" in test_metrics
    assert "roc_auc" in test_metrics
