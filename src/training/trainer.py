from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.training.metrics import compute_classification_metrics


@dataclass
class TrainingHistory:
    """Stores epoch-by-epoch loss, accuracy, and learning rate trajectories."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)


class ModelTrainer:
    """Production-grade PyTorch training engine optimized for CUDA execution."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.005,
        weight_decay: float = 1e-4,
        lr_scheduler_type: str = "cosine",
        early_stopping_patience: int = 20,
        min_delta: float = 1e-4,
        checkpoints_dir: str = "checkpoints"
    ):
        self.device = device
        self.model = model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.lr_scheduler_type = lr_scheduler_type
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.best_state_dict = None
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.history = TrainingHistory()

    def _setup_scheduler(self, epochs: int):
        if self.lr_scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=1e-5
            )
        elif self.lr_scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5
            )
        else:
            self.scheduler = None

    def train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        """Execute one training epoch entirely on GPU."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for x_batch, y_batch in loader:
            # Asynchronous DMA transfer to GPU
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()
            self.optimizer.step()

            # Vectorized metric accumulation on GPU
            total_loss += loss.item() * x_batch.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total_samples += x_batch.size(0)

        epoch_loss = total_loss / total_samples
        epoch_acc = correct / total_samples
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """Execute validation pass on GPU."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            logits = self.model(x_batch)
            loss = self.criterion(logits, y_batch)

            total_loss += loss.item() * x_batch.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total_samples += x_batch.size(0)

        return total_loss / total_samples, correct / total_samples

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 150,
        checkpoint_name: str = "best_model.pt",
        verbose: bool = True
    ) -> TrainingHistory:
        """Run complete model training with early stopping and checkpointing."""
        self._setup_scheduler(epochs)
        if verbose:
            print(f"[TRAINING INITIATED] Target Epochs: {epochs} | Device: {self.device}")

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = self.train_epoch(train_loader)
            va_loss, va_acc = self.evaluate(val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history.train_loss.append(tr_loss)
            self.history.val_loss.append(va_loss)
            self.history.train_acc.append(tr_acc)
            self.history.val_acc.append(va_acc)
            self.history.learning_rates.append(current_lr)

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(va_loss)
                else:
                    self.scheduler.step()

            # Early stopping & Checkpointing
            if va_loss < (self.best_val_loss - self.min_delta):
                self.best_val_loss = va_loss
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                self.epochs_without_improvement = 0
                torch.save(self.best_state_dict, self.checkpoints_dir / checkpoint_name)
            else:
                self.epochs_without_improvement += 1

            if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
                print(
                    f"Epoch [{epoch:03d}/{epochs:03d}] | "
                    f"Train Loss: {tr_loss:.4f} - Acc: {tr_acc * 100:.2f}% | "
                    f"Val Loss: {va_loss:.4f} - Acc: {va_acc * 100:.2f}% | "
                    f"LR: {current_lr:.6f}"
                )

            if self.epochs_without_improvement >= self.early_stopping_patience:
                if verbose:
                    print(f"[EARLY STOPPING TRIGGERED] Epoch {epoch}. Best Val Loss: {self.best_val_loss:.4f}")
                break

        # Restore optimal model weights
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        return self.history

    @torch.no_grad()
    def predict_loader(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Obtain probabilities and true labels across an entire loader."""
        self.model.eval()
        all_probs: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device, non_blocking=True)
            probs = torch.sigmoid(self.model(x_batch))
            all_probs.append(probs.cpu())
            all_labels.append(y_batch)

        y_prob = torch.cat(all_probs, dim=0).numpy().ravel()
        y_true = torch.cat(all_labels, dim=0).numpy().ravel()
        return y_prob, y_true

    def evaluate_test_set(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Compute full benchmark metrics on the unseen test set."""
        y_prob, y_true = self.predict_loader(test_loader)
        metrics = compute_classification_metrics(y_true, y_prob)
        return metrics
