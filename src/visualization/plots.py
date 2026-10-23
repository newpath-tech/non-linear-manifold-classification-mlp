from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Clean visual style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


@torch.no_grad()
def evaluate_decision_grid_gpu(
    model: nn.Module,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    resolution: int = 600,
    device: Optional[torch.device] = None,
    batch_size: int = 50000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate 2D decision boundary grid directly on the GPU.

    Vectors millions of points across CUDA cores, strictly avoiding slow CPU Python loops.

    Args:
        model: Evaluated PyTorch model.
        x_min, x_max, y_min, y_max: Coordinate bounding limits.
        resolution: Grid resolution (e.g., 600 yields 360,000 evaluations).
        device: CUDA device index.
        batch_size: Parallel chunk size sent to GPU.

    Returns:
        Tuple of (xx, yy, zz) where zz contains sigmoid probabilities of shape (res, res).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution, dtype=np.float32),
        np.linspace(y_min, y_max, resolution, dtype=np.float32)
    )
    grid_coords = np.column_stack([xx.ravel(), yy.ravel()])
    total_points = len(grid_coords)
    
    probs_list = []
    for i in range(0, total_points, batch_size):
        chunk = torch.from_numpy(grid_coords[i:i + batch_size]).to(device, non_blocking=True)
        logits = model(chunk)
        probs = torch.sigmoid(logits)
        probs_list.append(probs.cpu())

    zz = torch.cat(probs_list, dim=0).numpy().reshape(xx.shape)
    return xx, yy, zz


def plot_decision_boundary(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    save_path: str,
    resolution: int = 600,
    device: Optional[torch.device] = None
) -> None:
    """Render high-resolution probability contour map and coordinate scatter."""
    margin = 0.5
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin

    xx, yy, zz = evaluate_decision_grid_gpu(
        model, x_min, x_max, y_min, y_max, resolution=resolution, device=device
    )

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Probability surface contour
    contour = ax.contourf(xx, yy, zz, levels=30, cmap="RdBu_r", alpha=0.85)
    ax.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=2.0, linestyles="--")

    # Data scatter overlay
    scatter = ax.scatter(
        X[:, 0], X[:, 1],
        c=y,
        cmap=plt.cm.Spectral,
        edgecolors="k",
        s=30,
        linewidths=0.5,
        alpha=0.8
    )

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Posterior Probability $P(y=1 | \\mathbf{x})$", fontsize=11)
    
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Normalized Feature $x_1$", fontsize=11)
    ax.set_ylabel("Normalized Feature $x_2$", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_history(history: Any, save_path: str) -> None:
    """Plot dual-axis loss and accuracy learning trajectories across epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    epochs = range(1, len(history.train_loss) + 1)

    # Loss trajectory
    ax1.plot(epochs, history.train_loss, label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, history.val_loss, label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    ax1.set_title("Optimization Objective: Binary Cross-Entropy", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Training Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Accuracy trajectory
    ax2.plot(epochs, [acc * 100 for acc in history.train_acc], label="Train Acc", color="#2ca02c", linewidth=2)
    ax2.plot(epochs, [acc * 100 for acc in history.val_acc], label="Val Acc", color="#d62728", linewidth=2, linestyle="--")
    ax2.set_title("Empirical Classification Accuracy (%)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Training Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_comparative_decision_boundaries(
    models_dict: Dict[str, nn.Module],
    X: np.ndarray,
    y: np.ndarray,
    save_path: str,
    resolution: int = 500,
    device: Optional[torch.device] = None
) -> None:
    """Generate a multi-panel comparison of decision boundaries across architectures."""
    margin = 0.5
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin

    num_models = len(models_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5.5), dpi=300)
    if num_models == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models_dict.items()):
        xx, yy, zz = evaluate_decision_grid_gpu(
            model, x_min, x_max, y_min, y_max, resolution=resolution, device=device
        )
        ax.contourf(xx, yy, zz, levels=30, cmap="RdBu_r", alpha=0.85)
        ax.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=2.0, linestyles="--")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors="k", s=20, linewidths=0.4, alpha=0.7)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("$x_1$", fontsize=10)
        ax.set_ylabel("$x_2$", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix_heatmap(
    cm: list,
    save_path: str,
    model_name: str = "Deep Modular MLP"
) -> None:
    """Render annotated confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Class 0", "Class 1"],
        yticklabels=["Class 0", "Class 1"],
        ax=ax, annot_kws={"size": 13, "weight": "bold"}
    )
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Class", fontsize=11)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
