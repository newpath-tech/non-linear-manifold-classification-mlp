import argparse
import json
from pathlib import Path
import torch

from src.utils.seed import set_seed
from src.utils.device import get_compute_device, log_device_telemetry
from src.utils.config import load_config
from src.data.dataset import get_pipeline_data
from src.models.architectures import build_model
from src.training.trainer import ModelTrainer
from src.visualization.plots import (
    plot_decision_boundary,
    plot_training_history,
    plot_confusion_matrix_heatmap
)


def run_training(config_path: str = "configs/default_config.yaml"):
    # 1. Load Config & Seed
    config = load_config(config_path)
    set_seed(config["experiment"]["seed"])

    # 2. Configure Device
    device = get_compute_device(
        preferred=config["hardware"]["preferred_device"],
        gpu_id=config["hardware"]["gpu_id"]
    )
    log_device_telemetry(device)

    # 3. Ingest and Preprocess Data
    data_bundle = get_pipeline_data(config)
    print(f"[DATA PIPELINE] Generated {len(data_bundle.X_train) + len(data_bundle.X_val) + len(data_bundle.X_test)} samples.")
    print(f"                Train: {len(data_bundle.X_train)} | Val: {len(data_bundle.X_val)} | Test: {len(data_bundle.X_test)}")

    # 4. Construct Architecture
    model = build_model(config).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL CREATED] Architecture: {config['model']['architecture']} | Trainable Parameters: {param_count:,}")

    # 5. Initialize Trainer
    trainer = ModelTrainer(
        model=model,
        device=device,
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        lr_scheduler_type=config["training"]["lr_scheduler"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
        min_delta=config["training"]["min_delta"],
        checkpoints_dir=config["paths"]["checkpoints_dir"]
    )

    # 6. Fit Model
    history = trainer.fit(
        train_loader=data_bundle.train_loader,
        val_loader=data_bundle.val_loader,
        epochs=config["training"]["epochs"],
        checkpoint_name=f"{config['model']['architecture']}_best.pt",
        verbose=True
    )

    # 7. Evaluate on Unseen Test Partition
    metrics = trainer.evaluate_test_set(data_bundle.test_loader)
    print("=" * 60)
    print("[FINAL UNSEEN TEST EVALUATION]")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:<15} : {v:.4f}")
    print("=" * 60)

    # 8. Render GPU-Accelerated Visualizations
    fig_dir = Path(config["paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    arch_name = config["model"]["architecture"]
    history_fig = fig_dir / f"{arch_name}_training_history.png"
    plot_training_history(history, str(history_fig))
    print(f"[SAVED] Learning history curve: {history_fig}")

    boundary_fig = fig_dir / f"{arch_name}_decision_boundary.png"
    plot_decision_boundary(
        model=model,
        X=data_bundle.X_test,
        y=data_bundle.y_test,
        title=f"Non-Linear Decision Manifold: {arch_name}",
        save_path=str(boundary_fig),
        resolution=config["visualization"]["grid_resolution"],
        device=device
    )
    print(f"[SAVED] GPU-computed decision boundary: {boundary_fig}")

    cm_fig = fig_dir / f"{arch_name}_confusion_matrix.png"
    plot_confusion_matrix_heatmap(
        cm=metrics["confusion_matrix"],
        save_path=str(cm_fig),
        model_name=arch_name
    )
    print(f"[SAVED] Confusion matrix: {cm_fig}")

    # 9. Save Metrics Summary
    metrics_file = Path(config["paths"]["metrics_file"])
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"[SAVED] Metrics JSON report: {metrics_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP on Non-Linear Manifold")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    run_training(args.config)
