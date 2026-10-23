import json
from pathlib import Path
import torch

from src.utils.seed import set_seed
from src.utils.device import get_compute_device, log_device_telemetry
from src.utils.config import load_config
from src.data.dataset import get_pipeline_data
from src.models.architectures import LinearPerceptron, ShallowMLP, DeepModularMLP
from src.training.trainer import ModelTrainer
from src.visualization.plots import (
    plot_comparative_decision_boundaries,
    plot_decision_boundary,
    plot_training_history,
    plot_confusion_matrix_heatmap
)


def run_benchmark(config_path: str = "configs/default_config.yaml"):
    config = load_config(config_path)
    seed = config["experiment"]["seed"]
    set_seed(seed)

    device = get_compute_device(
        preferred=config["hardware"]["preferred_device"],
        gpu_id=config["hardware"]["gpu_id"]
    )
    log_device_telemetry(device)

    # Ingest and prepare data
    data_bundle = get_pipeline_data(config)
    print(f"\n[BENCHMARK DATASET] Total Samples: {len(data_bundle.X_train) + len(data_bundle.X_val) + len(data_bundle.X_test)}")
    print(f"  Train: {len(data_bundle.X_train)} | Val: {len(data_bundle.X_val)} | Test: {len(data_bundle.X_test)}")
    print(f"  Noise Level: sigma = {config['data']['noise']}")

    # Candidate Architectures
    models = {
        "Linear Perceptron (Baseline)": LinearPerceptron(input_dim=2, output_dim=1),
        "Shallow MLP (1-Hidden Layer)": ShallowMLP(input_dim=2, hidden_dim=16, output_dim=1),
        "Deep Modular MLP (Proposed)": DeepModularMLP(
            input_dim=2,
            hidden_dims=[64, 32, 16],
            output_dim=1,
            activation="gelu",
            dropout_rate=0.10,
            use_layer_norm=True
        )
    }

    benchmark_results = {}
    trained_models = {}

    print("\n" + "=" * 75)
    print("STARTING ARCHITECTURAL BENCHMARK ON CUDA ACCELERATOR")
    print("=" * 75)

    fig_dir = Path(config["paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        print(f"\n>>> Benchmarking Architecture: {name}")
        model = model.to(device)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        trainer = ModelTrainer(
            model=model,
            device=device,
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            lr_scheduler_type="cosine",
            early_stopping_patience=20,
            min_delta=1e-4,
            checkpoints_dir=config["paths"]["checkpoints_dir"]
        )

        safe_slug = name.split()[0].lower()
        history = trainer.fit(
            train_loader=data_bundle.train_loader,
            val_loader=data_bundle.val_loader,
            epochs=config["training"]["epochs"],
            checkpoint_name=f"{safe_slug}_benchmark_best.pt",
            verbose=False
        )

        metrics = trainer.evaluate_test_set(data_bundle.test_loader)
        metrics["parameters"] = param_count
        benchmark_results[name] = metrics
        trained_models[name] = model

        # Plot individual training history
        plot_training_history(history, str(fig_dir / f"{safe_slug}_history.png"))
        plot_confusion_matrix_heatmap(
            metrics["confusion_matrix"],
            str(fig_dir / f"{safe_slug}_cm.png"),
            model_name=name
        )

        print(f"    Trainable Params : {param_count:,}")
        print(f"    Test Accuracy    : {metrics['accuracy'] * 100:.2f}%")
        print(f"    Test ROC-AUC     : {metrics['roc_auc']:.4f}")
        print(f"    Test F1-Score    : {metrics['f1_score']:.4f}")
        print(f"    Test Brier Score : {metrics['brier_score']:.4f}")

    # Comparative decision boundary figure
    comp_fig = fig_dir / "model_comparison_decision_boundaries.png"
    plot_comparative_decision_boundaries(
        models_dict=trained_models,
        X=data_bundle.X_test,
        y=data_bundle.y_test,
        save_path=str(comp_fig),
        resolution=config["visualization"]["grid_resolution"],
        device=device
    )
    print(f"\n[GENERATED] Comparative decision boundary figure: {comp_fig}")

    # Save benchmark JSON summary
    summary_path = Path("reports/benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=4)
    print(f"[GENERATED] Benchmark summary JSON: {summary_path}")

    # Display comparison table
    print("\n" + "=" * 90)
    print(f"{'Model Architecture':<32} | {'Params':<8} | {'Accuracy':<10} | {'F1-Score':<10} | {'ROC-AUC':<10} | {'Brier':<8}")
    print("-" * 90)
    for name, res in benchmark_results.items():
        print(f"{name:<32} | {res['parameters']:<8} | {res['accuracy']*100:>8.2f}% | {res['f1_score']:>10.4f} | {res['roc_auc']:>10.4f} | {res['brier_score']:>8.4f}")
    print("=" * 90)


if __name__ == "__main__":
    run_benchmark()
