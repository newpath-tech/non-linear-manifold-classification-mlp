# Non-Linear Binary Classification: Multi-Layer Perceptron Benchmark on Interleaved Manifolds

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA Acceleration](https://img.shields.io/badge/CUDA-12.1%20Accelerated-76b900.svg)](https://developer.nvidia.com/cuda-zone)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-grade deep learning benchmark evaluating the representation capacity of Multi-Layer Perceptrons (MLPs) on non-linearly separable coordinate topologies (specifically Scikit-Learn's `make_moons` manifold). Built with PyTorch, CUDA-accelerated tensor execution, strict leakage-free stratified data pipelines, and reproducible experiment tracking.

---

## 1. Problem Formulation & Theoretical Foundations

### 1.1 Topological Non-Separability
Binary classification on the synthetic two-moons manifold involves partitioning interleaved, non-convex crescent distributions in $\mathbb{R}^2$:

$$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N, \quad \mathbf{x}_i \in \mathbb{R}^2, \quad y_i \in \{0, 1\}$$

Linear hyperplanes ($\mathbf{w}^T \mathbf{x} + b = 0$) fail inherently because the convex hulls of the two classes intersect ($\text{conv}(\mathcal{C}_0) \cap \text{conv}(\mathcal{C}_1) \neq \emptyset$). A single-layer perceptron exhibits an upper-bound accuracy limit of $\approx 85-88\%$ under standard Gaussian noise.

### 1.2 Manifold Folding via Non-Linear Representation
To resolve the entanglement, a multi-layer neural network applies successive affine transformations interleaved with non-linear activation functions $\phi(\cdot)$:

$$\mathbf{h}^{(l)} = \phi\left(\mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\right), \quad l \in \{1, \dots, L\}$$

By expanding the 2D input coordinates into intermediate latent spaces ($\mathbb{R}^2 \to \mathbb{R}^{64} \to \mathbb{R}^{32} \to \mathbb{R}^{16}$), the network deforms the coordinate manifold, transforming the entangled crescents into linearly separable clusters in the final representation layer.

### 1.3 Numerically Stable Loss Formulation
We compute Binary Cross-Entropy directly on raw logit activations $z$ rather than feeding sigmoid probabilities through a log transformation:

$$\mathcal{L}_{\text{BCE}}(y, z) = \max(z, 0) - z \cdot y + \log\left(1 + e^{-|z|}\right)$$

This formulation avoids gradient saturation, underflow, and floating-point overflow during backpropagation.

---

## 2. Benchmark Architectures

Three distinct model families are benchmarked under identical seeds and data splits:

```
[Input: R^2]
     │
     ├──> [Linear Perceptron] ──────> Linear(2 -> 1) ──────────────────────────────────> Output Logit (3 params)
     │
     ├──> [Shallow MLP] ────────────> Linear(2 -> 16) -> ReLU -> Linear(16 -> 1) ──────> Output Logit (65 params)
     │
     └──> [Deep Modular MLP] ───────> Linear(2 -> 64) -> LayerNorm -> GELU -> Dropout ─┐
                                      ┌────────────────────────────────────────────────┘
                                      └─> Linear(64 -> 32) -> LayerNorm -> GELU -> Dropout
                                      └─> Linear(32 -> 16) -> LayerNorm -> GELU -> Dropout
                                      └─> Linear(16 -> 1) ────────────────────────────> Output Logit (3,041 params)
```

1. **Linear Perceptron (Baseline):** Demonstrates linear boundary limitations.
2. **Shallow MLP (1-Hidden Layer):** Single hidden layer proving basic manifold folding.
3. **Deep Modular MLP (Proposed):** Multi-stage progressive feature extraction with Layer Normalization, GELU activations, and Dropout regularization.

---

## 3. Hardware & GPU Acceleration Architecture

To maximize computational throughput and eliminate CPU bottlenecks:
- **Direct GPU Memory Allocation:** Model parameters, forward passes, backpropagation gradients, and loss computations execute on `cuda:0` (NVIDIA RTX 4050 Laptop GPU).
- **Asynchronous DMA Transfer:** DataLoaders leverage page-locked host memory (`pin_memory=True`) and asynchronous transfers (`non_blocking=True`).
- **Parallelized Decision Grid Evaluation:** Decision boundary visualization requires evaluating dense 2D coordinate spaces ($600 \times 600 = 360,000$ points). Instead of slow CPU loops, coordinate grids are vectorized and evaluated in batched CUDA streaming multiprocessors in under 50ms.

---

## 4. Empirical Benchmark Results

Evaluated on an independent, unseen test partition ($N=450$, Gaussian noise $\sigma=0.15$):

| Architecture | Parameters | Test Accuracy | Test F1-Score | Test ROC-AUC | Brier Calibration |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Perceptron (Baseline)** | 3 | 88.44% | 0.8855 | 0.9633 | 0.0779 |
| **Shallow MLP (1-Hidden Layer)** | 65 | 99.11% | 0.9912 | 0.9999 | 0.0073 |
| **Deep Modular MLP (Proposed)** | 3,041 | **99.11%** | **0.9912** | **0.9999** | **0.0073** |

### Decision Boundary Synthesis
Visual comparison of the learned classification surfaces across models:
- **Linear Perceptron:** A single rigid planar partition bisecting the crescent clusters.
- **Shallow MLP:** A piecewise-linear curved boundary successfully bending between the moons.
- **Deep Modular MLP:** A smooth, continuous posterior probability gradient contour adhering strictly to the topological geometry without high-frequency over-fitting artifacts.

*(Generated figure saved at `reports/figures/model_comparison_decision_boundaries.png`)*

---

## 5. Repository Structure

```
COE-ANN/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated Flake8 linting & Pytest test runner
├── configs/
│   └── default_config.yaml      # Declarative hyperparameters (seed, data, model, hardware)
├── docs/
│   └── ENGINEERING_WORKFLOW_AND_SPECIFICATION.md  # Exhaustive technical process documentation
├── notebooks/
│   └── 01_decision_surface_analysis.ipynb  # Interactive workshop walkthrough
├── reports/
│   ├── figures/                 # Rendered decision manifolds & loss landscapes
│   └── benchmark_summary.json   # Machine-readable metric exports
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py           # Generation, stratified splitting, and zero-leakage scaling
│   ├── models/
│   │   ├── __init__.py
│   │   └── architectures.py     # LinearPerceptron, ShallowMLP, DeepModularMLP
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # CUDA-optimized training loop with EarlyStopping & Checkpointing
│   │   └── metrics.py           # Comprehensive classification metric suite
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py             # GPU-accelerated decision boundary & contour visualizer
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # YAML configuration loader
│       ├── device.py            # CUDA device acquisition & telemetry
│       └── seed.py              # Strict multi-backend determinism
├── tests/
│   ├── __init__.py
│   ├── test_data.py             # Split ratios, balance, zero-leakage tests
│   ├── test_models.py           # Forward pass, gradient flow, device allocation
│   └── test_trainer.py          # Training loop convergence & metrics tests
├── .gitignore
├── LICENSE                      # MIT License
├── Makefile                     # Developer commands
├── pyproject.toml               # Package build metadata
├── requirements.txt             # Locked dependencies
├── train.py                     # Main CLI training script
└── benchmark.py                 # Multi-model ablation benchmark runner
```

---

## 6. Getting Started

### 6.1 Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA drivers (optional, will safely fall back to CPU if unavailable)

### 6.2 Installation
Clone repository and install dependencies:
```bash
git clone <repository-url>
cd COE-ANN
pip install -r requirements.txt
```

### 6.3 Execute Unit Tests
Validate environment, data splitting, model gradients, and CUDA allocation:
```bash
pytest -v
```

### 6.4 Model Training
Train the default deep modular architecture:
```bash
python train.py --config configs/default_config.yaml
```

### 6.5 Run Architectural Benchmark
Execute the multi-model ablation benchmark across all architectures:
```bash
python benchmark.py
```

---

## 7. Quality Assurance & Engineering Standards

- **Strict Featurization Ordering:** The dataset is split into Train (70%), Validation (15%), and Test (15%) partitions **before** any feature scaling is fitted. Standardizers are fitted exclusively on the training subset.
- **Reproducibility:** Seed determinism is enforced across Python `random`, `numpy`, and `torch` (including `torch.backends.cudnn.deterministic = True`).
- **Continuous Integration:** Automated GitHub Actions pipeline (`.github/workflows/ci.yml`) validates linting (`flake8`, `black`) and executes the full `pytest` suite across Python 3.10 and 3.11.

---

## 8. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
