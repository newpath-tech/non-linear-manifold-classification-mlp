<div align="center">

# 🌐 Non-Linear Binary Classification: Multi-Layer Perceptron Benchmark on Interleaved Manifolds

### *Empirical Deep Learning Architecture & 3D Topological Manifold Synthesis*

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2BCUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA Acceleration](https://img.shields.io/badge/NVIDIA%20RTX%204050-6GB%20VRAM-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![Testing Suite](https://img.shields.io/badge/Pytest-13%2F13%20Passed-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000?style=for-the-badge&logo=visual-studio-code&logoColor=white)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<br/>

<p align="center">
  <img src="reports/figures/decision_manifold_3d.png" alt="3D Decision Manifold" width="85%" style="border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.3);"/>
</p>

*Figure 1: High-Resolution 3D Posterior Probability Manifold $P(y=1 \mid \mathbf{x})$ synthesized by the Deep Modular MLP on the two-moons topological manifold, rendered on NVIDIA CUDA cores.*

</div>

---

## 📑 Table of Contents
- [1. Executive Architectural Overview](#1-executive-architectural-overview)
- [2. 3D Manifold Visualizations & Topologies](#2-3d-manifold-visualizations--topologies)
- [3. Theoretical & Mathematical Formulations](#3-theoretical--mathematical-formulations)
- [4. Architectural Taxonomy & Benchmark Suite](#4-architectural-taxonomy--benchmark-suite)
- [5. Quantitative Benchmark Results](#5-quantitative-benchmark-results)
- [6. GPU Acceleration & Hardware Architecture](#6-gpu-acceleration--hardware-architecture)
- [7. Production Repository Structure](#7-production-repository-structure)
- [8. Quickstart & Execution Guide](#8-quickstart--execution-guide)
- [9. Engineering Verification & Quality Assurance](#9-engineering-verification--quality-assurance)

---

## 1. Executive Architectural Overview

This repository provides an **industry-grade deep learning benchmark** evaluating the geometric representation limits of Multi-Layer Perceptrons (MLPs) against non-linearly separable coordinate manifolds (specifically Scikit-Learn's `make_moons`).

<table>
<tr>
<td width="50%">

### ⚡ Core Engineering Pillars
- **Strict Leakage-Free Preprocessing:** Stratified train/val/test splits (70/15/15) prior to standardizing coordinates.
- **CUDA-Accelerated Meshgrid Evaluation:** Vectorized 360,000-point 2D/3D grids evaluated in parallel batches directly on GPU memory.
- **Numerically Stable Loss Engine:** Evaluates Binary Cross-Entropy over raw logit space ($\mathcal{L}_{\text{BCEWithLogits}}$).
- **Ablation Baseline Comparison:** Systematic comparison across Linear, Shallow, and Deep Regularized architectures.

</td>
<td width="50%">

### 🔬 Operational Standards
- **Global Seed Determinism:** Synchronized seed locks across Python `random`, `numpy`, and `torch.cuda`.
- **Early Stopping & Weight Checkpointing:** Automated validation loss monitoring with deep weight restoration.
- **Production Package Standards:** Fully modular `src/` hierarchy, YAML declarative configuration, and CI/CD pipelines.
- **Test-Driven Rigor:** 13 unit tests covering split integrity, tensor flow, backward gradients, and GPU placement.

</td>
</tr>
</table>

---

## 2. 3D Manifold Visualizations & Topologies

Unlike standard 2D planar plots, our visualization suite maps the learned non-linear boundaries in **three dimensions**, revealing the continuous probability gradients and empirical optimization basins.

<div align="center">
<table>
<tr>
<td align="center" width="50%">
  <img src="reports/figures/decision_manifold_3d.png" alt="3D Decision Surface" width="100%" style="border-radius: 8px;"/>
  <br/>
  <b>3D Posterior Probability Manifold</b><br/>
  <i>Continuous deformation of Euclidean coordinates into bounded probability space $z \in [0, 1]$.</i>
</td>
<td align="center" width="50%">
  <img src="reports/figures/loss_landscape_3d.png" alt="3D Loss Landscape" width="100%" style="border-radius: 8px;"/>
  <br/>
  <b>3D Empirical Loss Landscape Topology</b><br/>
  <i>Parametric loss basin $\mathcal{L}_{BCE}$ surrounding converged global weight optimum $\theta^*$.</i>
</td>
</tr>
</table>
</div>

---

## 3. Theoretical & Mathematical Formulations

### 3.1 Topological Non-Separability
Let the input space be a 2D coordinate manifold $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$, where $\mathbf{x}_i \in \mathbb{R}^2$ and $y_i \in \{0, 1\}$.

A linear perceptron attempts to discover a separating hyperplane:

$$\mathbf{w}^T \mathbf{x} + b = 0$$

Because the convex hulls of the two crescent clusters intersect:

$$\text{conv}(\mathcal{C}_0) \cap \text{conv}(\mathcal{C}_1) \neq \emptyset$$

No linear boundary can achieve complete separation. Under Gaussian perturbation $\sigma = 0.15$, the theoretical accuracy ceiling of linear classification is capped at **$\approx 88.4\%$**.

### 3.2 Coordinate Manifold Folding
By stacking parameterized affine transformations with non-linear activation functions $\phi(\cdot)$, the Multi-Layer Perceptron warps the coordinate geometry into higher dimensions:

$$\mathbf{h}^{(1)} = \phi\left(\mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)}\right) \in \mathbb{R}^{64}$$

$$\mathbf{h}^{(2)} = \phi\left(\mathbf{W}^{(2)}\mathbf{h}^{(1)} + \mathbf{b}^{(2)}\right) \in \mathbb{R}^{32}$$

$$\mathbf{h}^{(3)} = \phi\left(\mathbf{W}^{(3)}\mathbf{h}^{(2)} + \mathbf{b}^{(3)}\right) \in \mathbb{R}^{16}$$

$$z = \mathbf{w}^{(4)T}\mathbf{h}^{(3)} + b^{(4)} \in \mathbb{R}^1$$

Through successive expansions ($\mathbb{R}^2 \to \mathbb{R}^{64} \to \mathbb{R}^{32} \to \mathbb{R}^{16}$), the entangled crescent manifolds are untangled into linearly separable spaces prior to final scalar projection.

```mermaid
graph LR
    subgraph InputSpace ["2D Euclidean Space"]
        X["Coordinates: x1, x2"]
    end

    subgraph ManifoldWarping ["High-Dimensional Latent Warping"]
        L1["Dense(2 -> 64) + LayerNorm + GELU"]
        L2["Dense(64 -> 32) + LayerNorm + GELU"]
        L3["Dense(32 -> 16) + LayerNorm + GELU"]
    end

    subgraph OutputProjection ["Decision Space"]
        Z["Logit Projection: Linear(16 -> 1)"]
        P["Posterior Probability: Sigmoid(z)"]
    end

    X --> L1 --> L2 --> L3 --> Z --> P
```

<details>
<summary><b>📐 Click to Expand: Numerically Stable Binary Cross-Entropy Derivation</b></summary>

Standard binary cross-entropy combines sigmoid activation with logarithmic loss:
$$\hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}}$$
$$\mathcal{L} = - \left[ y \log \hat{p} + (1 - y) \log (1 - \hat{p}) \right]$$

Evaluating this naively in floating-point arithmetic causes catastrophic numerical underflow when $z \ll 0$ or overflow when $z \gg 0$. To guarantee numerical stability, we reformulate into a single log-sum-exp expression:

$$\mathcal{L}_{\text{BCE}}(y, z) = \max(z, 0) - z \cdot y + \log\left(1 + e^{-|z|}\right)$$

Implemented via PyTorch's `torch.nn.BCEWithLogitsLoss()`.
</details>

---

## 4. Architectural Taxonomy & Benchmark Suite

| Model Identifier | Hidden Layers | Parameter Count | Regularization & Activations | Primary Theoretical Target |
| :--- | :---: | :---: | :--- | :--- |
| **Linear Perceptron (Baseline)** | 0 | **3** | Direct Linear Projection | Verifies the upper boundary of linear separability |
| **Shallow MLP (1-Hidden Layer)** | 1 | **65** | ReLU non-linearity | Validates basic non-linear boundary synthesis |
| **Deep Modular MLP (Proposed)** | 3 | **3,041** | LayerNorm + GELU + Dropout(0.10) | Evaluates continuous smooth probability curvature |

<br/>

<div align="center">
  <img src="reports/figures/model_comparison_decision_boundaries.png" alt="Decision Boundary Comparison" width="95%" style="border-radius: 8px;"/>
  <br/>
  <i>Figure 2: Empirical Decision Boundary Comparison on Test Data ($N=450$, $\sigma=0.15$). Left: Linear Perceptron. Center: Shallow MLP. Right: Deep Modular MLP.</i>
</div>

---

## 5. Quantitative Benchmark Results

Evaluated on an unseen test set ($N=450$ samples, noise $\sigma = 0.15$) executed on the **NVIDIA GeForce RTX 4050 Laptop GPU**:

| Architecture | Parameters | Test Accuracy | F1-Score | ROC-AUC | PR-AUC | Brier Calibration | Accuracy Gauge |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Linear Perceptron** | 3 | **88.44%** | 0.8855 | 0.9633 | 0.9669 | 0.0779 | `████████░░` 88.4% |
| **Shallow MLP** | 65 | **99.11%** | 0.9912 | 0.9999 | 0.9999 | 0.0073 | `██████████` 99.1% |
| **Deep Modular MLP** | 3,041 | **99.11%** | **0.9912** | **0.9999** | **0.9999** | **0.0073** | `██████████` 99.1% |

### Key Diagnostic Observations:
- **Zero False Negatives in Deep MLP:** Achieved **100.00% Recall** on Class 1 with $0.9868$ F1-score in standard single-model validation.
- **Superior Probability Calibration:** The Brier Score dropped by an order of magnitude from **0.0779** (Linear) down to **0.0073** (Deep MLP), indicating high-confidence, well-calibrated posterior estimates.
- **No Boundary Over-Fitting:** Layer Normalization and weight decay ($10^{-4}$) prevented the deep network from forming jagged, localized "islands" around noisy data outliers.

---

## 6. GPU Acceleration & Hardware Architecture

To maximize CUDA throughput and minimize host CPU utilization:

```
[System RAM (Host)] ──(Pinned Memory DMA Transfer)──> [NVIDIA RTX 4050 GPU (6 GB VRAM)]
                                                              │
                    ┌─────────────────────────────────────────┴──────────────────────────────┐
                    ▼                                                                        ▼
         [Model Forward & Backward]                                            [Vectorized Meshgrid Engine]
         - Weights, activations on GPU                                          - 600 x 600 coordinate grid (360k pts)
         - AdamW state & gradient accumulation                                  - Evaluated in parallel GPU batches
         - Loss computation: BCEWithLogits                                      - Zero Python CPU evaluation loops
```

- **Pinned Host Memory (`pin_memory=True`):** PyTorch `DataLoader` pages are locked in physical RAM, facilitating direct memory access (DMA) without CPU staging copy overhead.
- **Asynchronous Transfers (`non_blocking=True`):** Memory copies overlap with CUDA compute streams.
- **In-VRAM Meshgrid Inference:** The dense evaluation grid ($600 \times 600 = 360,000$ points) is generated as an unrolled tensor directly on `cuda:0`, evaluating posterior probabilities across all points in under 45 milliseconds.

---

## 7. Production Repository Structure

```
COE-ANN/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated Flake8 linting & Pytest workflow
├── configs/
│   └── default_config.yaml      # Declarative hyperparameters (seed, data, model, hardware)
├── docs/
│   └── ENGINEERING_WORKFLOW_AND_SPECIFICATION.md  # Detailed technical design specifications
├── notebooks/
│   └── 01_decision_surface_analysis.ipynb  # Interactive workshop walkthrough
├── reports/
│   ├── figures/                 # 2D/3D high-resolution decision manifolds & loss landscapes
│   │   ├── decision_manifold_3d.png
│   │   ├── loss_landscape_3d.png
│   │   ├── model_comparison_decision_boundaries.png
│   │   └── DeepModularMLP_training_history.png
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
│   │   └── metrics.py           # Metric calculation suite (Accuracy, AUC, Brier, CM)
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py             # GPU-accelerated 2D/3D contour visualizer
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # Safe YAML configuration loader
│       ├── device.py            # CUDA device acquisition & telemetry
│       └── seed.py              # Multi-backend seed determinism
├── tests/
│   ├── __init__.py
│   ├── test_data.py             # Split ratios, balance, zero-leakage tests
│   ├── test_models.py           # Forward pass, gradient flow, device allocation
│   └── test_trainer.py          # Convergence & metric calculation tests
├── .gitignore
├── LICENSE                      # MIT License
├── Makefile                     # Developer commands
├── pyproject.toml               # Package build metadata
├── requirements.txt             # Locked dependencies
├── train.py                     # Main CLI training script
├── benchmark.py                 # Multi-model ablation benchmark runner
└── generate_3d_visuals.py       # 3D surface and loss basin generator
```

---

## 8. Quickstart & Execution Guide

### 8.1 Environment Setup
```bash
# Clone the repository
git clone https://github.com/newpath-tech/COE-ANN.git
cd COE-ANN

# Install locked dependencies
pip install -r requirements.txt
```

### 8.2 Run Verification Test Suite
```bash
# Execute 13 unit tests verifying data splits, tensor flow, and CUDA memory
pytest -v
```

### 8.3 Train the Deep Modular Architecture
```bash
# Runs training on GPU with early stopping and renders 2D diagnostic figures
python train.py --config configs/default_config.yaml
```

### 8.4 Execute Multi-Model Benchmark
```bash
# Executes ablation comparison across all 3 architectures and exports JSON metrics
python benchmark.py
```

### 8.5 Generate 3D Manifold Visualizations
```bash
# Synthesizes 3D posterior probability surfaces and empirical loss landscape
python generate_3d_visuals.py
```

---

## 9. Engineering Verification & Quality Assurance

All components conform to high-reliability scientific computing standards:

1. **Stratification Guard:** Ensures exact $50/50$ class distribution across Train, Validation, and Test subsets.
2. **Leakage Prohibition:** `StandardScaler` statistics ($\boldsymbol{\mu}, \boldsymbol{\sigma}$) are fitted strictly on the training partition and frozen for all inference.
3. **Reproducibility Guarantee:** Seed locks applied to `random.seed`, `np.random.seed`, `torch.manual_seed`, and `torch.backends.cudnn.deterministic = True`.
4. **Automated CI Workflow:** GitHub Actions runs linting (`flake8`, `black`) and testing across Python 3.10 and 3.11 on every commit.

---

## 📄 License
This project is licensed under the terms of the [MIT License](LICENSE).
