# Engineering Workflow & Technical Execution Document

**Project:** Empirical Analysis of Multi-Layer Perceptron Architectures for Non-Linear Decision Boundary Synthesis on Interleaved Manifolds  
**Author:** Senior Artificial Neural Network (ANN) Systems & Research Engineer  
**Hardware Platform:** NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.1) / Python 3.10  
**Repository Standard:** Production-grade, CI/CD-tested, PyTorch 2.5.1  

---

## 1. Project Overview & Problem Formulation

### 1.1 Objective
The goal is to design, implement, benchmark, and deploy a multi-layer perceptron (MLP) neural network capable of resolving non-linearly separable classification topologies, specifically Scikit-learn's synthetic two-moons manifold (`make_moons`).

### 1.2 Mathematical Formulation
The dataset consists of two interleaved crescent distributions in Euclidean 2D space:
$$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N, \quad \mathbf{x}_i \in \mathbb{R}^2, \quad y_i \in \{0, 1\}$$

1. **Topological Non-Separability:**
   Linear classifiers solve for hyperplanes:
   $$\mathbf{w}^T \mathbf{x} + b = 0$$
   Because the convex hulls of class 0 and class 1 overlap in $\mathbb{R}^2$, linear classification exhibits a mathematical failure ceiling ($\approx 82-86\%$ maximum accuracy).
   
2. **Latent Space Manifold Folding:**
   An Artificial Neural Network parameterizes a non-linear composite mapping:
   $$f(\mathbf{x}; \boldsymbol{\theta}) = \sigma \left( \mathbf{W}^{(L)} \phi\left( \mathbf{W}^{(L-1)} \dots \phi\left( \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)} \right) \dots + \mathbf{b}^{(L-1)} \right) + b^{(L)} \right)$$
   Where each layer transforms and unfolds the non-linear manifold into higher dimensional feature spaces $\mathbb{R}^{d_l}$ where the classes become linearly separable.

3. **Loss Function (Numerically Stable BCE with Logits):**
   $$\mathcal{L}_{BCE}(y, z) = \max(z, 0) - z \cdot y + \log(1 + e^{-|z|})$$
   Evaluating on raw logits avoids the numerical instability of standard sigmoid calculation prior to logarithmic cross-entropy.

---

## 2. Hardware Acceleration & Compute Constraints

To adhere strictly to senior engineering performance criteria:
- **Primary Compute Device:** All tensor creation, backpropagation gradients, forward passes, parameter updates, and evaluation grids are bound directly to `cuda:0` (NVIDIA GeForce RTX 4050 Laptop GPU).
- **CPU Offloading Minimization:**
  - CPU usage is limited strictly to initial generator calls and serializing final summary statistics to disk.
  - Data transfer employs pinned host memory (`pin_memory=True`) with asynchronous CUDA memory transfer (`non_blocking=True`).
  - High-resolution 2D grid decision surface evaluations ($1,000 \times 1,000 = 1,000,000$ coordinate points) are vectorized and parallelized across CUDA streaming multiprocessors rather than computed on CPU loops.

---

## 3. Implementation Phases & Architecture

### Phase 1: Environment Scaffolding & Configuration Management
- Standardized directory layout separating `src/`, `configs/`, `tests/`, `reports/`, and `notebooks/`.
- Declarative configuration file (`configs/default_config.yaml`) defining all data, model, and training hyperparameters.
- Centralized seed manager guaranteeing strict reproducibility across Python `random`, `numpy`, and `torch`.
- Centralized device manager ensuring GPU precedence.

### Phase 2: Data Engineering & Leakage Prevention Pipeline
- Implementation of `src/data/dataset.py`.
- **Stratified Partitioning:** Strict 70% Train, 15% Validation, 15% Test split with deterministic stratification.
- **Leakage Prevention:** Standard feature normalization (Z-score scaling $\mathbf{z} = \frac{\mathbf{x} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$) fitted **strictly** on the training partition and applied statelessly to validation/test partitions.
- PyTorch custom Dataset and DataLoader wrapper with pinned memory enabled.
- Unit testing in `tests/test_data.py`.

### Phase 3: Modular ANN Architectures
- Implementation of `src/models/architectures.py`:
  - **Model 1: `LinearPerceptron`** ($2 \to 1$): Baseline demonstrating linear non-separability.
  - **Model 2: `ShallowMLP`** ($2 \to 16 \to 1$): Single-hidden-layer non-linear proof-of-concept.
  - **Model 3: `DeepModularMLP`** ($2 \to 64 \to 32 \to 16 \to 1$): Multi-layer deep representation learner with configurable activations (ReLU, LeakyReLU, GELU, Tanh), Layer Normalization, and Dropout.
- Unit testing in `tests/test_models.py` verifying parameter shapes, device transfers, and gradient flows.

### Phase 4: Production Training Engine & Verification
- Implementation of `src/training/trainer.py` and `src/training/metrics.py`.
- Features:
  - AdamW optimizer with weight decay regularization.
  - Cosine Annealing learning rate schedule (`CosineAnnealingLR`).
  - Validation-loss-driven early stopping with model checkpoint persistence (`best_model.pt`).
  - Full evaluation metrics suite: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier calibration score, and Confusion Matrix.
- Unit testing in `tests/test_trainer.py` to ensure convergence stability.

### Phase 5: High-Throughput GPU Decision Boundary Synthesis & Benchmarking
- Implementation of `src/visualization/plots.py`.
- CUDA-accelerated dense meshgrid inference producing probability gradient contour maps.
- Complete benchmark execution comparing all 3 architectures across noise profiles ($\sigma \in \{0.10, 0.20, 0.30\}$).
- Artifact generation: High-resolution PNGs in `reports/figures/` and metrics in `reports/metrics_summary.json`.

### Phase 6: Workshop Interactive Notebook & Documentation
- Creation of `notebooks/01_decision_surface_analysis.ipynb` formatted according to strict data science best practices.
- Exhaustive, publication-quality `README.md` with ASCII architecture maps, mathematical notation, benchmark comparisons, and CLI reproduction commands.

### Phase 7: Version Control & GitHub Readiness
- Git initialization, `.gitignore` validation, clean branching/commit history according to conventional commit guidelines.

---

## 4. Verification & Testing Matrix

| Test Suite | Target File | Verification Criteria |
| :--- | :--- | :--- |
| Data Split Integrity | `tests/test_data.py` | Exact proportions (70/15/15), exact class balance, zero statistical contamination across splits. |
| Model Forward & GPU Alloc | `tests/test_models.py` | Valid tensor shapes, correct logit range, successful allocation to `cuda:0`, clean backprop. |
| Training Engine Sanity | `tests/test_trainer.py` | Loss strictly decreases over 5 epochs; early stopping triggers appropriately. |
