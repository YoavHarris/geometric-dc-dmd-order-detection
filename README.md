# DMD Order Detection

Research codebase for **Dynamic Mode Decomposition (DMD) order detection** using delay-coordinate embeddings and mode classification algorithms.

## Overview

This repository implements and evaluates methods for automatically identifying the correct number of modes in DMD analysis, particularly in the presence of noise and delay-coordinate (Hankel) embeddings. The system includes:

- **Core DMD implementation** with delay embedding
- **Order detection algorithms** (clustering-based, information criteria)
- **PBS experiment framework** for large-scale HPC experiments
- **Analysis & plotting tools** for paper-ready figures

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic DMD Example

```python
from dmd.dmd_tools import fit_dmd
from utils.data_generation import DMDDataGenerator

# Generate synthetic data
generator = DMDDataGenerator(
    eigenvalue_magnitude=0.98,
    frequency_separation=0.03,
    snr_db=10.0
)
X, X_clean, eigs_true, modes_true, amps = generator.generate(
    n_spatial=50, n_timesteps=200, n_modes=3
)

# Fit DMD with delay embedding
dmd = fit_dmd(data=X, svd_rank=10, num_delays=5)

# Access results
print(f"Eigenvalues: {dmd.eigenvalues}")
print(f"Modes shape: {dmd.modes.shape}")
```

## Repository Structure

```
dc-dmd-order-detection/
├── dmd/                      # Core DMD implementation
│   └── dmd_tools.py         # DMD, delay embedding, reconstruction
├── algorithms/              # Order detection methods
│   ├── bic.py              # BIC (Bayesian Information Criterion)
│   ├── clustering.py       # 2-cluster mode classification
│   ├── stc.py              # Spatiotemporal Coupling (STC)
│   ├── kronecker_vandermonde_fit.py  # NestedDMD, FixedEigenvalueKVFit
│   └── estimated_subspace_leakage.py  # ESL-Norm
├── utils/                   # Data generation & visualization
│   ├── data_generation.py  # Synthetic DMD data with noise
│   ├── dmd_utils.py        # DMD fitting utilities
│   └── visualizations.py   # Plotting helpers
├── analysis/               # Experiment scripts
│   └── spurious_eigenvalues_and_L/  # Embedding length analysis
├── figures/                # Publication-ready plotting
│   ├── scans/             # Parameter scan plotter
│   └── spurious/          # Spurious eigenvalue plots
└── pbs_experimenting/      # HPC job submission framework
    └── README.md          # Full PBS framework documentation
```

## Core Systems

### 1. DMD Engine (`dmd/`)

Minimal, transparent DMD implementation with delay embedding (Hankel matrices).

**Key features:**
- **Delay embedding**: `DelayEmbedding` class for Hankel transforms
- **DMD variants**: Projected and Exact DMD
- **Reconstruction**: Time-step reconstruction and full data recovery

```python
from dmd.dmd_tools import DelayEmbedding
from dmd.custom_impl_dmd import DMD

# With delay embedding
embedding = DelayEmbedding(num_delays=5)
embedded_data = embedding.transform(data)

dmd = DMD(variant='projected', svd_rank=10)
dmd.fit(embedded_data)

# Reconstruct
reconstruction = dmd.reconstructed_data
original_space = embedding.inverse_transform(reconstruction)
```

### 2. Order Detection Algorithms (`algorithms/`)

Methods for identifying true vs. spurious DMD modes.

#### Information Criteria
- **BIC**: Model selection via Bayesian Information Criterion
- **GAP**: Largest singular value gap

#### Clustering-Based Methods
All use `ModeClustering` for 2-cluster classification:

**Feature extractors:**
- **STC**: Spatiotemporal coupling quotient consistency
- **NestedDMD**: Rank-1 DMD fit to each mode's block structure
- **FixedEigenvalueKVFit**: Kronecker-Vandermonde fit using external eigenvalues
- **ESL-Norm**: Estimated subspace leakage

**Clustering API:**
```python
from algorithms.clustering import ModeClustering
from algorithms.stc import STC

# Extract features
stc = STC(num_delays=5)
features = stc.compute_features(eigenvalues, modes)

# Cluster into true/spurious
clusterer = ModeClustering(algorithm='gmm', strategy='vote')
labels = clusterer.fit(features).labels_  # 1 = true, 0 = spurious

predicted_order = labels.sum()
```

**Combination methods:**
- Merge multiple feature dictionaries before clustering
- Example: `NestedDMD+ESL` combines reconstruction/consistency features with leakage

### 3. Data Generation (`utils/data_generation.py`)

`DMDDataGenerator` creates controlled synthetic DMD data with:
- Configurable eigenvalue magnitudes/frequencies
- Multiple noise models: Gaussian, Student-t, heteroscedastic, bi-Gaussian, AR(1)
- SNR control
- Orthogonal or random mode generation

```python
generator = DMDDataGenerator(
    eigenvalue_magnitude=0.98,
    frequency_separation=0.03,
    snr_db=10.0,
    noise_mode='gaussian'
)
X, X_clean, eigs, modes, amps = generator.generate(50, 200, 3)
```

### 4. PBS Experiment Framework (`pbs_experimenting/`)

YAML-based framework for running large-scale experiments on HPC clusters.

**Features:**
- Pre-generated job configurations (frozen experiments)
- Deterministic random seeds per job
- Method-agnostic design (easy to add/remove methods)
- Upfront config validation
- Incremental result combining

**Workflow:**
```bash
# 1. Configure experiment
cp example_config.yaml my_experiment.yaml

# 2. Generate job configs
python experiment_runner.py make_jobs --config=my_experiment.yaml

# 3. Submit to PBS
python experiment_runner.py submit --config=my_experiment.yaml

# 4. Monitor
python experiment_runner.py status --config=my_experiment.yaml

# 5. Combine results
python experiment_runner.py combine_results --config=my_experiment.yaml
```

See [pbs_experimenting/README.md](pbs_experimenting/README.md) for complete documentation.

### 5. Plotting System (`figures/scans/`)

Config-driven parameter scan plotter (223 lines of code).

**Architecture:**
- `SingleScanPlotter`: Draws one scan on one axis
- `PanelComposer`: Arranges multiple scans into panels
- All styling in YAML (colors, fonts, linestyles)

**Usage:**
```bash
# Single scan
python cli.py single \
  --csv_path=results.csv \
  --x_param=snr_db \
  --metric=order_hit_prob \
  --working_point="{'num_modes': 2, 'noise_mode': 'gaussian'}" \
  --output_path=output.png

# Multi-panel
python cli.py multi \
  --csv_path=results.csv \
  --x_param=snr_db \
  --metric=order_hit_prob \
  --panel_param=noise_mode \
  --panel_values="['gaussian', 'student_t']" \
  --output_path=output.png

# Batch from YAML
python cli.py batch --config_path=batch_config.yaml
```

See [figures/scans/README.md](figures/scans/README.md) for detailed usage.

## Analysis Experiments

### Spurious Eigenvalues vs. Embedding Length (`analysis/spurious_eigenvalues_and_L/`)

Studies how embedding length L affects spurious eigenvalue behavior.

Run experiment:
```bash
python spurious_eigs_L_experiment.py spurious_eigs_L_config.yaml
```

Plot results:
```bash
python spurious_eigs_L_plotter.py spurious_eigs_L_config.yaml
```

## Configuration

All experiments use YAML configuration files.

**Parameter specification:**
```yaml
parameters:
  snr_db:
    type: range          # or 'list', 'const'
    role: wp             # 'wp' (working point) or 'cartesian'
    start: -5
    end: 20
    num_steps: 26
    scale: lin           # or 'log'
  
  noise_mode:
    type: list
    role: cartesian
    values: [gaussian, student_t, hetero]
```

**Parameter roles:**
- `wp` (working point): One-at-a-time variation around working point
- `cartesian`: Full Cartesian product with other cartesian parameters

**Working points:**
- Specify all parameters except the one(s) being scanned
- Required for plotting: determines which slice of data to visualize
- See [figures/scans/WORKING_POINT_EXPLAINED.md](figures/scans/WORKING_POINT_EXPLAINED.md)

## Key Concepts

### Delay Embedding
Transform time series into augmented state vectors via Hankel (delay-coordinate) stacking:
- Input: `(D, N)` → Output: `(D×L, N-L+1)`
- Captures time-delay dynamics in spatial structure
- Essential for distinguishing true vs. spurious modes

### Kronecker-Vandermonde Structure
True DMD modes exhibit block-Vandermonde structure along delay axis:
```
Mode matrix (D × L):
[v₀, λv₀, λ²v₀, ..., λ^(L-1)v₀]
```
where `λ` is the mode's eigenvalue and `v₀` is the spatial pattern.

**Detection strategies:**
- **NestedDMD**: Fit rank-1 DMD to each mode's block structure
- **FixedEigenvalueKVFit**: Use external eigenvalue for closed-form fit
- **STC**: Quotient consistency across delays

### Leakage Norms
Measure how much a mode "leaks" outside the estimated/true subspace:
- **Estimated**: `||mode||² - |eigenvalue|²` (computable from DMD output)
- **Exact**: `||mode - projection_onto_true_subspace||²` (oracle metric)

True modes have small leakage; spurious modes have large leakage.

## Method Reference

| Method | Type | Features | Output |
|--------|------|----------|--------|
| BIC | Information Criteria | Model likelihood + penalty | Single rank estimate |
| GAP | Singular Value | Largest SV gap | Single rank estimate |
| STC | Clustering | Quotient consistency | Per-mode scores |
| NestedDMD | Clustering | Reconstruction error, eigenvalue consistency | Per-mode scores |
| FixedEigenvalueKVFit | Clustering | Kronecker-Vandermonde fit | Per-mode scores |
| ESL-Norm | Clustering | Subspace leakage | Per-mode scores |

**Clustering methods** can be combined:
- `NestedDMD+ESL`: Merge features before clustering
- `FixedEigenvalueKVFit+ESL`: Merge features before clustering

## Development

### Adding a New Method

1. **Create feature extractor** in `algorithms/your_method.py`:
```python
class YourMethod:
    def compute_features(self, modes, eigenvalues, plot=False):
        # Compute per-mode scores
        scores = ...  # shape (M,)
        return {"YourFeatureName": scores}
```

2. **Add to PBS config** (`pbs_experimenting/`):
   - Add method name to `VALID_METHODS` in `config_validator.py`
   - Add to `methods` list in experiment YAML
   - Implement in `MethodEvaluator` class in `run_single_job.py`

3. **Add to plotting config** (`figures/scans/`):
   - Add method color/linestyle in `design_config.yaml`

### Running Tests

Individual experiments can be run manually for debugging:
```bash
# Single PBS job
python run_single_job.py run job_configs/job0042.yaml --plot
```

## Citation

If you use this code, please cite:

```
[Your paper citation here]
```

## License

[Your license here]

## Documentation Index

- **PBS Framework**: [pbs_experimenting/README.md](pbs_experimenting/README.md)
- **Plotting System**: [figures/scans/README.md](figures/scans/README.md)
- **Working Points**: [figures/scans/WORKING_POINT_EXPLAINED.md](figures/scans/WORKING_POINT_EXPLAINED.md)
- **Label System**: [figures/scans/USAGE_GUIDE.md](figures/scans/USAGE_GUIDE.md)
