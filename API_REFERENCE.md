# API Reference

Quick reference for the main classes and functions in the DMD order detection codebase.

## DMD Core (`dmd/dmd_tools.py`)

### `fit_dmd()`
High-level function to fit DMD with optional delay embedding.

```python
fit_dmd(
    data: NDArray,           # (spatial_dim, num_snapshots)
    svd_rank: int,           # DMD rank
    mode: str = "projected", # "projected" or "exact"
    num_delays: int = 1,     # Number of delays (1 = no embedding)
    **kwargs
) -> DMD
```

**Returns:** Fitted `DMD` object

---

### `class DelayEmbedding`
Delay-coordinate (Hankel) embedding transformation.

```python
embedding = DelayEmbedding(num_delays=5)

# Forward transform
embedded = embedding.transform(snapshots)  # (D, N) → (D*L, N-L+1)

# Inverse transform
original = embedding.inverse_transform(embedded, agg='mean')
```

**Attributes:**
- `num_delays`: Number of time delays
- `spatial_dim`: Original spatial dimension (set after first transform)
- `num_snapshots`: Original number of snapshots (set after first transform)

---

### `class DMD`
Dynamic Mode Decomposition with flexible variants.

```python
dmd = DMD(
    variant="projected",              # or "exact"
    svd_rank=None,                    # None = full rank
    scale_mode_by_eigenvalue=False,   # Legacy scaling option
    phase_align=True                  # Make amplitudes real
)

dmd.fit(data)  # (spatial_dim, num_snapshots)
```

**Attributes after fitting:**
- `eigenvalues`: `(M,)` - DMD eigenvalues
- `modes`: `(spatial_dim, M)` - DMD modes
- `amplitudes`: `(M,)` - Mode amplitudes
- `reconstructed_data`: Full reconstruction of training data

**Methods:**
- `reconstruct(timesteps)`: Reconstruct at specific timesteps
- `reconstructed_data()`: Property returning full reconstruction

---

## Data Generation (`utils/data_generation.py`)

### `class DMDDataGenerator`
Generate synthetic DMD data with controlled parameters.

```python
generator = DMDDataGenerator(
    eigenvalue_magnitude=0.98,        # float or list
    frequency_separation=0.03,        # Frequency spacing
    snr_db=10.0,                      # Signal-to-noise ratio
    eigenvalue_magnitude_spread=None, # Heterogeneity in magnitudes
    rho_mode='linspace',              # 'linspace', 'random', or 'logspace'
    noise_mode='gaussian',            # Noise type
    random_seed=None
)

X, X_clean, eigs, modes, amps = generator.generate(
    n_spatial=50,
    n_timesteps=200,
    n_modes=3
)
```

**Noise modes:**
- `'gaussian'`: Standard Gaussian noise
- `'student_t'`: Heavy-tailed Student-t noise (df=2.01)
- `'hetero'`: Heteroscedastic (time-varying variance)
- `'bi_gaussian'`: Bi-modal Gaussian mixture
- `'ar1'`: AR(1) colored noise (phi=0.8)

**Returns:**
- `X`: Noisy data `(n_spatial, n_timesteps)`
- `X_clean`: Clean signal
- `eigs`: True eigenvalues `(n_modes,)`
- `modes`: True modes (scaled by amplitudes) `(n_spatial, n_modes)`
- `amps`: Mode amplitudes `(n_modes,)`

---

## Algorithms

### Information Criteria (`algorithms/bic.py`)

#### `compute_bic_rank()`

```python
from algorithms.bic import compute_bic_rank

best_rank = compute_bic_rank(
    data,           # (spatial_dim, num_snapshots)
    max_rank=20,
    min_rank=1,
    num_delays=5,
    plot=False
)
# Returns: int (best rank)
```

#### `gap_ranks()`

```python
rank = gap_ranks(data)  # Largest singular value gap
```

---

### Clustering (`algorithms/clustering.py`)

#### `class ModeClustering`
Two-cluster mode classification.

```python
clusterer = ModeClustering(
    normalization='min_max',    # 'min_max', 'z_score', or None
    strategy='vote',            # 'vote', 'mean', 'distance', 'single'
    algorithm='kmeans',         # 'kmeans' or 'gmm'
    random_state=None
)

labels = clusterer.fit(features).labels_  # 1 = true, 0 = spurious
# or
labels = clusterer(features)  # Convenience: fit + return labels
```

**Strategies:**
- `'vote'`: Majority vote across features (recommended)
- `'mean'`: Cluster with larger weighted mean
- `'distance'`: Cluster farther from origin
- `'single'`: Use one pilot feature

**Input:** `features` = dict of `{feature_name: scores}` with shape `(M,)` each

**Returns:** Binary labels `(M,)` where 1 = predicted true mode

---

### Spatiotemporal Coupling (`algorithms/stc.py`)

#### `class STC`

```python
stc = STC(num_delays=5, dt=1.0, epsilon=1e-12)

features = stc.compute_features(
    eigenvalues,  # (M,)
    modes,        # (D*L, M)
    plot=False
)
# Returns: {"STC": scores (M,)}
```

**Metric:** Quotient consistency across time delays. Higher = better consistency with eigenvalue.

---

### Block-Vandermonde Fitting (`algorithms/block_vandermonde_fit.py`)

#### `class NestedDMD`

```python
nested = NestedDMD(
    num_delays=5,
    spatial_dim=50
)

features = nested.compute_features(
    modes,        # (D*L, M)
    eigenvalues,  # (M,)
    plot=False
)
# Returns: {"Reconstruction": scores (M,), "Eigenvalue-Consistency": scores (M,)}
```

**Features:**
- **Reconstruction**: How well rank-1 DMD captures each mode
- **Eigenvalue-Consistency**: Agreement between nested and external eigenvalues

---

#### `class FixedEigenvalueBVFit`

```python
febvf = FixedEigenvalueBVFit(
    num_delays=5,
    spatial_dim=50
)

features = febvf.compute_features(
    modes,        # (D*L, M)
    eigenvalues,  # (M,)
    plot=False
)
# Returns: {"BV-Fit": scores (M,)}
```

**Metric:** Closed-form block-Vandermonde fit using external eigenvalues.

---

### Estimated Subspace Leakage (`algorithms/estimated_subspace_leakage.py`)

#### `class EstimatedSubspaceLeakage`

```python
esl = EstimatedSubspaceLeakage(epsilon=1e-12)

features = esl.compute_features(
    exact_modes,  # (D*L, M) - use exact modes, not projected
    eigenvalues,  # (M,)
    plot=False
)
# Returns: {"Estimated-Subspace-Leakage": scores (M,)}
```

**Metric:** `||mode||² - |eigenvalue|²` converted to score via `-log(leakage + ε)`

---

## Analysis Experiments

### Spurious Eigenvalues vs. L (`analysis/spurious_eigenvalues_and_L/`)

#### Experiment Script

```bash
python spurious_eigs_L_experiment.py spurious_eigs_L_config.yaml
```

**Configuration:**
```yaml
experiment:
  num_iter: 100
  random_seed: 42
  num_delays_range: [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]

generator:
  # DMD parameters

output:
  output_path: results/spurious_eigenvalues_L.pkl
```

**Output:** Pickle file with nested dict:
```python
{
    L_value: {
        'true_eigenvalues': array,
        'estimated_eigenvalues': array,
        'iteration_data': list of dicts
    }
}
```

#### Plotting

```bash
python spurious_eigs_L_plotter.py spurious_eigs_L_config.yaml
```

Generates multiple plots:
- Mean distance to nearest true eigenvalue vs. L
- Histogram of spurious eigenvalue locations
- Monte-Carlo mean of minimal eigenvalue magnitude vs. L

---

## Plotting System (`figures/scans/`)

### Command-Line Interface

```bash
# Single scan
python cli.py single \
  --csv_path=data.csv \
  --x_param=snr_db \
  --metric=order_hit_prob \
  --working_point="{'num_modes': 2}" \
  --output_path=output.png

# Multi-panel (varying panel parameter)
python cli.py multi \
  --csv_path=data.csv \
  --x_param=snr_db \
  --metric=order_hit_prob \
  --panel_param=noise_mode \
  --panel_values="['gaussian', 'student_t']" \
  --output_path=output.png

# Multi-panel (varying x parameter)
python cli.py multi_xparam \
  --csv_path=data.csv \
  --x_params="['snr_db', 'freq_sep', 'eig_mag']" \
  --metric=order_hit_prob \
  --working_point="{'num_modes': 3}" \
  --output_path=output.png

# Batch processing
python cli.py batch --config_path=batch_config.yaml
```

### Batch Configuration

```yaml
base:
  csv_path: results.csv
  metric: order_hit_prob
  panel_param: noise_mode
  panel_values: [gaussian, student_t]
  
  working_point:
    temporal_dim: 200
    # ... common parameters

plots:
  - working_point: {num_modes: 2, snr_db: 10}
    x_param: freq_sep
    output_path: output/freq_sep_nm2.png
    title: "Frequency Separation"
    xlim: [0.85, 0.95]  # Optional x-axis limits
```

### Design Configuration (`design_config.yaml`)

```yaml
figure:
  single_column_width_inches: 3.5
  double_column_width_inches: 7.0
  dpi: 300

methods:
  ESL-Norm:
    color: '#1f77b4'
    linestyle: '-'
  NestedDMD:
    color: '#ff7f0e'
    linestyle: '--'

parameter_labels:
  snr_db: "SNR (dB)"
  order_hit_prob: "$P_{\\mathrm{hit}}$"
```

---

## PBS Experiment Framework (`pbs_experimenting/`)

### Configuration Structure

```yaml
experiment:
  base_random_seed: 42

methods:
  - BIC
  - GAP
  - STC
  - NestedDMD
  - FixedEigenvalueBVFit
  - ESL-Norm
  - NestedDMD+ESL      # Combination method

clustering:
  algorithm: gmm         # or 'kmeans'
  decider: mean          # or 'distance', 'vote'
  normalization: min_max # or 'standard', 'null'

generator:
  working_point:
    snr_db: 10.0
    freq_sep: 0.03
    num_modes: 3
    # ... all DMD parameters

parameters:
  snr_db:
    type: range
    role: wp           # One-at-a-time variation
    start: -5
    end: 20
    num_steps: 26
    scale: lin
  
  noise_mode:
    type: list
    role: cartesian    # Full Cartesian product
    values: [gaussian, student_t, hetero]

pbs:
  walltime: "01:00:00"
  memory: "4gb"
  ncpus: 1
```

### Commands

```bash
# Generate job configs
python experiment_runner.py make_jobs --config=my_config.yaml

# Submit all jobs
python experiment_runner.py submit --config=my_config.yaml

# Submit specific jobs
python experiment_runner.py submit --config=my_config.yaml --ids="0-99"

# Check status
python experiment_runner.py status --config=my_config.yaml

# Resubmit failed jobs
python experiment_runner.py resubmit --config=my_config.yaml

# Combine results
python experiment_runner.py combine_results --config=my_config.yaml
python experiment_runner.py combine_results --config=my_config.yaml --incremental
```

### Manual Job Execution

```bash
# Run single job for debugging
python run_single_job.py run job_configs/job0042.yaml --plot
```

---

## Common Workflows

### 1. Generate Synthetic Data and Test Method

```python
from utils.data_generation import DMDDataGenerator
from dmd.dmd_tools import fit_dmd
from algorithms.stc import STC
from algorithms.clustering import ModeClustering

# Generate data
gen = DMDDataGenerator(
    eigenvalue_magnitude=0.98,
    frequency_separation=0.03,
    snr_db=10.0,
    noise_mode='gaussian',
    random_seed=42
)
X, X_clean, eigs_true, modes_true, amps = gen.generate(50, 200, 3)

# Fit DMD
dmd = fit_dmd(X, svd_rank=10, num_delays=5)

# Extract features
stc = STC(num_delays=5)
features = stc.compute_features(dmd.eigenvalues, dmd.modes)

# Cluster
clusterer = ModeClustering(algorithm='gmm', strategy='vote')
labels = clusterer(features)

print(f"Predicted order: {labels.sum()} (true: 3)")
```

### 2. Run Parameter Scan Locally

```python
# Create small test config
config = {
    'experiment': {'base_random_seed': 42},
    'parameters': {
        'snr_db': {'type': 'list', 'role': 'cartesian', 'values': [0, 10, 20]},
        'noise_mode': {'type': 'const', 'value': 'gaussian'}
    },
    # ... minimal config
}

# Run experiment (see pbs_experimenting/run_single_job.py)
```

### 3. Create Publication Figure

```bash
# 1. Run experiment (PBS or local)
python experiment_runner.py make_jobs --config=paper_exp.yaml
python experiment_runner.py submit --config=paper_exp.yaml
python experiment_runner.py combine_results --config=paper_exp.yaml

# 2. Create batch plotting config
# (See figures/scans/configs/ for examples)

# 3. Generate plots
cd figures/scans
python cli.py batch --config_path=configs/paper_plots.yaml
```

---

## Tips

### Random Seeds
- Set `random_seed` in `DMDDataGenerator` for reproducibility
- PBS framework auto-generates seeds: `base_seed + job_id * 1000 + iteration`

### Working Points
- Must specify ALL parameters except scanned ones
- Multi-panel: don't specify `x_param` or `panel_param`
- See `WORKING_POINT_EXPLAINED.md` for detailed examples

### Feature Combination
Merge feature dicts before clustering:
```python
features_combined = {**features_nested, **features_esl}
labels = clusterer(features_combined)
```

### Plotting Best Practices
- Define parameter labels globally in `design_config.yaml`
- Use `xlim` to zoom into interesting regions
- Batch configs allow DRY principle for multiple related plots
