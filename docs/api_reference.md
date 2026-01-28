# API Reference

Quick reference for the main classes and functions in the DMD order detection codebase.

## DMD Utilities (`dmd/dmd_utils.py`)

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

## DMD Implementation (`dmd/custom_impl_dmd.py`)

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

## Delay Embedding (`utils/delay_embedding.py`)

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
- `_original_shape`: Shape of original data (set after transform)

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

### Kronecker-Vandermonde Fitting (`algorithms/kronecker_vandermonde_fit.py`)

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

#### `class FixedEigenvalueKVFit`

```python
fekvf = FixedEigenvalueKVFit(
    num_delays=5,
    spatial_dim=50
)

features = fekvf.compute_features(
    modes,        # (D*L, M)
    eigenvalues,  # (M,)
    plot=False
)
# Returns: {"KV-Fit": scores (M,)}
```

**Metric:** Closed-form kronecker-Vandermonde fit using external eigenvalues.

---

### Estimated Subspace Residual (`algorithms/estimated_subspace_residual.py`)

#### `class EstimatedSubspaceResidual`

```python
esr = EstimatedSubspaceResidual(epsilon=1e-12)

features = esr.compute_features(
    exact_modes,  # (D*L, M) - use exact modes, not projected
    eigenvalues,  # (M,)
    plot=False
)
# Returns: {"Estimated-Subspace-Residual": scores (M,)}
```

**Metric:** `||mode||² - |eigenvalue|²` converted to score via `-log(residual + ε)`
