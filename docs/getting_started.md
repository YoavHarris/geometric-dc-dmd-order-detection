# Getting Started

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/dc-dmd-order-detection.git
    cd dc-dmd-order-detection
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Basic Example

Here is a minimal example of generating synthetic data and fitting DMD with delay coordinates.

```python
from dmd.dmd_utils import fit_dmd
from utils.data_generation import DMDDataGenerator

# 1. Generate synthetic data
generator = DMDDataGenerator(
    eigenvalue_magnitude=0.98,
    frequency_separation=0.03,
    snr_db=10.0
)
X, X_clean, eigs_true, modes_true, amps = generator.generate(
    n_spatial=50, n_timesteps=200, n_modes=3
)

# 2. Fit DMD with delay embedding (L=5)
# This mimics applying delay coordinates to handle noise or standing waves
dmd = fit_dmd(data=X, svd_rank=10, num_delays=5)

# 3. Inspect results
print(f"Computed {len(dmd.eigenvalues)} eigenvalues")
print(f"Reconstruction shape: {dmd.reconstructed_data.shape}")
```

## Running an Experiment

To evaluate order detection methods (like STC, BIC, or NestedDMD), use the `algorithms` module:

```python
from algorithms.stc import STC
from algorithms.clustering import ModeClustering

# Extract features
stc = STC(num_delays=5)
features = stc.compute_features(dmd.eigenvalues, dmd.modes)

# Classify modes (True vs Spurious)
clusterer = ModeClustering(algorithm='gmm', strategy='vote')
labels = clusterer(features)

predicted_order = labels.sum()
print(f"Predicted Order: {predicted_order}")
```

For large-scale experiments on clusters, refer to the [Experiment Framework Guide](experiment_framework.md).
