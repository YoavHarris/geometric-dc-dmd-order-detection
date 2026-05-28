# DC-DMD Order Detection

Research codebase for **Dynamic Mode Decomposition (DMD) order detection** using delay-coordinate embeddings and mode classification algorithms.

**Public repository:** [YoavHarris/geometric-dc-dmd-order-detection](https://github.com/YoavHarris/geometric-dc-dmd-order-detection)  
**Current release:** v1.1.0

## Overview

This repository implements methods for automatically identifying the correct number of modes in DMD analysis, particularly in the presence of noise. It includes a complete delay-coordinate DMD implementation, order detection algorithms (NestedDMD, STC, BIC, ResDMD, ESR-Energy, and others), bundled paper scan data, and a PBS framework for large-scale experiments.

## Documentation

Full documentation is available in the `docs/` directory:

*   **[Getting Started](docs/getting_started.md)**: Installation and basic usage.
*   **[Experiment Framework](docs/experiment_framework.md)**: Guide to running large-scale experiments on clusters.
*   **[Reproducing Figures](docs/reproducing_figures.md)**: How to generate the figures from the paper.
*   **[API Reference](docs/api_reference.md)**: Detailed class and function reference.
*   **[Developer Guide](docs/developer_guide.md)**: Contributing and adding new methods.

## Quick Start

1.  **Install**:
    ```bash
    git clone https://github.com/YoavHarris/geometric-dc-dmd-order-detection.git
    cd geometric-dc-dmd-order-detection
    pip install -r requirements.txt
    ```

2.  **Run a basic example**:
    ```python
    from dmd.dmd_utils import fit_dmd
    from utils.data_generation import DMDDataGenerator

    # Generate synthetic data
    X, _, _, _, _ = DMDDataGenerator(
        eigenvalue_magnitude=0.98,
        frequency_separation=0.03,
        snr_db=10.0,
        noise_mode="gaussian",
        random_seed=0,
    ).generate(50, 200, 3)

    # Fit DMD with delay embedding
    dmd = fit_dmd(X, svd_rank=10, num_delays=5)
    print(f"Found {len(dmd.eigs)} eigenvalues")
    ```

## Running CLIs (recommended)

All repository CLIs can be run from the repo root using module execution:

```bash
python -m pbs_experimenting.experiment_runner --help
python -m figures.scans.cli --help
python figures/scans/make_auc_table.py --help
```

## Citation

If you use this code, please cite it using the metadata in [`CITATION.cff`](CITATION.cff).

## License

This project is licensed under the terms of the [MIT License](LICENSE).
