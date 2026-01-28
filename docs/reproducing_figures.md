# Reproducing Figures

This guide explains how to reproduce the figures used in the paper. The repository contains two plotting systems:

1.  **Parameter Scan Plotter** (`figures/scans`): For method comparison results.
2.  **Spurious Eigenvalue Plotter** (`figures/spurious`): For detailed analysis of spurious eigenvalues.

## 1. Parameter Scans (`figures/scans`)

Config-driven system for plotting parameter scans.

### Setup

```bash
pip install -r figures/scans/requirements.txt
```

### Basic Usage

**Single Scan:**

```bash
python figures/scans/cli.py single \
  --csv_path=results.csv \
  --x_param=snr_db \
  --metric=order_hit_prob \
  --working_point="{'num_modes': 2, 'noise_mode': 'gaussian'}" \
  --output_path=output.png \
  --title="SNR Scan"
```

**Multi-Panel (Parameter Variation):**

```bash
python figures/scans/cli.py multi \
  --csv_path=results.csv \
  --x_param=snr_db \
  --metric=order_hit_prob \
  --working_point="{'num_modes': 2}" \
  --panel_param=noise_mode \
  --panel_values="['gaussian', 'student_t', 'hetero', 'bi_gaussian']" \
  --output_path=output.png \
  --title="SNR Scan by Noise Type"
```

**Batch Processing (Recommended):**

```bash
python figures/scans/cli.py batch --config_path=examples/simple_batch.yaml
```

### Configuration

*   **Design Config**: `figures/scans/design_config.yaml` controls colors, fonts, and linestyles.
*   **Batch Config**: YAML files defining which plots to generate.

See `figures/scans/WORKING_POINT_EXPLAINED.md` for details on selecting data slices.

## 2. Spurious Eigenvalue Analysis (`figures/spurious`)

Generates detailed figures showing the distribution of spurious eigenvalues and their metrics ($\rho_{\min}$, $\mu_L$, $\nu_L$).

### Usage

```bash
python figures/spurious/spurious_eigs_L_plotter.py path/to/config.yaml
```

### Experiment Workflow

1.  Run the experiment to generate data:
    ```bash
    python analysis/spurious_eigenvalues_and_L/spurious_eigs_L_experiment.py analysis/spurious_eigenvalues_and_L/spurious_eigs_L_config.yaml
    ```
2.  Run the plotter:
    ```bash
    python figures/spurious/spurious_eigs_L_plotter.py analysis/spurious_eigenvalues_and_L/spurious_eigs_L_config.yaml
    ```

### Generated Figures

*   `spurious_eigs_L_cdf.pdf`: CDF of eigenvalue magnitudes for different embeddings lengths $L$.
*   `spurious_eigs_rho_min_vs_L.pdf`: Minimum eigenvalue magnitude ($\rho_{\min}$) vs $L$.
*   `spurious_and_noise_cdf_comparison.pdf`: Comparison of mixture vs. pure noise eigenvalue distributions (KS test).
*   `cdf_and_rho_min_combined.pdf`: Combined single-column figure.
*   `mu_nu_vs_L.pdf`: Spectral norms $\mu_L$ and $\nu_L$ vs $L$.

## Reproducing Paper Figures

To reproduce the exact figures from the paper:

1.  **Generate Data**:
    *   Run the large-scale parameter scan using the [PBS Framework](experiment_framework.md).
    *   Run the spurious eigenvalue experiment.
2.  **Generate Plots**:
    *   For scans: Use the batch configs in `figures/scans/configs/`.
    *   For spurious analysis: Use the config in `analysis/spurious_eigenvalues_and_L/`.

```bash
# Example: Paper scans
python figures/scans/cli.py batch --config_path=figures/scans/configs/paper_plots.yaml
```
