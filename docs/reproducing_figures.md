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

### Bundled Paper Data

Pre-computed scan results are included under `figures/scans/data/paper/`:

| File | Description |
|------|-------------|
| `paper.csv.gz` | Main paper parameter scans |
| `paper_mechanical_system.csv.gz` | Mechanical-system (real-oscillation) scans |
| `paper_num_modes.csv.gz` | Num-modes-only scan data |

You can regenerate all paper scan figures from these files without re-running PBS jobs.

### Active Batch Configs

Configs in `figures/scans/configs/`:

**Main paper (`paper.csv.gz`), multi-axis scans by spatial dimension:**
- `paper_multi_num_modes_d1.yaml`
- `paper_multi_num_modes_d2.yaml`
- `paper_multi_num_modes_d3.yaml`

**Mechanical system (`paper_mechanical_system.csv.gz`), multi-parameter scans:**
- `paper_multi_xparam_mechanical_system_d1.yaml`
- `paper_multi_xparam_mechanical_system_d2.yaml`
- `paper_multi_xparam_mechanical_system_d3.yaml`

### Basic Usage

**Single Scan:**

```bash
python -m figures.scans.cli single \
  --csv_path=figures/scans/data/paper/paper.csv.gz \
  --x_param=snr_db \
  --metric=pr_auc_mean \
  --working_point="{'num_modes': 3, 'spatial_dim': 2, 'noise_mode': 'gaussian'}" \
  --output_path=output.png \
  --title="SNR Scan"
```

**Multi-Panel (Parameter Variation):**

```bash
python -m figures.scans.cli multi \
  --csv_path=figures/scans/data/paper/paper.csv.gz \
  --x_param=snr_db \
  --metric=pr_auc_mean \
  --working_point="{'num_modes': 3, 'spatial_dim': 2}" \
  --panel_param=noise_mode \
  --panel_values="['gaussian', 'bi_gaussian', 'hetero']" \
  --output_path=output.png \
  --title="SNR Scan by Noise Type"
```

**Batch Processing (Recommended):**

```bash
python -m figures.scans.cli plot_config \
  --config_path=figures/scans/configs/paper_multi_num_modes_d2.yaml
```

### AUC Tables (LaTeX)

Generate normalized PR-AUC summary tables with:

```bash
python figures/scans/make_auc_table.py \
  --csv_path=figures/scans/data/paper/paper.csv.gz \
  --output_dir=figures/scans/outputs/paper
```

This writes `auc_d1.tex`, `auc_d2.tex`, `auc_d3.tex`, and `auc_unified.tex`.

### Configuration

*   **Design Config**: `figures/scans/design_config.yaml` controls colors, fonts, and linestyles.
*   **Batch Config**: YAML files defining which plots to generate.

See `figures/scans/WORKING_POINT_EXPLAINED.md` for details on selecting data slices.

## 2. Spurious Eigenvalue Analysis (`figures/spurious`)

Generates detailed figures showing the distribution of spurious eigenvalues and their metrics ($\rho_{\min}$, $\mu_L$, $\nu_L$).

### Usage

```bash
python -m figures.spurious.spurious_eigs_L_plotter path/to/config.yaml
```

### Experiment Workflow

1.  Run the experiment to generate data:
    ```bash
    python -m analysis.spurious_eigenvalues_and_L.spurious_eigs_L_experiment --config=analysis/spurious_eigenvalues_and_L/spur_eigs_L_config.yaml
    ```
2.  Run the plotter:
    ```bash
    python -m figures.spurious.spurious_eigs_L_plotter figures/spurious/configs/paper_config.yaml
    ```

### Generated Figures

*   `spurious_eigs_L_cdf.pdf`: CDF of eigenvalue magnitudes for different embeddings lengths $L$.
*   `spurious_eigs_rho_min_vs_L.pdf`: Minimum eigenvalue magnitude ($\rho_{\min}$) vs $L$.
*   `spurious_and_noise_cdf_comparison.pdf`: Comparison of mixture vs. pure noise eigenvalue distributions (KS test).
*   `cdf_and_rho_min_combined.pdf`: Combined single-column figure.
*   `mu_nu_vs_L.pdf`: Spectral norms $\mu_L$ and $\nu_L$ vs $L$.

## Reproducing Paper Figures

### From bundled data (fast path)

```bash
# Main paper scan figures (12 PDFs across D=1,2,3)
for d in 1 2 3; do
  python -m figures.scans.cli plot_config \
    --config_path=figures/scans/configs/paper_multi_num_modes_d${d}.yaml
done

# Mechanical-system multi-parameter figures
for d in 1 2 3; do
  python -m figures.scans.cli plot_config \
    --config_path=figures/scans/configs/paper_multi_xparam_mechanical_system_d${d}.yaml
done

# PR-AUC summary tables
python figures/scans/make_auc_table.py \
  --csv_path=figures/scans/data/paper/paper.csv.gz \
  --output_dir=figures/scans/outputs/paper
```

Outputs are written under `figures/scans/outputs/` (gitignored by default).

### From scratch (full regeneration)

1.  **Generate Data**:
    *   Configure and run the large-scale parameter scan using the [PBS Framework](experiment_framework.md).
    *   Copy combined CSVs into `figures/scans/data/paper/`.
    *   Run the spurious eigenvalue experiment for spurious-eigenvalue figures.
2.  **Generate Plots**:
    *   For scans: use the batch configs in `figures/scans/configs/`.
    *   For spurious analysis: use the config in `analysis/spurious_eigenvalues_and_L/`.
