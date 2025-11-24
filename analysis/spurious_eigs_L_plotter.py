"""
Plot spurious eigenvalue magnitude results vs embedding length L.

This script reads the same YAML config used for the spurious-eigs-vs-L
experiment, but plotting is fully driven by the `plotting:` section:

    plotting:
      mplstyle_path: "figures/.../chaos_single.mplstyle"
      data_csv_path: "spurious_eigenvalues_L/spurious_eigs_L_results.csv"
      xlim: [0.85, 1.02]
      colors:
        colormap: viridis
      kde:
        x_eval_points: 500
        bandwidth: scott
      min_stats:
        interval_percentiles: [5, 95]
        show_mean: true

The CSV path is taken ONLY from `plotting.data_csv_path`, so the plotter
does not reconstruct it from the experiment's output config.

Figures are saved in `project_root / output.output_dir`.

Usage (via Fire):

    python spurious_eigs_L_plotter.py path/to/spurious_eigs_L_config.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import fire
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import yaml


# =============================================================================
# Config utilities
# =============================================================================


def load_plot_cfg(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract the plotting section. Fail loudly if missing."""
    try:
        return config["plotting"]
    except KeyError as exc:
        raise KeyError(
            "Config missing required section 'plotting'. "
            "All plotting parameters must be provided explicitly."
        ) from exc


def get_output_dir(config: Mapping[str, Any], project_root: Path) -> Path:
    """
    Locate the directory where figures will be saved.

    Uses config.output.output_dir (same as the experiment),
    but does NOT use output.csv_filename for reading data.
    """
    try:
        out_cfg = config["output"]
        output_dir = out_cfg["output_dir"]
    except KeyError as exc:
        raise KeyError("Config 'output' section must contain key: output_dir") from exc

    out_dir_path = project_root / output_dir
    out_dir_path.mkdir(parents=True, exist_ok=True)
    return out_dir_path


def get_csv_path(plot_cfg: Mapping[str, Any], project_root: Path) -> Path:
    """
    Construct the CSV path from plotting.data_csv_path (relative to project root).

    This decouples the plotter from the experiment's output_dir/csv_filename.
    """
    try:
        rel_csv = plot_cfg["data_csv_path"]
    except KeyError as exc:
        raise KeyError(
            "Config plotting section must contain 'data_csv_path', "
            "relative to the project root."
        ) from exc

    csv_path = project_root / rel_csv
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found at '{csv_path}'. "
            "Check plotting.data_csv_path in your config."
        )
    return csv_path


def apply_style(plot_cfg: Mapping[str, Any], project_root: Path) -> None:
    """Apply the mplstyle specified in plotting.mplstyle_path."""
    try:
        rel_style = plot_cfg["mplstyle_path"]
    except KeyError as exc:
        raise KeyError(
            "Config plotting section must contain 'mplstyle_path', "
            "relative to the project root."
        ) from exc

    style_path = project_root / rel_style
    if not style_path.exists():
        raise FileNotFoundError(
            f"mplstyle file not found at '{style_path}'. "
            "Update plotting.mplstyle_path in your config."
        )

    plt.style.use(str(style_path))


# =============================================================================
# Plotting functions
# =============================================================================


def plot_cdf(df: pd.DataFrame, cfg: Mapping[str, Any], output_path: Path) -> None:
    """
    Plot empirical CDFs of spurious eigenvalue magnitudes for each L.

    Config fields used:
        plotting.xlim = [xmin, xmax]
        plotting.colors.colormap
    """
    L_values = sorted(df["L"].unique())
    xmin, xmax = cfg["xlim"]

    cmap = plt.get_cmap(cfg["colors"]["colormap"])
    colors = cmap(np.linspace(0.0, 1.0, len(L_values)))

    fig, ax = plt.subplots()

    for L, color in zip(L_values, colors):
        vals = df.loc[df["L"] == L, "eigenvalue_magnitude"].to_numpy()

        # Manual ECDF
        xs = np.sort(vals)
        n = len(xs)
        ys = np.arange(1, n + 1) / n

        ax.plot(xs, ys, label=f"L={L}", color=color, alpha=0.8)

    # Reference line at |λ| = 1
    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.7, label="Unit circle")

    ax.set_xlabel("Eigenvalue magnitude |λ|")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("CDF: Spurious eigenvalues → unit circle as L ↑")
    ax.grid(True, alpha=0.3, linestyle=":")

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([0.0, 1.0])

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"CDF saved to: {output_path}")


def plot_pdf(df: pd.DataFrame, cfg: Mapping[str, Any], output_path: Path) -> None:
    """
    Plot smoothed PDFs via Gaussian KDE for each L.

    Config fields used:
        plotting.xlim = [xmin, xmax]
        plotting.kde.x_eval_points
        plotting.kde.bandwidth
        plotting.colors.colormap
    """
    L_values = sorted(df["L"].unique())
    xmin, xmax = cfg["xlim"]

    cmap = plt.get_cmap(cfg["colors"]["colormap"])
    colors = cmap(np.linspace(0.0, 1.0, len(L_values)))

    x_eval_points = int(cfg["kde"]["x_eval_points"])
    bw_method = cfg["kde"]["bandwidth"]
    x_eval = np.linspace(xmin, xmax, x_eval_points)

    fig, ax = plt.subplots()

    for L, color in zip(L_values, colors):
        vals = df.loc[df["L"] == L, "eigenvalue_magnitude"].to_numpy()

        kde = stats.gaussian_kde(vals, bw_method=bw_method)
        pdf = kde(x_eval)

        ax.plot(x_eval, pdf, label=f"L={L}", color=color, alpha=0.8)

    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.7, label="Unit circle")

    ax.set_xlabel("Eigenvalue magnitude |λ|")
    ax.set_ylabel("Probability density")
    ax.set_title("PDF: Spurious eigenvalues → unit circle as L ↑")
    ax.grid(True, alpha=0.3, linestyle=":")

    ax.set_xlim([xmin, xmax])
    ax.set_ylim(bottom=0.0)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"PDF saved to: {output_path}")


def plot_min_stats(df: pd.DataFrame, cfg: Mapping[str, Any], output_path: Path) -> None:
    """
    Plot minimal spurious eigenvalue magnitude vs L, per Monte-Carlo iteration,
    with an empirical percentile band.

    Config fields used:
        plotting.min_stats.interval_percentiles = [low, high]
        plotting.min_stats.show_mean = bool
    """
    if "mc_iter" not in df.columns:
        raise ValueError(
            "Dataframe is missing 'mc_iter'; cannot compute minimal-eigenvalue statistics."
        )

    low_p, high_p = cfg["min_stats"]["interval_percentiles"]
    show_mean = bool(cfg["min_stats"]["show_mean"])

    # Min magnitude per (L, mc_iter)
    mins = df.groupby(["L", "mc_iter"])["eigenvalue_magnitude"].min().reset_index()

    grouped = mins.groupby("L")["eigenvalue_magnitude"]
    stats_df = grouped.agg(
        median="median",
        lower=lambda x: np.percentile(x, low_p),
        upper=lambda x: np.percentile(x, high_p),
        mean="mean",
        count="count",
    ).reset_index()

    fig, ax = plt.subplots()

    ax.plot(
        stats_df["L"],
        stats_df["median"],
        marker="o",
        linestyle="-",
        label="Median min |λ|",
    )

    ax.fill_between(
        stats_df["L"],
        stats_df["lower"],
        stats_df["upper"],
        alpha=0.2,
        label=f"Empirical {low_p}–{high_p} percentile band",
    )

    if show_mean:
        ax.plot(
            stats_df["L"],
            stats_df["mean"],
            linestyle="--",
            linewidth=1.0,
            label="Mean min |λ|",
        )

    ax.set_xlabel("Embedding length L")
    ax.set_ylabel("Minimal spurious eigenvalue magnitude |λ|")
    ax.set_title("Minimal spurious |λ| vs L")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Min eigenvalue stats saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================


def main(config_path: str) -> None:
    """
    CLI entry point.

    Args:
        config_path: Path to YAML config used for the experiment and plotting.
                     Must contain:
                       - plotting.mplstyle_path
                       - plotting.data_csv_path
                       - plotting.xlim
                       - plotting.colors.colormap
                       - plotting.kde.x_eval_points
                       - plotting.kde.bandwidth
                       - plotting.min_stats.interval_percentiles
                       - plotting.min_stats.show_mean
                       - output.output_dir (for figure location)
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    project_root = Path(__file__).parents[1]

    plot_cfg = load_plot_cfg(config)
    apply_style(plot_cfg, project_root)

    csv_path = get_csv_path(plot_cfg, project_root)
    out_dir = get_output_dir(config, project_root)

    print(f"Loading results from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows.")
    print(f"L values: {sorted(df['L'].unique())}")
    if "mc_iter" not in df.columns:
        print(
            "Note: 'mc_iter' column missing; minimal-eigenvalue statistics will be skipped."
        )
    print()

    # Figure paths (filenames are fixed; directory from config.output.output_dir)
    cdf_path = out_dir / "spurious_eigs_L_cdf.png"
    pdf_path = out_dir / "spurious_eigs_L_pdf.png"
    min_stats_path = out_dir / "spurious_eigs_L_min_stats.png"

    # Plots
    plot_cdf(df, plot_cfg, cdf_path)
    plot_pdf(df, plot_cfg, pdf_path)

    if "mc_iter" in df.columns:
        plot_min_stats(df, plot_cfg, min_stats_path)

    print("Done.")


if __name__ == "__main__":
    fire.Fire(main)
