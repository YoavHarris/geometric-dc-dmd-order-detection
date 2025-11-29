"""
Plot spurious eigenvalue magnitude results vs embedding length L.

This script reads the YAML config and plotting parameters to generate
CDFs, PDFs, and minimal eigenvalue statistics.

Usage (via Fire):
    python spurious_eigs_L_plotter.py path/to/config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import fire
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Add figures/ directory to sys.path to import common
# Assumes this script is in figures/spurious/
sys.path.append(str(Path(__file__).parents[1]))
from common import plotting_common


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
        xs = np.sort(vals)
        n = xs.size
        ys = np.arange(1, n + 1) / n

        # Extend to full axis range: start flat at 0, end flat at 1
        xs_plot = np.concatenate(([xmin], xs, [xmax]))
        ys_plot = np.concatenate(([0.0], ys, [1.0]))

        ax.step(xs_plot, ys_plot, where="post", label=f"L={L}", color=color, alpha=0.8)

    # Reference line at |λ| = 1
    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.7, label="Unit circle")

    ax.set_xlabel("Eigenvalue magnitude |λ|")
    ax.set_ylabel("Cumulative probability")

    ax.grid(True, alpha=0.3, linestyle=":")

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([0.0, 1.2])

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
        label="Median($r_{min}$)",
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
            label="Mean($r_{min}$)",
        )

    ax.set_xlabel("Embedding length L")
    ax.set_ylabel("Spurious Eigenvalue Magnitude $r$")
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
    """
    config = plotting_common.load_yaml_config(config_path)
    project_root = plotting_common.resolve_project_root(config, config_path)

    # 1. Get plotting config
    try:
        plot_cfg = config["plotting"]
    except KeyError as exc:
        raise KeyError("Config missing required section 'plotting'.") from exc

    # 2. Apply style
    plotting_common.apply_style(plot_cfg, project_root)

    # 3. Resolve CSV path
    try:
        rel_csv = plot_cfg["data"]["csv_path"]
    except KeyError:
        # Fallback to old key if 'data.csv_path' structure not used, or error
        # The plan says: plotting.data.csv_path
        # But let's support the old key 'data_csv_path' if we want to be nice,
        # but the plan specifies a new schema. I will stick to the plan schema.
        raise KeyError("Config must contain plotting.data.csv_path")

    csv_path = plotting_common.resolve_path(project_root, rel_csv)

    # 4. Resolve output directory
    try:
        rel_out_dir = config["output"]["output_dir"]
    except KeyError:
        raise KeyError("Config must contain output.output_dir")

    out_dir = plotting_common.resolve_path(project_root, rel_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows.")
    print(f"L values: {sorted(df['L'].unique())}")

    if "mc_iter" not in df.columns:
        print(
            "Note: 'mc_iter' column missing; minimal-eigenvalue statistics will be skipped."
        )
    print()

    # 5. Figure filenames from config
    out_cfg = config["output"]
    cdf_filename = out_cfg.get("cdf_filename", "spurious_eigs_L_cdf.png")
    pdf_filename = out_cfg.get("pdf_filename", "spurious_eigs_L_pdf.png")
    min_stats_filename = out_cfg.get(
        "min_stats_filename", "spurious_eigs_L_min_stats.png"
    )

    cdf_path = out_dir / cdf_filename
    pdf_path = out_dir / pdf_filename
    min_stats_path = out_dir / min_stats_filename

    # 6. Generate plots
    plot_cdf(df, plot_cfg, cdf_path)
    plot_pdf(df, plot_cfg, pdf_path)

    if "mc_iter" in df.columns:
        plot_min_stats(df, plot_cfg, min_stats_path)

    print("Done.")


if __name__ == "__main__":
    fire.Fire(main)
