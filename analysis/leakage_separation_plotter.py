"""
Plot leakage separation CDFs: SELN and RELN for true vs spurious components.

This script reads the CSV output from leakage_separation_experiment.py and creates
a two-panel figure showing empirical CDFs of SELN and RELN for true and spurious
components, demonstrating the theoretical separation.

Usage:
    python leakage_separation_plotter.py <csv_path> [--output OUTPUT_PATH]
"""

from __future__ import annotations

import fire
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_two_panel_cdfs(
    df: pd.DataFrame,
    output_path: str,
):
    """
    Create two-panel CDF plot: SELN (left) and RELN (right).

    Designed for AIP single-column width (3.375 inches).
    """
    # AIP single column width is 3.375 inches
    # We'll use a slightly taller aspect ratio to fit two panels vertically or
    # side-by-side with small fonts.
    # Actually, for single column, side-by-side might be too cramped.
    # But the user asked for a two-panel plot for a single column.
    # Let's try side-by-side with carefully chosen fonts.

    width_in = 3.375
    height_in = 1.8  # Aspect ratio ~ 2:1

    # Set style parameters using the project's mplstyle file
    project_root = Path(__file__).parents[1]
    style_path = project_root / "figures" / "mplstyle_files" / "chaos_single.mplstyle"

    if style_path.exists():
        plt.style.use(str(style_path))
    else:
        print(f"Warning: Style file not found at {style_path}, using defaults.")
        # Fallback to some reasonable defaults if style file is missing
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 8,
                "axes.labelsize": 8,
                "axes.titlesize": 8,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "legend.fontsize": 6,
                "lines.linewidth": 1.0,
            }
        )

    # Split data by true/spurious
    true_df = df[df["is_true"] == 1]
    spurious_df = df[df["is_true"] == 0]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)

    # --- Left panel: SELN (oracle) ---
    _plot_single_cdf(
        ax1,
        true_data=true_df["seln"].values,
        spurious_data=spurious_df["seln"].values,
        metric_name=r"$r_{\mathcal{S}}(\phi)$",  # Mathtext equivalent of \RSLN
        title="(a) SELN (Oracle)",
    )

    # --- Right panel: RELN (practical) ---
    _plot_single_cdf(
        ax2,
        true_data=true_df["reln"].values,
        spurious_data=spurious_df["reln"].values,
        metric_name=r"$r_{U_M}(\phi)$",  # Mathtext equivalent of \RELN
        title="(b) RELN (Practical)",
    )

    # Create a shared legend at the top
    # We create dummy lines to generate the legend handles
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="#1f77b4", lw=1.0, label="True"),
        Line2D([0], [0], color="#d62728", lw=1.0, label="Spurious"),
    ]

    fig.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        ncol=1,
        frameon=False,
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {output_path}")


def _plot_single_cdf(
    ax: plt.Axes,
    true_data: np.ndarray,
    spurious_data: np.ndarray,
    metric_name: str,
    title: str,
):
    """
    Plot a single CDF panel with true and spurious distributions.
    """
    # Compute empirical CDFs
    true_sorted = np.sort(true_data)
    spurious_sorted = np.sort(spurious_data)

    true_cdf = np.arange(1, len(true_sorted) + 1) / len(true_sorted)
    spurious_cdf = np.arange(1, len(spurious_sorted) + 1) / len(spurious_sorted)

    # Append point at x=1.1 to show plateau
    true_sorted = np.append(true_sorted, 1.1)
    true_cdf = np.append(true_cdf, 1.0)
    spurious_sorted = np.append(spurious_sorted, 1.1)
    spurious_cdf = np.append(spurious_cdf, 1.0)

    # Prepend point at small x to show start from zero
    min_val = min(np.min(true_sorted), np.min(spurious_sorted))
    start_val = min_val / 10.0

    true_sorted = np.insert(true_sorted, 0, start_val)
    true_cdf = np.insert(true_cdf, 0, 0.0)
    spurious_sorted = np.insert(spurious_sorted, 0, start_val)
    spurious_cdf = np.insert(spurious_cdf, 0, 0.0)

    # Plot CDFs
    ax.plot(
        true_sorted,
        true_cdf,
        label="True",
        color="#1f77b4",  # Standard blue
        alpha=0.9,
        drawstyle="steps-post",
    )
    ax.plot(
        spurious_sorted,
        spurious_cdf,
        label="Spurious",
        color="#d62728",  # Standard red
        alpha=0.9,
        drawstyle="steps-post",
    )

    # Formatting
    ax.set_xlabel(metric_name)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle=":", which="both", linewidth=0.5)

    # Only show y-label and ticks for the left panel (SELN)
    if "SELN" in title:
        ax.set_ylabel("CDF")
    else:
        # Remove y-label and y-tick labels for right panel
        ax.set_ylabel("")
        ax.set_yticklabels([])

    # Use log scale for x-axis
    ax.set_xscale("log")
    ax.set_ylim([-0.05, 1.05])

    # Set x-ticks to be cleaner (powers of 10)
    # ax.xaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=5))


def main(csv_path: str, output: str = "results/leakage_separation_cdfs.png"):
    """
    Plot leakage separation CDFs from experiment results.
    """
    # Load data
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} component measurements")

    # Check required columns
    required_cols = ["is_true", "seln", "reln"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Create output directory
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate plot
    plot_two_panel_cdfs(df, str(output_path))

    print()
    print("Done!")


if __name__ == "__main__":
    fire.Fire(main)
