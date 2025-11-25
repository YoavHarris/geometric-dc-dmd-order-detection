"""
Plot leakage separation visualizations: SELN and RELN for true vs spurious components.

This script reads the CSV output from leakage_separation_experiment.py and creates
two types of visualizations:
1. Two-panel CDF figure showing empirical CDFs of SELN and RELN
2. Two-panel scatter plot showing the distribution of SELN and RELN values

Usage:
    python leakage_separation_plotter.py <csv_path> [--output_dir OUTPUT_DIR] [--seln_version {clean,perturbed}]
"""

from __future__ import annotations

import fire
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _load_plot_style():
    """Load matplotlib style from project's mplstyle file."""
    project_root = Path(__file__).parents[2]
    style_path = project_root / "figures" / "mplstyle_files" / "chaos_single.mplstyle"

    if style_path.exists():
        plt.style.use(str(style_path))
    else:
        print(f"Warning: Style file not found at {style_path}, using defaults.")
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


def _split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe by true/spurious components."""
    true_df = df[df["is_true"] == 1]
    spurious_df = df[df["is_true"] == 0]
    return true_df, spurious_df


def plot_two_panel_cdfs(
    df: pd.DataFrame,
    output_path: str,
    seln_version: str = "perturbed",
):
    """
    Create two-panel CDF plot: SELN (left) and RELN (right).

    Designed for AIP single-column width (3.375 inches).
    
    Args:
        df: DataFrame with leakage measurements
        output_path: Path to save the figure
        seln_version: Version of SELN to plot, either "clean" or "perturbed"
    """
    # Validate seln_version
    if seln_version not in ["clean", "perturbed"]:
        raise ValueError(f"seln_version must be 'clean' or 'perturbed', got {seln_version}")
    
    seln_col = f"seln_{seln_version}"
    
    # AIP single column width is 3.375 inches
    # We'll use a slightly taller aspect ratio to fit two panels vertically or
    # side-by-side with small fonts.
    # Actually, for single column, side-by-side might be too cramped.
    # But the user asked for a two-panel plot for a single column.
    # Let's try side-by-side with carefully chosen fonts.

    _load_plot_style()

    width_in = 3.375
    height_in = 1.8  # Aspect ratio ~ 2:1

    # Split data by true/spurious
    true_df, spurious_df = _split_data(df)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)

    # --- Left panel: SELN (oracle) ---
    _plot_single_cdf(
        ax1,
        true_data=true_df[seln_col].values,
        spurious_data=spurious_df[seln_col].values,
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


def plot_two_panel_scatter(
    df: pd.DataFrame,
    output_path: str,
    seln_version: str = "perturbed",
):
    """
    Create two-panel scatter plot: SELN (left) and RELN (right).

    Shows distribution of leakage values for true (x=0) and spurious (x=1) components
    with logarithmic y-axis to handle the wide range of values.
    
    Args:
        df: DataFrame with leakage measurements
        output_path: Path to save the figure
        seln_version: Version of SELN to plot, either "clean" or "perturbed"
    """
    # Validate seln_version
    if seln_version not in ["clean", "perturbed"]:
        raise ValueError(f"seln_version must be 'clean' or 'perturbed', got {seln_version}")
    
    seln_col = f"seln_{seln_version}"
    
    _load_plot_style()

    # Split data by true/spurious
    true_df, spurious_df = _split_data(df)

    # Extract leakage values
    seln_true = true_df[seln_col].values
    seln_spurious = spurious_df[seln_col].values
    reln_true = true_df["reln"].values
    reln_spurious = spurious_df["reln"].values

    # Create figure with two subplots - wider for better visibility
    width_in = 6.75  # AIP double column width
    height_in = 3.0
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(width_in, height_in), constrained_layout=True, sharey=True
    )

    # --- Left panel: SELN (oracle) ---
    _plot_single_scatter(
        ax1,
        true_values=seln_true,
        spurious_values=seln_spurious,
        metric_name=r"$r_{\mathcal{S}}(\phi)$",
        title="(a) SELN (Oracle)",
    )

    # --- Right panel: RELN (practical) ---
    _plot_single_scatter(
        ax2,
        true_values=reln_true,
        spurious_values=reln_spurious,
        metric_name=r"$r_{U_M}(\phi)$",
        title="(b) RELN (Practical)",
    )

    # Add legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#1f77b4",
            markersize=6,
            alpha=0.6,
            label="True",
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#d62728",
            markersize=6,
            alpha=0.6,
            label="Spurious",
            linestyle="None",
        ),
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
    print(f"Scatter plot saved to: {output_path}")


def _plot_single_scatter(
    ax: plt.Axes,
    true_values: np.ndarray,
    spurious_values: np.ndarray,
    metric_name: str,
    title: str,
):
    """
    Plot a single scatter panel with true and spurious distributions.

    True components are placed at x=0, spurious at x=1.
    Y-axis uses logarithmic scale to handle wide range of leakage values.
    """
    # Create x positions: 0 for true, 1 for spurious
    x_true = np.zeros(len(true_values))
    x_spurious = np.ones(len(spurious_values))

    # Plot with transparency to show density
    ax.scatter(
        x_true,
        true_values,
        color="#1f77b4",
        alpha=0.6,
        s=10,
        label="True",
        edgecolors="none",
    )
    ax.scatter(
        x_spurious,
        spurious_values,
        color="#d62728",
        alpha=0.6,
        s=10,
        label="Spurious",
        edgecolors="none",
    )

    # Formatting
    ax.set_ylabel(f"Leakage (log scale)")
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["True", "Spurious"])
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linestyle=":", axis="y", linewidth=0.5)

    # Set x-axis limits to give some space around the points
    ax.set_xlim([-0.3, 1.3])


def main(
    csv_path: str,
    output_dir: str = "results",
    seln_version: str = "perturbed",
):
    """
    Plot leakage separation visualizations from experiment results.

    Args:
        csv_path: Path to CSV file with leakage measurements
        output_dir: Directory where output plots will be saved
        seln_version: Version of SELN to plot, either "clean" or "perturbed" (default: "perturbed")
    """
    # Load data
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} component measurements")

    # Check required columns
    seln_col = f"seln_{seln_version}"
    required_cols = ["is_true", seln_col, "reln"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define output file paths
    cdf_output = output_path / f"leakage_separation_cdfs_{seln_version}.png"
    scatter_output = output_path / f"leakage_separation_scatter_{seln_version}.png"

    # Generate CDF plot
    print(f"\nGenerating CDF plot (SELN version: {seln_version})...")
    plot_two_panel_cdfs(df, str(cdf_output), seln_version)

    # Generate scatter plot
    print(f"\nGenerating scatter plot (SELN version: {seln_version})...")
    plot_two_panel_scatter(df, str(scatter_output), seln_version)

    print()
    print("Done!")


if __name__ == "__main__":
    fire.Fire(main)
