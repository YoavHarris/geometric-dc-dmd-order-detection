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
                "font.size": 6,
                "axes.labelsize": 6,
                "axes.titlesize": 6,
                "xtick.labelsize": 6,
                "ytick.labelsize": 6,
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
        raise ValueError(
            f"seln_version must be 'clean' or 'perturbed', got {seln_version}"
        )

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
        title="(a) RSLN (Oracle)",
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
        ax.set_ylabel("Cumulative Probability")
    else:
        # Remove y-label and y-tick labels for right panel
        ax.set_ylabel("")
        ax.set_yticklabels([])

    # Use log scale for x-axis
    ax.set_xscale("log")
    ax.set_ylim([-0.05, 1.05])


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
        raise ValueError(
            f"seln_version must be 'clean' or 'perturbed', got {seln_version}"
        )

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
        title="(a) RSLN (Oracle)",
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
    # Only set ylabel for left panel (SELN) since y-axis is shared
    if "SELN" in title:
        ax.set_ylabel("Relative Norm (log-scale)")
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["True", "Spurious"])
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linestyle=":", axis="y", linewidth=0.5)

    # Set x-axis limits to give some space around the points
    ax.set_xlim([-0.3, 1.3])


def plot_multi_scenario_boxplot(
    csv_paths: list[str],
    panel_labels: list[str],
    output_path: str,
    seln_version: str = "perturbed",
    annotate_gap_panel: int = 0,
    style_file: str = "double",
):
    """
    Create multi-panel box plot comparing SELN and RELN distributions across scenarios.

    Each panel shows 4 box plots:
    - True SELN (blue, no hatch)
    - Spurious SELN (red, no hatch)
    - True RELN (blue, diagonal hatch)
    - Spurious RELN (red, diagonal hatch)

    One panel can optionally show gap annotation with tau_spur, tau_true, and gamma.

    Args:
        csv_paths: List of paths to CSV files with leakage measurements
        panel_labels: List of labels for each panel (same length as csv_paths)
        output_path: Path to save the figure
        seln_version: Version of SELN to plot, either "clean" or "perturbed"
        annotate_gap_panel: Which panel (0-indexed) to annotate with gap markers (default: 0)
        style_file: 'single', 'double', or path to mplstyle file (default: 'double')
    """
    if len(csv_paths) != len(panel_labels):
        raise ValueError("csv_paths and panel_labels must have the same length")

    if seln_version not in ["clean", "perturbed"]:
        raise ValueError(
            f"seln_version must be 'clean' or 'perturbed', got {seln_version}"
        )

    # Load plot style
    project_root = Path(__file__).parents[2]

    if style_file == "double":
        style_path = (
            project_root / "figures" / "mplstyle_files" / "chaos_double.mplstyle"
        )
        total_width = 7.0
    elif style_file == "single":
        style_path = (
            project_root / "figures" / "mplstyle_files" / "chaos_single.mplstyle"
        )
        total_width = 3.375
    else:
        style_path = Path(style_file)
        # Default to double width if custom style is provided, user can override via mplstyle
        total_width = 7.0
    if style_path.exists():
        plt.style.use(str(style_path))
    else:
        print(f"Warning: Style file {style_path} not found.")

    n_panels = len(csv_paths)
    if style_file == "single":
        panel_height = 2.0
    else:
        panel_height = 3.0  # Match parameter_scans_plotting height

    # Create horizontal panels
    # Use squeeze=False to always get an array of axes
    # Don't use sharey=True to match parameter_scans_plotting behavior exactly
    fig, axes = plt.subplots(
        1, n_panels, figsize=(total_width, panel_height), squeeze=False
    )
    axes = axes[0]  # Flatten the 2D array (1, n) to 1D

    # Plot each scenario
    for i, (csv_path, label) in enumerate(zip(csv_paths, panel_labels)):
        df = pd.read_csv(csv_path)
        show_gap_annotation = i == annotate_gap_panel

        # Only show ylabel on the first panel
        show_ylabel = i == 0

        _plot_boxplot_panel(
            axes[i],
            df,
            title=label,
            show_ylabel=show_ylabel,
            seln_version=seln_version,
            show_gap_annotation=show_gap_annotation,
        )

        # If not the first panel, remove y-tick labels manually since we aren't using sharey=True
        if i > 0:
            axes[i].set_yticklabels([])

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Multi-scenario box plot saved to: {output_path}")


def _plot_boxplot_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    show_ylabel: bool,
    seln_version: str,
    show_gap_annotation: bool,
):
    """
    Plot a single panel with box plots for SELN and RELN.

    Layout:
    - Position 0: True SELN
    - Position 1: Spurious SELN
    - Gap at position 2
    - Position 3: True RELN
    - Position 4: Spurious RELN
    """
    # Split data
    true_df, spurious_df = _split_data(df)

    seln_col = f"seln_{seln_version}"

    # Extract values
    data_dict = {
        "true_seln": true_df[seln_col].values,
        "spurious_seln": spurious_df[seln_col].values,
        "true_reln": true_df["reln"].values,
        "spurious_reln": spurious_df["reln"].values,
    }

    # Box plot positions with gap
    positions = [0, 1, 3, 4]
    data_list = [
        data_dict["true_seln"],
        data_dict["spurious_seln"],
        data_dict["true_reln"],
        data_dict["spurious_reln"],
    ]

    # Create box plots
    bp = ax.boxplot(
        data_list,
        positions=positions,
        widths=0.45,  # Narrower boxes to give room for annotations
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", markersize=2, alpha=0.3),
    )

    # Colors: blue for true, red for spurious
    colors = ["#1f77b4", "#d62728", "#1f77b4", "#d62728"]

    # Color boxes
    for i, (patch, color) in enumerate(zip(bp["boxes"], colors)):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
        patch.set_alpha(0.7)

    # Color other box plot elements
    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_color("black")
            item.set_linewidth(0.8)

    # Set x-axis with individual labels for each box (rotated diagonally)
    ax.set_xticks([0, 1, 3, 4])
    labels = [
        "True-RSLN",
        "Spur-RSLN",
        "True-RELN",
        "Spur-RELN",
    ]
    # Remove fontsize=5 to let mplstyle control it (likely 7pt or 9pt)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    # Set consistent x-limits for all panels including reserved space for arrow at position 5
    ax.set_xlim([-0.5, 5.5])

    # Set y-axis
    ax.set_yscale("log")
    if show_ylabel:
        ax.set_ylabel("Relative Leakage Norm")

    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle=":", axis="y", linewidth=0.5)

    # Add gap annotation if requested
    if show_gap_annotation:
        # Check if trial_id exists for per-iteration stats
        if "trial_id" in df.columns:
            # Compute per-trial stats
            # tau_spur: minimum spurious SELN per trial
            tau_spur_per_trial = df[df["is_true"] == 0].groupby("trial_id")[seln_col].min()
            # tau_true: maximum true SELN per trial
            tau_true_per_trial = df[df["is_true"] == 1].groupby("trial_id")[seln_col].max()
            
            # Compute statistics
            tau_spur_median = tau_spur_per_trial.median()
            tau_spur_low = np.percentile(tau_spur_per_trial, 5)
            tau_spur_high = np.percentile(tau_spur_per_trial, 95)
            
            tau_true_median = tau_true_per_trial.median()
            tau_true_low = np.percentile(tau_true_per_trial, 5)
            tau_true_high = np.percentile(tau_true_per_trial, 95)
            
            # Plot shaded regions (CI)
            ax.axhspan(tau_spur_low, tau_spur_high, color='red', alpha=0.15, zorder=0, label="5-95% CI")
            ax.axhspan(tau_true_low, tau_true_high, color='blue', alpha=0.15, zorder=0)
            
            # Plot median lines
            ax.axhline(tau_spur_median, color='red', linestyle='--', linewidth=1.0, alpha=0.7, zorder=10)
            ax.axhline(tau_true_median, color='blue', linestyle='--', linewidth=1.0, alpha=0.7, zorder=10)
            
            # Use medians for labels and arrows
            tau_spur_val = tau_spur_median
            tau_true_val = tau_true_median
            
        else:
            # Fallback to global min/max if trial_id is missing
            tau_spur_val = np.min(data_dict["spurious_seln"])
            tau_true_val = np.max(data_dict["true_seln"])
            
            ax.axhline(tau_spur_val, color='red', linestyle='--', linewidth=1.0, alpha=0.7, zorder=10)
            ax.axhline(tau_true_val, color='blue', linestyle='--', linewidth=1.0, alpha=0.7, zorder=10)

        # Add labels for the horizontal lines at the right edge
        # Remove manual fontsize
        ax.text(5.2, tau_spur_val, r"$\tau_{\mathrm{spur}}$", va="top", ha="left")
        ax.text(5.2, tau_true_val, r"$\tau_{\mathrm{true}}$", va="top", ha="left")

        # Add bidirectional arrow at reserved position 5
        from matplotlib.patches import FancyArrowPatch

        arrow = FancyArrowPatch(
            (5.0, tau_spur_val),
            (5.0, tau_true_val),
            arrowstyle="<->",
            mutation_scale=15,  # Increased scale for larger arrow
            linewidth=1.2,
            color="black",
            zorder=10,
        )
        ax.add_patch(arrow)

        # Add gamma label next to arrow
        gamma_y = np.sqrt(tau_spur_val * tau_true_val)  # Geometric mean for log scale
        # Remove manual fontsize, keep bold
        ax.text(5.15, gamma_y, r"$\gamma$", va="center", ha="left", weight="bold")

    # Don't create per-panel legend - explicit tick labels are sufficient


def main(
    csv_paths: str | list[str],
    panel_labels: str | list[str] | None = None,
    output_dir: str = "results",
    seln_version: str = "perturbed",
    style_file: str = "double",
    annotate_gap_panel: int = 0,
    output_filename: str | None = None,
):
    """
    Plot leakage separation visualizations from experiment results.

    Args:
        csv_paths: Path or list of paths to CSV files with leakage measurements
        panel_labels: Label or list of labels for each panel (optional)
        output_dir: Directory where output plots will be saved
        seln_version: Version of SELN to plot, either "clean" or "perturbed" (default: "perturbed")
        style_file: 'single', 'double', or path to mplstyle file (default: 'double')
        annotate_gap_panel: Index of the panel to annotate with gap markers (default: 0)
        output_filename: Specific filename for the output plot (optional)
    """
    # Handle inputs
    if isinstance(csv_paths, str):
        # Fire might pass a string like "[path1,path2]" or just "path1"
        if csv_paths.startswith("[") and csv_paths.endswith("]"):
            # Simple parsing for list string
            csv_paths = [p.strip() for p in csv_paths[1:-1].split(",")]
        else:
            csv_paths = [csv_paths]

    if panel_labels is None:
        panel_labels = [f"Scenario {i+1}" for i in range(len(csv_paths))]
    elif isinstance(panel_labels, str):
        if panel_labels.startswith("[") and panel_labels.endswith("]"):
            panel_labels = [l.strip() for l in panel_labels[1:-1].split(",")]
        else:
            panel_labels = [panel_labels]

    if len(csv_paths) != len(panel_labels):
        # If single label provided for multiple CSVs, repeat it? No, better to error or auto-generate
        print(f"Warning: Number of labels ({len(panel_labels)}) does not match number of CSVs ({len(csv_paths)}). Using defaults.")
        panel_labels = [f"Scenario {i+1}" for i in range(len(csv_paths))]

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define output file paths
    if output_filename:
        cdf_output = output_path / output_filename
    elif style_file == "double":
        cdf_output = output_path / "seln_reln_scatters.png"
    else:
        cdf_output = (
            output_path / f"leakage_separation_cdfs_{seln_version}_{style_file}.png"
        )

    print(f"\nGenerating box plot (SELN version: {seln_version})...")
    print(f"CSVs: {csv_paths}")
    print(f"Labels: {panel_labels}")

    plot_multi_scenario_boxplot(
        csv_paths=csv_paths,
        panel_labels=panel_labels,
        output_path=str(cdf_output),
        seln_version=seln_version,
        annotate_gap_panel=annotate_gap_panel,
        style_file=style_file,
    )

    print()
    print("Done!")


if __name__ == "__main__":
    fire.Fire(main)
