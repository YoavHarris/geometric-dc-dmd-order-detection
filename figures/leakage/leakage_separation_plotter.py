"""
Plot leakage separation visualizations: RSLN and RELN for true vs spurious components.

This script reads the YAML config and plotting parameters to generate
multi-panel box plots or scatter plots.

Usage (via Fire):
    python leakage_separation_plotter.py path/to/config.yaml
"""

from __future__ import annotations

from pathlib import Path
import fire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from figures.common import plotting_common


def _split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe by true/spurious components."""
    true_df = df[df["is_true"] == 1]
    spurious_df = df[df["is_true"] == 0]
    return true_df, spurious_df


def plot_multi_scenario_boxplot(
    csv_paths: list[Path],
    panel_labels: list[str],
    output_path: Path,
    rsln_version: str,
    annotate_gap_panel: int | None,
    project_root: Path,
    show_titles: bool = True,
    panel_height: float | None = None,
):
    """
    Create multi-panel box plot comparing RSLN and RELN distributions across scenarios.
    """
    if len(csv_paths) != len(panel_labels):
        raise ValueError("csv_paths and panel_labels must have the same length")

    # Determine dimensions
    # Default to rcParams (from mplstyle)
    default_width, default_height = plt.rcParams["figure.figsize"]

    if panel_height is None:
        panel_height = default_height

    # Use default width, but override height if specified
    figsize = (default_width, panel_height)

    n_panels = len(csv_paths)

    # Create horizontal panels
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
    axes = axes[0]  # Flatten

    for i, (csv_path, label) in enumerate(zip(csv_paths, panel_labels)):
        print(f"Reading {csv_path}")
        df = pd.read_csv(csv_path)

        show_gap_annotation = i == annotate_gap_panel
        show_ylabel = i == 0

        _plot_boxplot_panel(
            axes[i],
            df,
            title=label,
            show_ylabel=show_ylabel,
            rsln_version=rsln_version,
            show_gap_annotation=show_gap_annotation,
            show_title=show_titles,
        )

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
    rsln_version: str,
    show_gap_annotation: bool,
    show_title: bool = True,
):
    """
    Plot a single panel with box plots for RSLN and RELN.
    """
    true_df, spurious_df = _split_data(df)
    rsln_col = f"rsln_{rsln_version}"

    data_dict = {
        "true_rsln": true_df[rsln_col].values,
        "spurious_rsln": spurious_df[rsln_col].values,
        "true_reln": true_df["reln"].values,
        "spurious_reln": spurious_df["reln"].values,
    }

    positions = [0, 1, 3, 4]
    data_list = [
        data_dict["true_rsln"],
        data_dict["spurious_rsln"],
        data_dict["true_reln"],
        data_dict["spurious_reln"],
    ]

    _create_boxplot(ax, data_list, positions)

    # First level: True/Spurious labels
    ax.set_xticks([0, 1, 3, 4])
    labels = ["True", "Spurious", "True", "Spurious"]
    ax.set_xticklabels(labels, rotation=0, ha="center")

    ax.set_xlim((-0.5, 4.75))

    _add_underbraces(ax)

    ax.set_yscale("log")

    if show_ylabel:
        ax.set_ylabel("Relative Leakage Norm")

    if show_title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle=":", axis="y", linewidth=0.5)

    if show_gap_annotation:
        _annotate_gap(ax, df, rsln_col, data_dict)


def _create_boxplot(ax, data_list, positions):
    """Create and style the boxplot."""
    bp = ax.boxplot(
        data_list,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", markersize=2, alpha=0.3),
    )

    colors = ["#1f77b4", "#d62728", "#1f77b4", "#d62728"]

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
        patch.set_alpha(0.7)

    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_color("black")
            item.set_linewidth(0.8)


def _add_underbraces(ax):
    """Add underbraces for Signal and Estimated groups."""
    # Position brackets closer to x-axis labels
    bracket_y = -0.13  # Position below x-axis in axis fraction
    text_y = -0.20  # Position for text labels

    # Draw underbraces using simple bracket style (horizontal line with end ticks)
    # Signal brace (spanning positions 0 and 1)
    brace_height = 0.015
    ax.plot(
        [-0.25, -0.25],
        [bracket_y, bracket_y + brace_height],
        transform=ax.get_xaxis_transform(),
        color="black",
        lw=0.8,
        clip_on=False,
        solid_capstyle="butt",
    )
    ax.plot(
        [-0.25, 1.25],
        [bracket_y, bracket_y],
        transform=ax.get_xaxis_transform(),
        color="black",
        lw=0.8,
        clip_on=False,
    )
    ax.plot(
        [1.25, 1.25],
        [bracket_y, bracket_y + brace_height],
        transform=ax.get_xaxis_transform(),
        color="black",
        lw=0.8,
        clip_on=False,
        solid_capstyle="butt",
    )
    ax.text(
        0.5,
        text_y,
        "Signal",
        ha="center",
        va="top",
        transform=ax.get_xaxis_transform(),
        fontsize=9,
    )

    # Estimated brace (spanning positions 3 and 4)
    ax.plot(
        [2.75, 2.75],
        [bracket_y, bracket_y + brace_height],
        transform=ax.get_xaxis_transform(),
        color="black",
        lw=0.8,
        clip_on=False,
        solid_capstyle="butt",
    )
    ax.plot(
        [2.75, 4.25],
        [bracket_y, bracket_y],
        transform=ax.get_xaxis_transform(),
        color="black",
        lw=0.8,
        clip_on=False,
    )
    ax.plot(
        [4.25, 4.25],
        [bracket_y, bracket_y + brace_height],
        transform=ax.get_xaxis_transform(),
        color="black",
        lw=0.8,
        clip_on=False,
        solid_capstyle="butt",
    )
    ax.text(
        3.5,
        text_y,
        "Estimated",
        ha="center",
        va="top",
        transform=ax.get_xaxis_transform(),
        fontsize=9,
    )


def _compute_tau_thresholds_per_trial(
    df: pd.DataFrame, rsln_col: str
) -> tuple[pd.Series, pd.Series]:
    """Compute tau_spur and tau_true thresholds per trial."""
    tau_spur_per_trial = df[df["is_true"] == 0].groupby("trial_id")[rsln_col].min()
    tau_true_per_trial = df[df["is_true"] == 1].groupby("trial_id")[rsln_col].max()
    return tau_spur_per_trial, tau_true_per_trial


def _annotate_gap(ax, df, rsln_col, data_dict):
    """Add gap annotation to the plot."""
    if "trial_id" in df.columns:
        tau_spur_per_trial, tau_true_per_trial = _compute_tau_thresholds_per_trial(df, rsln_col)

        tau_spur_val = tau_spur_per_trial.median()
        tau_true_val = tau_true_per_trial.median()
    else:
        tau_spur_val = np.min(data_dict["spurious_rsln"])
        tau_true_val = np.max(data_dict["true_rsln"])

    ax.axhline(
        tau_spur_val, color="red", linestyle="--", linewidth=1.0, alpha=0.7, zorder=10
    )
    ax.axhline(
        tau_true_val, color="blue", linestyle="--", linewidth=1.0, alpha=0.7, zorder=10
    )

    # Position arrow and labels between Signal boxes (between positions 0 and 1)
    arrow_x = 0.5
    text_x = 0.55

    ax.text(text_x, tau_spur_val, r"$\tau_{\mathrm{spur}}$", va="top", ha="left")
    ax.text(
        text_x, tau_true_val * 1.2, r"$\tau_{\mathrm{true}}$", va="bottom", ha="left"
    )

    arrow = FancyArrowPatch(
        (arrow_x, tau_spur_val),
        (arrow_x, tau_true_val),
        arrowstyle="<->",
        mutation_scale=15,
        linewidth=1.2,
        color="black",
        zorder=10,
    )
    ax.add_patch(arrow)

    gamma_y = np.sqrt(tau_spur_val * tau_true_val)
    ax.text(text_x, gamma_y, r"$\gamma$", va="center", ha="left", weight="bold")


def main(config_path: str) -> None:
    """
    CLI entry point.
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

    # 3. Parse scenarios
    scenarios = plot_cfg.get("scenarios", [])
    if not scenarios:
        raise ValueError("No scenarios defined in plotting.scenarios")

    csv_paths = []
    panel_labels = []
    for sc in scenarios:
        rel_path = sc["csv_path"]
        csv_paths.append(plotting_common.resolve_path(project_root, rel_path))
        panel_labels.append(sc["label"])

    # 4. Other params
    rsln_version = plot_cfg.get("rsln", {}).get("version", "perturbed")
    annotate_gap_panel = plot_cfg.get("figure", {}).get("annotate_gap_panel", 0)
    show_titles = plot_cfg.get("figure", {}).get("show_titles", True)
    panel_height = plot_cfg.get("figure", {}).get("panel_height", None)

    # 5. Output
    try:
        rel_out_dir = config["output"]["output_dir"]
        filename = config["output"]["filename"]
    except KeyError:
        raise KeyError("Config must contain output.output_dir and output.filename")

    out_dir = plotting_common.resolve_path(project_root, rel_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / filename

    print(f"Generating plot for {len(scenarios)} scenarios...")
    print(f"Output: {output_path}")

    plot_multi_scenario_boxplot(
        csv_paths=csv_paths,
        panel_labels=panel_labels,
        output_path=output_path,
        rsln_version=rsln_version,
        annotate_gap_panel=annotate_gap_panel,
        project_root=project_root,
        show_titles=show_titles,
        panel_height=panel_height,
    )

    # Generate Gamma (gap) CDF plot
    gamma_output_path = out_dir / (output_path.stem + "_gamma_cdf" + output_path.suffix)
    _generate_gamma_cdf_plot(csv_paths, panel_labels, rsln_version, gamma_output_path, show_titles)

    print("Done.")


def _generate_gamma_cdf_plot(
    csv_paths: list[Path],
    panel_labels: list[str],
    rsln_version: str,
    output_path: Path,
    show_titles: bool,
) -> None:
    """Generate and save CDF plot of gamma (gap) values across scenarios."""
    print(f"Generating Gamma CDF plot...")
    print(f"Output: {output_path}")

    gaps_list = []
    labels_list = []
    
    for csv_path, label in zip(csv_paths, panel_labels):
        df = pd.read_csv(csv_path)
        rsln_col = f"rsln_{rsln_version}"
        gap_values = _compute_gap_values(df, rsln_col)
        gaps_list.append(gap_values)
        labels_list.append(label)

    _plot_gamma_cdf(
        gaps_list=gaps_list,
        labels_list=labels_list,
        output_path=output_path,
        show_titles=show_titles,
    )


def _compute_gap_values(df: pd.DataFrame, rsln_col: str) -> np.ndarray:
    """
    Compute gap values gamma = tau_spur - tau_true per trial.
    """
    if "trial_id" not in df.columns:
        # Fallback if no trial_id (e.g. summarized data, though normally we have raw)
        true_df, spur_df = _split_data(df)
        tau_spur = spur_df[rsln_col].min()
        tau_true = true_df[rsln_col].max()
        return np.array([tau_spur - tau_true])

    # Use shared helper to compute per-trial thresholds
    tau_spur_per_trial, tau_true_per_trial = _compute_tau_thresholds_per_trial(df, rsln_col)

    # Align indices just in case
    common_trials = tau_spur_per_trial.index.intersection(tau_true_per_trial.index)
    gamma_per_trial = tau_spur_per_trial.loc[common_trials] - tau_true_per_trial.loc[common_trials]
    
    return gamma_per_trial.values


def _plot_gamma_cdf(
    gaps_list: list[np.ndarray],
    labels_list: list[str],
    output_path: Path,
    show_titles: bool = True,
):
    """
    Plot CDF of gamma values for multiple scenarios.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    for gaps, label in zip(gaps_list, labels_list):
        # Sort data for CDF
        sorted_gaps = np.sort(gaps)
        yvals = np.arange(1, len(sorted_gaps) + 1) / len(sorted_gaps)
        
        # Plot step function
        ax.step(sorted_gaps, yvals, where='post', label=label, linewidth=1.5)

    ax.set_xlabel(r"Gap $\gamma = \tau_{\mathrm{spur}} - \tau_{\mathrm{true}}$")
    ax.set_ylabel("CDF")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    # Legend
    if show_titles:
        ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fire.Fire(main)
