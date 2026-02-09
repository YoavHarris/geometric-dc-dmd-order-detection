"""
Clean, minimal parameter scan plotter.

Two core classes:
1. SingleScanPlotter - draws one scan on one axis
2. PanelComposer - arranges multiple scans into panels
"""

from pathlib import Path
from typing import Any
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from figures.common.plotting_common import (
    apply_style,
    resolve_project_root,
    load_yaml_config,
)


class SingleScanPlotter:
    """Draws a single parameter scan on a given axis."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @staticmethod
    def _detect_axis_type(values: np.ndarray, abs_tol: float = 0.01) -> str:
        """
        Detect axis spacing type from parameter values.

        Aligns with param_generator.py output:
        - 'linear': scale="lin" - constant differences
        - 'log': scale="log" - constant log-differences
        - 'categorical': type="list" - neither

        Uses absolute tolerance based on param_generator rounding errors
        (~1e-3 for log spacing) + CSV round-trip precision.

        Args:
            values: Array of parameter values from filtered data
            abs_tol: Absolute std tolerance (default 0.01, 10x generator error)

        Returns:
            'linear', 'log', or 'categorical'
        """
        unique_vals = np.unique(values)
        n = len(unique_vals)

        if n < 3:
            return "linear"  # Too few points to distinguish

        sorted_vals = np.sort(unique_vals)

        # Test 1: Are differences constant? (linear spacing)
        diffs = np.diff(sorted_vals)
        if np.std(diffs) < abs_tol:
            return "linear"

        # Test 2: Are log-differences constant? (log spacing)
        if np.all(sorted_vals > 0):
            log_diffs = np.diff(np.log(sorted_vals))
            if np.std(log_diffs) < abs_tol:
                return "log"

        # Neither linear nor log: categorical
        return "categorical"

    def plot(
        self,
        ax: plt.Axes,
        df: pd.DataFrame,
        x_param: str,
        metric: str,
        working_point: dict[str, Any] | None = None,
        methods: list[str] | None = None,
        xscale: str | None = None,
        xlim: tuple[float, float] | None = None,
        max_ticks: int | None = None,
        show_legend: bool = True,
        show_ylabel: bool = True,
        show_xlabel: bool = True,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ):
        """
        Plot one parameter scan on the given axis.

        Args:
            ax: Matplotlib axis
            df: Data (already filtered to working point)
            x_param: Parameter on x-axis
            metric: Metric on y-axis
            working_point: Working point dict (for vertical line)
            methods: List of methods to plot (if None, plot all)
            xscale: Force axis scale ('linear', 'log', 'categorical', or None for auto)
            xlim: X-axis limits as (min, max) tuple (if None, use automatic)
            max_ticks: Maximum number of ticks (overrides global config if provided)
            show_legend: Show legend
            show_ylabel: Show y-axis label
            show_xlabel: Show x-axis label
            title: Axis title
            xlabel: Custom x-axis label (overrides auto-formatting, supports LaTeX)
            ylabel: Custom y-axis label (overrides auto-formatting, supports LaTeX)
        """
        if methods is None:
            methods = sorted(df["method"].unique())

        # Detect or override axis type
        if xscale is not None:
            axis_type = xscale
        else:
            axis_type = self._detect_axis_type(df[x_param].values)

        # Plot each method
        for method in methods:
            method_data = df[df["method"] == method]

            # Group by x and compute mean
            grouped = method_data.groupby(x_param)[metric].mean()
            x = grouped.index.values
            y = grouped.values

            # Get style
            style = self._get_style(method)

            label = method
            if method in ["ESR-Energy", "NestedDMD", "FixedEigenvalueKVFit"]:
                label += " (Ours)"

            # Plot
            ax.plot(
                x,
                y,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=None,
                marker=style.get("marker", "o"),
                markersize=None,
                label=label,
            )

        # Apply axis scaling based on detected type
        if axis_type == "log":
            ax.set_xscale("log")
        elif axis_type == "categorical":
            # Set explicit tick positions for categorical data
            unique_vals = sorted(df[x_param].unique())

            # Fallback to linear scaling for dense numerical data to respect max_ticks
            is_numeric = np.issubdtype(df[x_param].dtype, np.number)
            if max_ticks is not None:
                tick_limit = max_ticks
            else:
                tick_limit = self.config.get("ticks", {}).get("max_ticks", 6)

            if is_numeric and len(unique_vals) > tick_limit:
                axis_type = "linear"
            else:
                ax.set_xticks(unique_vals)

        # Apply tick number limit (for non-categorical axes)
        if axis_type != "categorical":
            from matplotlib.ticker import MaxNLocator

            if max_ticks is not None:
                max_tick_count = max_ticks
            else:
                tick_cfg = self.config.get("ticks", {})
                max_tick_count = tick_cfg.get("max_ticks", 6)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=max_tick_count, integer=False))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=max_tick_count, integer=False))

        # Add vertical line at working point value (if provided)
        wp_line_cfg = self.config.get("working_point_line", {})
        if working_point and x_param in working_point and wp_line_cfg.get("show", True):
            wp_value = working_point[x_param]
            ax.axvline(
                wp_value,
                color=wp_line_cfg.get("color", "black"),
                linestyle=wp_line_cfg.get("linestyle", "--"),
                linewidth=wp_line_cfg.get("linewidth", 0.8),
                alpha=wp_line_cfg.get("alpha", 0.7),
                zorder=0,
                label="Working Point",
            )

        # Apply x-axis limits (if provided)
        if xlim is not None:
            ax.set_xlim(xlim)

        # Format
        if show_ylabel:
            ylabel_text = ylabel if ylabel is not None else self._format_label(metric)
            ax.set_ylabel(
                ylabel_text,
            )
        if show_xlabel:
            xlabel_text = xlabel if xlabel is not None else self._format_label(x_param)
            ax.set_xlabel(
                xlabel_text,
            )
        ax.grid(True, alpha=0.3)

        if title:
            ax.set_title(title)

        if show_legend:
            ax.legend()

    def _get_style(self, method: str) -> dict[str, str]:
        """Get color, linestyle, and marker for method."""
        methods_cfg = self.config.get("methods", {})
        if method in methods_cfg:
            return methods_cfg[method]
        else:
            defaults = self.config.get("defaults", {})
            return {
                "color": defaults.get("color", "#999999"),
                "linestyle": defaults.get("linestyle", ":"),
                "marker": defaults.get("marker", "o"),
            }

    def _format_label(self, text: str) -> str:
        """
        Format parameter/metric name.

        Priority:
        1. Check global parameter_labels mapping in config
        2. Fall back to auto-formatting (replace _ with space, title case)
        """
        # Check if there's a global mapping for this parameter
        param_labels = self.config.get("parameter_labels", {})
        if text in param_labels:
            return param_labels[text]

        # Fall back to auto-formatting
        return text.replace("_", " ").title()

    def format_value(self, param_name: str, value: Any) -> str:
        """
        Format parameter value for display.

        Priority:
        1. Check parameter_value_labels mapping in config
        2. Fall back to string representation of value

        Args:
            param_name: Name of the parameter
            value: Value to format

        Returns:
            Formatted string representation of value
        """
        value_labels = self.config.get("parameter_value_labels", {})
        if param_name in value_labels and value in value_labels[param_name]:
            return value_labels[param_name][value]

        # Fall back to string representation
        return str(value)


class PanelComposer:
    """Composes multiple SingleScanPlotters into a panel figure."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.plotter = SingleScanPlotter(config)

    def compose(
        self,
        panels: list[dict[str, Any]],
        layout: str = "horizontal",
        overall_title: str | None = None,
        output_path: str | None = None,
        show: bool = False,
    ):
        """
        Compose multiple scan panels into one figure.

        Args:
            panels: List of panel specs, each with keys:
                - df: DataFrame (already filtered)
                - x_param: str
                - metric: str
                - title: Optional[str]
            layout: 'horizontal' or 'vertical'
            overall_title: Title for entire figure
            output_path: Path to save
            show: Display figure?
        """
        n_panels = len(panels)

        # Create figure
        fig_cfg = self.config["figure"]
        total_width = fig_cfg["total_width"]
        panel_height = fig_cfg["panel_height"]

        if layout == "horizontal":
            figsize = (total_width, panel_height)
            fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
            axes = axes[0]
        else:  # vertical
            figsize = (total_width, panel_height * n_panels)
            fig, axes = plt.subplots(n_panels, 1, figsize=figsize, squeeze=False)
            axes = axes[:, 0]

        # Plot each panel
        for i, panel_spec in enumerate(panels):
            show_ylabel = (i == 0) if layout == "horizontal" else True
            # Show xlabel only on middle panel for horizontal layouts (unless overridden)
            default_show_xlabel = (
                (i == n_panels // 2) if layout == "horizontal" else True
            )
            show_xlabel = panel_spec.get("show_xlabel", default_show_xlabel)

            self.plotter.plot(
                ax=axes[i],
                df=panel_spec["df"],
                x_param=panel_spec["x_param"],
                metric=panel_spec["metric"],
                working_point=panel_spec.get("working_point"),
                methods=panel_spec.get("methods"),
                xscale=panel_spec.get("xscale"),
                xlim=panel_spec.get("xlim"),
                max_ticks=panel_spec.get("max_ticks"),
                show_legend=False,
                show_ylabel=show_ylabel,
                show_xlabel=show_xlabel,
                title=panel_spec.get("title"),
                xlabel=panel_spec.get("xlabel"),
                ylabel=panel_spec.get("ylabel"),
            )

        # Add shared legend based on layout
        # Get handles and labels from any axis (they should all have the same)
        handles, labels = axes[0].get_legend_handles_labels()

        if layout == "horizontal":
            # Horizontal layout: shared legend at the bottom
            # Estimate legend width: ~0.1" per character + 0.5" per entry for marker/spacing
            estimated_width = sum(len(label) * 0.1 + 0.5 for label in labels)

            # Decide number of columns based on estimated width
            # Use single row if it fits in 95% of figure width, otherwise use 2 rows
            if estimated_width < total_width * 0.95:
                ncol = len(labels)  # Single row
            else:
                ncol = max(1, (len(labels) + 1) // 2)  # Two rows

            # Create a centered legend below the subplots
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.1),  # Position below xlabel
                ncol=ncol,
                frameon=True,
            )
        else:
            # Vertical layout: shared legend on the right
            fig.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),  # Position to the right of panels
                frameon=True,
            )

        plt.tight_layout()

        # Save
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, bbox_inches="tight")
            print(f"Saved: {output_path}")

        if show:
            plt.show()

        plt.close()


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load configuration and apply plotting style.

    Args:
        path: Path to config file (defaults to design_config.yaml)

    Returns:
        Configuration dictionary
    """
    if path is None:
        path = Path(__file__).parent / "design_config.yaml"

    # Load config using shared utility
    config = load_yaml_config(path)

    # Resolve project root and apply style
    project_root = resolve_project_root(config, path)

    # Apply style if plotting config exists
    if "plotting" in config:
        apply_style(config["plotting"], project_root)

    return config


def load_data(csv_path: str) -> pd.DataFrame:
    """Load CSV data."""
    return pd.read_csv(csv_path)


def filter_data(
    df: pd.DataFrame,
    working_point: dict[str, Any],
    exclude_params: list[str] | None = None,
) -> pd.DataFrame:
    """
    Filter data to working point.

    Args:
        df: Input dataframe
        working_point: Parameter values to fix
        exclude_params: Parameters to exclude from filtering (vary in scan)
    """
    exclude_params = exclude_params or []
    filtered = df.copy()

    # Process filters
    params = list(working_point.keys())

    for param in params:
        value = working_point[param]
        if param in exclude_params or param not in df.columns:
            continue

        if isinstance(value, (float, np.floating)):
            mask = np.isclose(filtered[param], value, rtol=1e-5, atol=1e-8)
            filtered = filtered[mask]
        else:
            filtered = filtered[filtered[param] == value]



    return filtered
