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

plt.style.use("mplstyle_files/chaos_single.mplstyle")


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
        show_legend: bool = True,
        show_ylabel: bool = True,
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
            show_legend: Show legend?
            show_ylabel: Show y-axis label?
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

        # Get styling config
        line_cfg = self.config.get("lines", {})
        linewidth = line_cfg.get("linewidth", 2)
        markersize = line_cfg.get("markersize", 5)

        # Plot each method
        for method in methods:
            method_data = df[df["method"] == method]

            # Group by x and compute mean
            grouped = method_data.groupby(x_param)[metric].mean()
            x = grouped.index.values
            y = grouped.values

            # Get style
            style = self._get_style(method)

            # Plot
            ax.plot(
                x,
                y,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=linewidth,
                marker=style.get("marker", "o"),
                markersize=markersize,
                label=method,
            )

        # Apply axis scaling based on detected type
        if axis_type == "log":
            ax.set_xscale("log")
        elif axis_type == "categorical":
            # Set explicit tick positions for categorical data
            unique_vals = sorted(df[x_param].unique())
            ax.set_xticks(unique_vals)

        # Add vertical line at working point value (if provided)
        if working_point and x_param in working_point:
            wp_value = working_point[x_param]
            ax.axvline(
                wp_value,
                color="black",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
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
                fontsize=self.config.get("labels", {}).get("fontsize", 10),
            )
        xlabel_text = xlabel if xlabel is not None else self._format_label(x_param)
        ax.set_xlabel(
            xlabel_text,
            fontsize=self.config.get("labels", {}).get("fontsize", 10),
        )
        ax.grid(True, alpha=0.3)

        if title:
            ax.set_title(
                title, fontsize=self.config.get("labels", {}).get("title_fontsize", 14)
            )

        if show_legend:
            legend_cfg = self.config.get("legend", {})
            ax.legend(
                fontsize=legend_cfg.get("fontsize", 9),
                framealpha=legend_cfg.get("framealpha", 0.9),
            )

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
        fig_cfg = self.config.get("figure", {})
        panel_width = fig_cfg.get("panel_width", 5)
        panel_height = fig_cfg.get("panel_height", 4)
        dpi = fig_cfg.get("dpi", 150)

        if layout == "horizontal":
            figsize = (panel_width * n_panels, panel_height)
            fig, axes = plt.subplots(
                1, n_panels, figsize=figsize, dpi=dpi, squeeze=False
            )
            axes = axes[0]
        else:  # vertical
            figsize = (panel_width, panel_height * n_panels)
            fig, axes = plt.subplots(
                n_panels, 1, figsize=figsize, dpi=dpi, squeeze=False
            )
            axes = axes[:, 0]

        # Plot each panel
        for i, panel_spec in enumerate(panels):
            show_ylabel = (i == 0) if layout == "horizontal" else True
            show_legend = (i == n_panels - 1) if layout == "horizontal" else True

            self.plotter.plot(
                ax=axes[i],
                df=panel_spec["df"],
                x_param=panel_spec["x_param"],
                metric=panel_spec["metric"],
                working_point=panel_spec.get("working_point"),
                methods=panel_spec.get("methods"),
                xscale=panel_spec.get("xscale"),
                xlim=panel_spec.get("xlim"),
                show_legend=show_legend,
                show_ylabel=show_ylabel,
                title=panel_spec.get("title"),
                xlabel=panel_spec.get("xlabel"),
                ylabel=panel_spec.get("ylabel"),
            )

            # Adjust legend for rightmost panel
            if show_legend and layout == "horizontal":
                legend_cfg = self.config.get("legend", {})
                axes[i].legend(
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    fontsize=legend_cfg.get("fontsize", 9),
                    framealpha=legend_cfg.get("framealpha", 0.9),
                )

        # Overall title
        if overall_title:
            label_cfg = self.config.get("labels", {})
            fig.suptitle(
                overall_title,
                fontsize=label_cfg.get("title_fontsize", 14),
                fontweight="bold",
                y=1.02,
            )

        plt.tight_layout()

        # Save
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved: {output_path}")

        if show:
            plt.show()

        plt.close()


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load configuration."""
    if path is None:
        path = Path(__file__).parent / "design_config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


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

    for param, value in working_point.items():
        if param in exclude_params or param not in df.columns:
            continue

        if isinstance(value, (float, np.floating)):
            mask = np.isclose(filtered[param], value, rtol=1e-5, atol=1e-8)
            filtered = filtered[mask]
        else:
            filtered = filtered[filtered[param] == value]

    return filtered
