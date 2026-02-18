"""Command-line interface for scan plotter."""

import fire
import yaml

from .plotter import (
    SingleScanPlotter,
    PanelComposer,
    load_config,
    load_data,
    filter_data,
)


class ScanPlotterCLI:
    """CLI for creating parameter scan plots."""

    def __init__(self):
        self.config = load_config()
        self.plotter = SingleScanPlotter(self.config)
        self.composer = PanelComposer(self.config)

    def single(
        self,
        csv_path: str,
        x_param: str,
        metric: str,
        output_path: str,
        working_point: dict,
        title: str = None,
        show: bool = False,
        methods: list[str] | None = None,
        xscale: str | None = None,
        xlim: tuple[float, float] | None = None,
        max_ticks: int | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        style_mode: str | None = None,
    ):
        """
        Create a single scan plot.
        
        Example:
            python cli.py single \\
                --csv_path=data.csv \\
                --x_param=snr_db \\
                --metric=order_hit_prob \\
                --output_path=output.pdf \\
                --working_point="{'num_modes': 2, 'noise_mode': 'gaussian'}" \\
                --xlim="(0.85, 0.95)"
        """
        df = load_data(csv_path)
        filtered = filter_data(df, working_point, exclude_params=[x_param])

        if len(filtered) == 0:
            raise ValueError("No data matches working point")

        if methods:
            filtered = filtered[filtered["method"].isin(methods)]
            if len(filtered) == 0:
                raise ValueError("No data matches specified methods")

        # Use composer with single panel
        panels = [
            {
                "df": filtered,
                "x_param": x_param,
                "metric": metric,
                "working_point": working_point,
                "methods": methods,
                "xscale": xscale,
                "xlim": xlim,
                "max_ticks": max_ticks,
                "title": title,
                "xlabel": xlabel,
                "ylabel": ylabel,
            }
        ]

        self.composer.compose(
            panels=panels, overall_title=title, output_path=output_path, show=show,
            style_mode=style_mode,
        )

    def multi(
        self,
        csv_path: str,
        x_param: str,
        metric: str,
        output_path: str,
        working_point: dict,
        panel_param: str,
        panel_values: list,
        title: str = None,
        show: bool = False,
        methods: list[str] | None = None,
        xscale: str | None = None,
        xlim: tuple[float, float] | None = None,
        max_ticks: int | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        style_mode: str | None = None,
    ):
        """
        Create multi-panel plot varying panel_param.
        
        Example:
            python cli.py multi \\
                --csv_path=data.csv \\
                --x_param=snr_db \\
                --metric=order_hit_prob \\
                --panel_param=noise_mode \\
                --panel_values="['gaussian', 'student_t']" \\
                --output_path=output.pdf \\
                --working_point="{'num_modes': 2}" \\
                --xlim="(0.85, 0.95)"
        """
        df = load_data(csv_path)

        # Filter to working point (exclude both x_param and panel_param)
        filtered = filter_data(df, working_point, exclude_params=[x_param, panel_param])

        if len(filtered) == 0:
            raise ValueError("No data matches working point")

        if methods:
            filtered = filtered[filtered["method"].isin(methods)]
            if len(filtered) == 0:
                raise ValueError("No data matches specified methods")

        # Create panel for each panel_value
        panels = []
        for pval in panel_values:
            panel_df = filtered[filtered[panel_param] == pval]
            if len(panel_df) > 0:
                # Format parameter name and value using the plotter's methods
                param_label = self.plotter._format_label(panel_param)
                value_label = self.plotter.format_value(panel_param, pval)
                panels.append(
                    {
                        "df": panel_df,
                        "x_param": x_param,
                        "metric": metric,
                        "working_point": working_point,
                        "methods": methods,
                        "xscale": xscale,
                        "xlim": xlim,
                        "max_ticks": max_ticks,
                        "title": f"{param_label} = {value_label}",
                        "xlabel": xlabel,
                        "ylabel": ylabel,
                    }
                )

        self.composer.compose(
            panels=panels,
            layout="horizontal",
            overall_title=title,
            output_path=output_path,
            show=show,
            style_mode=style_mode,
        )

    def multi_xparam(
        self,
        csv_path: str,
        x_params: list[str],
        metric: str,
        output_path: str,
        working_point: dict,
        title: str = None,
        show: bool = False,
        methods: list[str] | None = None,
        xlim: tuple[float, float] | None = None,
        max_ticks: int | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        style_mode: str | None = None,
    ):
        """
        Create multi-panel plot with different x-axis parameter per panel.

        Each panel shows the same metric, but scanned against a different parameter.
        This is useful for comparing sensitivity across multiple parameters.

        Example:
            python cli.py multi_xparam \\
                --csv_path=data.csv \\
                --x_params="['snr_db', 'freq_sep', 'eig_mag']" \\
                --metric=order_hit_prob \\
                --output_path=output.pdf \\
                --working_point="{'num_modes': 3, 'noise_mode': 'gaussian'}"
        """
        df = load_data(csv_path)

        panels = []
        for x_param in x_params:
            # Filter excluding current x_param
            filtered = filter_data(df, working_point, exclude_params=[x_param])

            if len(filtered) == 0:
                print(f"Warning: No data for x_param={x_param}, skipping")
                continue

            if methods:
                filtered = filtered[filtered["method"].isin(methods)]
                if len(filtered) == 0:
                    print(
                        f"Warning: No data for x_param={x_param} with specified methods, skipping"
                    )
                    continue

            # Create panel title from x_param label
            param_label = self.plotter._format_label(x_param)

            panels.append(
                {
                    "df": filtered,
                    "x_param": x_param,
                    "metric": metric,
                    "working_point": working_point,
                    "methods": methods,
                    "xscale": None,  # Auto-detect per panel
                    "xlim": xlim,
                    "max_ticks": max_ticks,
                    "title": param_label,
                    "xlabel": None,  # Auto from x_param
                    "ylabel": ylabel,
                    "show_xlabel": True,  # Always show xlabel (different per panel)
                }
            )

        if not panels:
            raise ValueError("No valid panels created - check data availability")

        self.composer.compose(
            panels=panels,
            layout="horizontal",
            overall_title=title,
            output_path=output_path,
            show=show,
            style_mode=style_mode,
        )

    def batch(self, config_path: str):
        """
        Run batch of plots from YAML config.

        Config should have:
            base: (common settings)
            plots: (list of plot specs)
        """

        with open(config_path, "r") as f:
            batch_cfg = yaml.safe_load(f)

        base = batch_cfg.get("base", {})
        baseline_wp = batch_cfg.get("baseline_working_point", {})
        plots = batch_cfg["plots"]

        print(f"Batch: {len(plots)} plots")
        print("=" * 60)

        data_cache = {}

        for i, plot_cfg in enumerate(plots, 1):
            # Merge base with plot config
            merged = {**base, **plot_cfg}

            # Deep merge working_point: baseline -> plot-specific
            plot_wp = plot_cfg.get("working_point", {})
            merged["working_point"] = {**baseline_wp, **plot_wp}

            print(f"[{i}/{len(plots)}] {merged.get('title', f'Plot {i}')}")

            try:
                csv_path = merged["csv_path"]

                # Load data (cached)
                if csv_path not in data_cache:
                    data_cache[csv_path] = load_data(csv_path)
                df = data_cache[csv_path]

                # Route to appropriate method
                if "x_params" in merged:
                    # Multi x-param mode (different x-axis per panel)
                    self.multi_xparam(
                        csv_path=csv_path,
                        x_params=merged["x_params"],
                        metric=merged["metric"],
                        output_path=merged["output_path"],
                        working_point=merged["working_point"],
                        title=merged.get("title"),
                        show=merged.get("show", False),
                        methods=merged.get("methods"),
                        xlim=merged.get("xlim"),
                        max_ticks=merged.get("max_ticks"),
                        xlabel=merged.get("xlabel"),
                        ylabel=merged.get("ylabel"),
                        style_mode=merged.get("style_mode"),
                    )
                elif "panel_param" in merged:
                    # Multi-panel mode (same x-axis, varying panel_param)
                    self.multi(
                        csv_path=csv_path,
                        x_param=merged["x_param"],
                        metric=merged["metric"],
                        output_path=merged["output_path"],
                        working_point=merged["working_point"],
                        panel_param=merged["panel_param"],
                        panel_values=merged["panel_values"],
                        title=merged.get("title"),
                        show=merged.get("show", False),
                        methods=merged.get("methods"),
                        xscale=merged.get("xscale"),
                        xlim=merged.get("xlim"),
                        max_ticks=merged.get("max_ticks"),
                        xlabel=merged.get("xlabel"),
                        ylabel=merged.get("ylabel"),
                        style_mode=merged.get("style_mode"),
                    )
                else:
                    self.single(
                        csv_path=csv_path,
                        x_param=merged["x_param"],
                        metric=merged["metric"],
                        output_path=merged["output_path"],
                        working_point=merged["working_point"],
                        title=merged.get("title"),
                        show=merged.get("show", False),
                        methods=merged.get("methods"),
                        xscale=merged.get("xscale"),
                        xlim=merged.get("xlim"),
                        max_ticks=merged.get("max_ticks"),
                        xlabel=merged.get("xlabel"),
                        ylabel=merged.get("ylabel"),
                        style_mode=merged.get("style_mode"),
                    )

                print("  [OK]")
            except Exception as e:
                print(f"  [FAILED] {e}")

        print("=" * 60)
        print("Batch complete!")


if __name__ == "__main__":
    fire.Fire(ScanPlotterCLI)
