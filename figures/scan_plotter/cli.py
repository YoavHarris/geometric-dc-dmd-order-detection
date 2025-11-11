"""Command-line interface for scan plotter."""

import fire
from plotter import (
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
    ):
        """
        Create a single scan plot.
        
        Example:
            python cli.py single \\
                --csv_path=data.csv \\
                --x_param=snr_db \\
                --metric=order_hit_prob \\
                --output_path=output.png \\
                --working_point="{'num_modes': 2, 'noise_mode': 'gaussian'}"
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
                "title": title,
            }
        ]

        self.composer.compose(
            panels=panels, overall_title=title, output_path=output_path, show=show
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
                --output_path=output.png \\
                --working_point="{'num_modes': 2}"
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
                panels.append(
                    {
                        "df": panel_df,
                        "x_param": x_param,
                        "metric": metric,
                        "working_point": working_point,
                        "methods": methods,
                        "title": f"{panel_param.replace('_', ' ').title()} = {pval}",
                    }
                )

        self.composer.compose(
            panels=panels,
            layout="horizontal",
            overall_title=title,
            output_path=output_path,
            show=show,
        )

    def batch(self, config_path: str):
        """
        Run batch of plots from YAML config.

        Config should have:
            base: (common settings)
            plots: (list of plot specs)
        """
        import yaml

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
                if "panel_param" in merged:
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
                        methods=merged["methods"],
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
                    )

                print("  [OK]")
            except Exception as e:
                print(f"  [FAILED] {e}")

        print("=" * 60)
        print("Batch complete!")


if __name__ == "__main__":
    fire.Fire(ScanPlotterCLI)
