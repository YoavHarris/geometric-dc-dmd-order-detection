#!/usr/bin/env python
"""Build PDF/HTML report from a CSV + spec YAML (Fire CLI)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml
from fire import Fire

import plotting
import reporting


class CLI:
    # ──────────────────────────────────────────────────────────────────
    def build(self, *, csv: str, spec: str = "report.yaml") -> None:
        """
        Parameters
        ----------
        csv  : path to aggregated results CSV
        spec : YAML describing baseline, filters, plots, options
        """
        csv_path = Path(csv)
        cfg = yaml.safe_load(open(Path(spec), "r", encoding="utf-8"))
        baseline = cfg["baseline"]
        filters = cfg.get("filters", {})
        plots_cfg = cfg["plots"]
        opts = cfg.get("options", {})
        style = cfg.get("style", {})

        metric = cfg.get("metric", "order_hit_prob")
        ci = bool(cfg.get("ci_bands", False))
        output_dir = Path(cfg["output_dir"])
        report_format = cfg.get("output_format", "html")
        closest = bool(opts.get("closest_match", False))
        axis_scale: Dict[str, str] = cfg.get("axis_scale", {})

        df = pd.read_csv(csv_path)

        # -------------------------------------------------- apply filters
        if filters.get("methods"):
            df = df[df["method"].isin(filters["methods"])]
        if filters.get("noise_modes"):
            df = df[df["noise_mode"].isin(filters["noise_modes"])]

        # -------------------------------------------------- generate figs
        figs: List = []

        # Section 0 - working point table
        figs.append(plotting.baseline_table_fig(baseline))

        # Section 1 – sweeps
        sweeps_cfg = plots_cfg.get("sweeps", [])
        if sweeps_cfg in (["all"], "all"):
            vary_cols = [c for c in baseline if c != "noise_mode"]
        else:
            vary_cols = sweeps_cfg

        figs += plotting.make_sweep_plots(
            df,
            metric,
            baseline,
            ci,
            vary_cols=vary_cols,
            report_auc=bool(opts.get("report_auc", False)),
            auc_metrics=opts.get("auc_metrics", ["order_hit_prob"]),
            closest_match=closest,
            style=style,
            axis_scale=axis_scale,
        )

        # Section 2 – heat-maps
        for pair in plots_cfg.get("heatmaps", []):
            xcol, ycol = pair
            figs += plotting.make_heatmaps(
                df,
                xcol,
                ycol,
                metric,
                baseline,
                cmap=opts.get("heatmap_cmap", "viridis"),
                closest_match=closest,
                style=style,
                axis_scale=axis_scale,
            )

        # -------------------------------------------------- save report
        reporting.save(figs, output_dir=output_dir, report_format=report_format)

    # ──────────────────────────────────────────────────────────────────
    def header(self, *, csv: str) -> None:
        """Print column names of a CSV."""
        cols = pd.read_csv(csv, nrows=0).columns.tolist()
        print(cols)


if __name__ == "__main__":
    Fire(CLI)
