#!/usr/bin/env python3
"""
Generate normalized AUC tables from one-dimensional metric scans.

Each cell is the trapezoidal AUC of a selected metric along a one-dimensional
parameter sweep, divided by the sweep span. For metrics already in [0, 1], this
is the sweep-average metric value over the chosen scan range.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import fire
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from figures.scans.plotter import filter_data


DEFAULT_PARAM_COLS = ["snr_db", "freq_sep", "eig_mag", "top_amplitude", "max_rank"]

PARAM_LATEX = {
    "snr_db": "SNR",
    "freq_sep": "$\\Delta\\theta$",
    "eig_mag": "$r$",
    "top_amplitude": "$\\kappa_b$",
    "max_rank": "$M$",
}

DEFAULT_METHODS = [
    "STC",
    "ResDMDResidual",
    "ESR-Energy",
    "NestedDMD",
    "FixedEigenvalueKVFit",
]

DEFAULT_BASE_WP: dict[str, Any] = {
    "snr_db": 0.0,
    "freq_sep": 0.007,
    "eig_mag": 1.0,
    "delays_over_timesteps": 0.33,
    "top_amplitude": 2.5,
    "max_rank": 15,
    "temporal_dim": 200,
    "noise_mode": "gaussian",
}

DEFAULT_NUM_MODES = [2, 3, 5]
DEFAULT_METRIC_COL = "pr_auc_mean"


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "metric"


def default_metric_label(metric_col: str) -> str:
    labels = {
        "pr_auc_mean": "PR-AUC",
        "roc_auc_mean": "ROC-AUC",
        "order_hit_prob": "order-hit probability",
        "accuracy": "accuracy",
        "f1": "F1",
        "precision": "precision",
        "recall": "recall",
    }
    return labels.get(metric_col, metric_col.replace("_", " "))


def compute_normalized_auc(
    df: pd.DataFrame,
    methods: list[str],
    param_cols: list[str],
    metric_col: str,
) -> dict[tuple[int, str, str], float]:
    if metric_col not in df.columns:
        raise ValueError(f"Expected column '{metric_col}' in results CSV")

    auc_dict: dict[tuple[int, str, str], float] = {}

    for param in param_cols:
        if param not in df.columns:
            continue

        for m in sorted(df["num_modes"].unique()):
            df_m = df[df["num_modes"] == m]
            for method in methods:
                df_mm = df_m[df_m["method"] == method].copy()
                if df_mm.empty:
                    continue

                grouped = (
                    df_mm.groupby(param)[metric_col]
                    .mean()
                    .reset_index()
                    .sort_values(param)
                )
                grouped = grouped.dropna(subset=[metric_col])

                if len(grouped) < 2:
                    continue

                x = grouped[param].to_numpy()
                y = grouped[metric_col].to_numpy()
                span = float(x[-1] - x[0])
                if span <= 0:
                    continue

                auc_dict[(m, method, param)] = float(trapezoid(y, x)) / span

    return auc_dict


def compute_wp_auc(
    df: pd.DataFrame,
    methods: list[str],
    param_cols: list[str],
    base_wp: dict[str, Any],
    num_modes: list[int],
    metric_col: str,
) -> dict[tuple[int, str, str], float]:
    auc_dict: dict[tuple[int, str, str], float] = {}
    for param in param_cols:
        if param not in df.columns:
            continue
        wp = base_wp.copy()
        if param == "max_rank":
            wp.pop("max_rank", None)
        filtered = filter_data(df, wp, exclude_params=[param])
        filtered = filtered[filtered["num_modes"].isin(num_modes)]
        auc_dict.update(
            compute_normalized_auc(filtered, methods, [param], metric_col=metric_col)
        )
    return auc_dict


def count_wins(
    auc_dict: dict[tuple[int, str, str], float],
    methods_with_data: list[str],
    method: str,
    all_ms: list[int],
    param_cols: list[str],
) -> int:
    wins = 0
    for m in all_ms:
        for param in param_cols:
            key = (m, method, param)
            if key not in auc_dict:
                continue

            value = auc_dict[key]
            best_value = max(
                auc_dict.get((m, candidate, param), -np.inf)
                for candidate in methods_with_data
            )
            if f"{value:.3f}" == f"{best_value:.3f}" and best_value > 0:
                wins += 1
    return wins


def format_wide_latex_table(
    auc_dict: dict[tuple[int, str, str], float],
    methods: list[str],
    param_cols: list[str],
    table_label: str,
    caption: str,
    output_path: Path,
) -> None:
    lines: list[str] = []
    all_ms = sorted({key[0] for key in auc_dict.keys()})
    methods_with_data = [
        method for method in methods if any(key[1] == method for key in auc_dict.keys())
    ]

    if not methods_with_data or not all_ms:
        print(f"Warning: No data for {table_label}", file=sys.stderr)
        return

    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append(f"\\begin{{tabular}}{{l{'c' * (len(param_cols) * len(all_ms) + 1)}}}")
    lines.append("\\hline")

    header1 = [""]
    for m in all_ms:
        header1.append(f"\\multicolumn{{{len(param_cols)}}}{{c}}{{$m = {m}$}}")
    header1.append("")
    lines.append(" & ".join(header1) + " \\\\")

    header2 = ["Method"]
    param_headers = [PARAM_LATEX.get(p, p.replace("_", " ")) for p in param_cols]
    for _ in all_ms:
        header2.extend(param_headers)
    header2.append("\\# Wins")
    lines.append(" & ".join(header2) + " \\\\")
    lines.append("\\hline")

    for method in methods_with_data:
        row = [method]
        for m in all_ms:
            for param in param_cols:
                key = (m, method, param)
                if key not in auc_dict:
                    row.append("--")
                    continue

                value = auc_dict[key]
                best_value = max(
                    auc_dict.get((m, candidate, param), -np.inf)
                    for candidate in methods_with_data
                )

                cell = f"{value:.3f}"
                if cell == f"{best_value:.3f}" and best_value > 0:
                    cell = f"\\textbf{{{cell}}}"
                row.append(cell)

        wins = count_wins(auc_dict, methods_with_data, method, all_ms, param_cols)
        row.append(str(wins))
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\caption{" + caption + "}")
    lines.append(f"\\label{{tab:{table_label}}}")
    lines.append("\\end{table*}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated: {output_path}")


def format_unified_latex_table(
    auc_by_spatial_dim: dict[int, dict[tuple[int, str, str], float]],
    methods: list[str],
    param_cols: list[str],
    table_label: str,
    caption: str,
    output_path: Path,
) -> None:
    lines: list[str] = []
    all_ms = sorted(
        {
            key[0]
            for auc_dict in auc_by_spatial_dim.values()
            for key in auc_dict.keys()
        }
    )
    methods_with_data = [
        method
        for method in methods
        if any(
            key[1] == method
            for auc_dict in auc_by_spatial_dim.values()
            for key in auc_dict.keys()
        )
    ]

    if not methods_with_data or not all_ms:
        print(f"Warning: No data for {table_label}", file=sys.stderr)
        return

    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{2pt}")
    lines.append(
        f"\\begin{{tabular}}{{ll{'c' * (len(param_cols) * len(all_ms) + 1)}}}"
    )
    lines.append("\\hline")

    header1 = ["", ""]
    for m in all_ms:
        header1.append(f"\\multicolumn{{{len(param_cols)}}}{{c}}{{$m = {m}$}}")
    header1.append("")
    lines.append(" & ".join(header1) + " \\\\")

    header2 = ["$D$", "Method"]
    param_headers = [PARAM_LATEX.get(p, p.replace("_", " ")) for p in param_cols]
    for _ in all_ms:
        header2.extend(param_headers)
    header2.append("\\# Wins")
    lines.append(" & ".join(header2) + " \\\\")
    lines.append("\\hline")

    for spatial_dim, auc_dict in sorted(auc_by_spatial_dim.items()):
        first_row_for_dim = True
        for method in methods_with_data:
            if not any(key[1] == method for key in auc_dict.keys()):
                continue

            row = [str(spatial_dim) if first_row_for_dim else "", method]
            first_row_for_dim = False
            for m in all_ms:
                for param in param_cols:
                    key = (m, method, param)
                    if key not in auc_dict:
                        row.append("--")
                        continue

                    value = auc_dict[key]
                    best_value = max(
                        auc_dict.get((m, candidate, param), -np.inf)
                        for candidate in methods_with_data
                    )

                    cell = f"{value:.3f}"
                    if cell == f"{best_value:.3f}" and best_value > 0:
                        cell = f"\\textbf{{{cell}}}"
                    row.append(cell)

            wins = count_wins(auc_dict, methods_with_data, method, all_ms, param_cols)
            row.append(str(wins))
            lines.append(" & ".join(row) + " \\\\")

        lines.append("\\hline")

    lines.append("\\end{tabular}")
    lines.append("\\caption{" + caption + "}")
    lines.append(f"\\label{{tab:{table_label}}}")
    lines.append("\\end{table*}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated: {output_path}")


def caption_for(metric_label: str, spatial_dim: int) -> str:
    return (
        f"Normalized {metric_label} for $D={spatial_dim}$ (Gaussian noise) "
        "across one-dimensional parameter sweeps at "
        "$\\mathrm{SNR}=0\\,\\mathrm{dB}$, $\\Delta\\theta=0.007$, "
        "$r=1.0$, $\\kappa_b=2.5$, $M=15$ (except $M$ sweep)."
    )


def unified_caption_for(metric_label: str) -> str:
    return (
        f"Normalized {metric_label} for Gaussian-noise DC experiments across "
        "one-dimensional parameter sweeps at "
        "$\\mathrm{SNR}=0\\,\\mathrm{dB}$, $\\Delta\\theta=0.007$, "
        "$r=1.0$, $\\kappa_b=2.5$, $M=15$ (except $M$ sweep)."
    )


def main(
    csv_path: str = "figures/scans/data/180526/combined_results.csv",
    output_dir: str | None = None,
    output_path: str | None = None,
    spatial_dims: list[int] | None = None,
    num_modes: list[int] | None = None,
    noise_mode: str = "gaussian",
    metric_col: str = DEFAULT_METRIC_COL,
    metric_label: str | None = None,
    methods: list[str] | None = None,
    param_cols: list[str] | None = None,
    table_prefix: str | None = None,
) -> None:
    spatial_dims = spatial_dims or [1, 2, 3]
    num_modes = num_modes or DEFAULT_NUM_MODES
    methods = methods or DEFAULT_METHODS
    param_cols = param_cols or DEFAULT_PARAM_COLS
    metric_label = metric_label or default_metric_label(metric_col)
    table_prefix = table_prefix or slugify(metric_col.removesuffix("_mean"))
    unified_output_path = Path(output_path) if output_path else None
    if output_dir:
        out_dir = Path(output_dir)
    elif unified_output_path:
        out_dir = unified_output_path.parent
    else:
        out_dir = Path(f"figures/scans/outputs/180526_{table_prefix}")

    df = pd.read_csv(csv_path)
    if "noise_mode" in df.columns:
        df = df[df["noise_mode"] == noise_mode].copy()

    base_wp = DEFAULT_BASE_WP.copy()
    base_wp["noise_mode"] = noise_mode

    auc_by_spatial_dim: dict[int, dict[tuple[int, str, str], float]] = {}
    for spatial_dim in spatial_dims:
        auc = compute_wp_auc(
            df[df["spatial_dim"] == spatial_dim],
            methods,
            param_cols,
            base_wp,
            num_modes,
            metric_col,
        )
        auc_by_spatial_dim[spatial_dim] = auc
        format_wide_latex_table(
            auc,
            methods,
            param_cols,
            f"{table_prefix}_d{spatial_dim}",
            caption_for(metric_label, spatial_dim),
            out_dir / f"auc_d{spatial_dim}.tex",
        )

    format_unified_latex_table(
        auc_by_spatial_dim,
        methods,
        param_cols,
        f"{table_prefix}_unified",
        unified_caption_for(metric_label),
        unified_output_path or out_dir / "auc_unified.tex",
    )


if __name__ == "__main__":
    fire.Fire(main)
