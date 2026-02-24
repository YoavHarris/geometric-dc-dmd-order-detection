#!/usr/bin/env python3
"""
Generate AUC tables for paper: two wide tables (DC and No-DC).

Usage:
    python auc_tables.py dc_combined_results.csv no_dc_combined_results.csv

Outputs:
    figures/scans/outputs/auc_dc.tex
    figures/scans/outputs/auc_no_dc.tex
"""

from __future__ import annotations

import sys
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid


# Parameters for No-DC table
PARAM_COLS_NO_DC = ["snr_db", "freq_sep", "eig_mag", "top_amplitude"]

# Parameters for DC table (includes extra sweeps)
# User requested to omit num_modes from this table
PARAM_COLS_DC = PARAM_COLS_NO_DC + ["max_rank"]

PARAM_LATEX = {
    "snr_db": "SNR",
    "freq_sep": "$\\Delta\\theta$",
    "eig_mag": "$r$",
    "top_amplitude": "$\\kappa_b$",
    "max_rank": "$M$",
    "num_modes": "$m_{\\text{sweep}}$",
}

# Methods for DC table
DC_METHODS = [
    "BIC",
    "GAP",
    "STC",
    "ESR-Energy",
    "NestedDMD",
    "FixedEigenvalueKVFit",
]

# Methods for No-DC table (only those that work without delays)
NO_DC_METHODS = [
    "BIC",
    "GAP",
    "ESR-Energy",
    "ExactModeNorm",
    "EigenvalueMagnitude",
]

# Methods to always exclude
EXCLUDE_METHODS = {"NestedDMD+ESR", "FixedEigenvalueKVFit+ESR"}


def compute_normalized_auc(
    df: pd.DataFrame, methods_to_include: list[str], param_cols: list[str]
) -> dict[tuple[int, str, str], float]:
    """
    Compute normalized AUC of order_hit_prob for each (num_modes, method, param).

    Args:
        df: DataFrame with experimental results
        methods_to_include: List of methods to compute AUCs for
        param_cols: List of parameters to compute AUC for

    Returns:
        dict keyed by (num_modes, method, param_col) -> auc_norm
    """
    if "order_hit_prob" not in df.columns:
        raise ValueError("Expected column 'order_hit_prob' in results CSV")

    auc_dict: dict[tuple[int, str, str], float] = {}

    for param in param_cols:
        if param not in df.columns:
            continue

        # Standard logic for parameters (snr, max_rank, etc)
        # These are calculated PER true num_modes (m)
        for m in sorted(df["num_modes"].unique()):
            df_m = df[df["num_modes"] == m]

            for method in methods_to_include:
                if method in EXCLUDE_METHODS:
                    continue

                df_mm = df_m[df_m["method"] == method].copy()
                if df_mm.empty:
                    continue

                # Average over replicates at each parameter value
                grouped = (
                    df_mm.groupby(param)["order_hit_prob"]
                    .mean()
                    .reset_index()
                    .sort_values(param)
                )

                if len(grouped) < 2:
                    continue

                x = grouped[param].to_numpy()
                y = grouped["order_hit_prob"].to_numpy()

                span = float(x[-1] - x[0])
                if span <= 0:
                    continue

                auc = float(trapezoid(y, x))
                auc_norm = auc / span

                auc_dict[(m, method, param)] = auc_norm

    return auc_dict


def format_wide_latex_table(
    auc_dict: dict[tuple[int, str, str], float],
    methods: list[str],
    param_cols: list[str],
    table_label: str,
    caption: str,
    output_path: Path,
) -> None:
    """
    Generate one wide table spanning all m values.

    Structure:
    - Rows: methods
    - Columns: grouped by m, each group has len(param_cols) params
    """
    lines: list[str] = []

    # Get all m values present in the data
    all_ms = sorted({key[0] for key in auc_dict.keys()})

    # Filter to only include methods that have data
    methods_with_data = [
        m for m in methods if any(key[1] == m for key in auc_dict.keys())
    ]

    if not methods_with_data or not all_ms:
        print(f"Warning: No data for {table_label}", file=sys.stderr)
        return

    # Table header
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3pt}")

    # Tabular environment: 1 col for method names + len(param_cols)*len(all_ms) for data
    num_data_cols = len(param_cols) * len(all_ms)
    col_spec = "l" + "c" * num_data_cols
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

    # Top rule
    lines.append("\\hline")

    # First header row: m groups
    header1_parts = [""]  # Empty for method column
    for m in all_ms:
        header1_parts.append(f"\\multicolumn{{{len(param_cols)}}}{{c}}{{$m = {m}$}}")
    lines.append(" & ".join(header1_parts) + " \\\\")

    # Second header row: parameter labels (repeated for each m)
    header2_parts = ["Method"]
    param_headers = [PARAM_LATEX[p] for p in param_cols]
    for _ in all_ms:
        header2_parts.extend(param_headers)
    lines.append(" & ".join(header2_parts) + " \\\\")
    lines.append("\\hline")

    # Data rows
    for method in methods_with_data:
        row_parts = [method]

        for m in all_ms:
            for param in param_cols:
                key = (m, method, param)
                if key not in auc_dict:
                    row_parts.append("--")
                    continue

                v = auc_dict[key]

                # Find best value for this (m, param) block
                best_v = max(
                    auc_dict.get((m, mtd, param), -np.inf) for mtd in methods_with_data
                )
                best_v_str = f"{best_v:.3f}"

                v_str = f"{v:.3f}"
                if v_str == best_v_str and best_v > 0:
                    v_str = f"\\textbf{{{v_str}}}"

                row_parts.append(v_str)

        lines.append(" & ".join(row_parts) + " \\\\")

    # Bottom rule
    lines.append("\\hline")
    lines.append("\\end{tabular}")

    # Caption and label
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{tab:{table_label}}}")
    lines.append("\\end{table*}")

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated: {output_path}")


def main(dc_csv: str, no_dc_csv: str) -> None:
    """
    Generate both AUC tables.

    Args:
        dc_csv: Path to dc_combined_results.csv
        no_dc_csv: Path to no_dc_combined_results.csv
    """
    dc_path = Path(dc_csv)
    no_dc_path = Path(no_dc_csv)

    if not dc_path.is_file():
        print(f"Error: file not found: {dc_path}", file=sys.stderr)
        sys.exit(1)
    if not no_dc_path.is_file():
        print(f"Error: file not found: {no_dc_path}", file=sys.stderr)
        sys.exit(1)

    # Load data
    df_dc = pd.read_csv(dc_path)
    df_no_dc = pd.read_csv(no_dc_path)

    # Filter to Gaussian noise if available
    if "noise_mode" in df_dc.columns:
        df_dc = df_dc[df_dc["noise_mode"] == "gaussian"].copy()
    if "noise_mode" in df_no_dc.columns:
        df_no_dc = df_no_dc[df_no_dc["noise_mode"] == "gaussian"].copy()

    # Compute AUCs
    auc_dc = compute_normalized_auc(df_dc, DC_METHODS, PARAM_COLS_DC)
    auc_no_dc = compute_normalized_auc(df_no_dc, NO_DC_METHODS, PARAM_COLS_NO_DC)

    # Output directory
    output_dir = Path("figures/scans/outputs")

    # Generate DC table
    dc_caption = (
        "Normalized AUC for DC experiments "
        "(delay-embedded) across one-dimensional parameter sweeps. "
        "Columns grouped by number of true modes ($m$) contain AUC values for: "
        "SNR, phase separation ($\\Delta\\theta$), damping coefficient ($r$), "
        "amplitude ratio ($\\kappa_b = b_{\\max}/b_{\\min}$), and "
        "order-overestimation ($M$)."
    )
    format_wide_latex_table(
        auc_dc,
        DC_METHODS,
        PARAM_COLS_DC,
        "auc_dc",
        dc_caption,
        output_dir / "auc_dc.tex",
    )

    # Generate No-DC table
    no_dc_caption = (
        "Normalized AUC for No-DC experiments "
        "($L=1$, no delays) across one-dimensional parameter sweeps. "
        "Columns grouped by number of true modes ($m$) contain AUC values for: "
        "SNR, phase separation ($\\Delta\\theta$), damping coefficient ($r$), "
        "and amplitude ratio ($\\kappa_b = b_{\\max}/b_{\\min}$)."
    )
    format_wide_latex_table(
        auc_no_dc,
        NO_DC_METHODS,
        PARAM_COLS_NO_DC,
        "auc_no_dc",
        no_dc_caption,
        output_dir / "auc_no_dc.tex",
    )


if __name__ == "__main__":
    fire.Fire(main)
