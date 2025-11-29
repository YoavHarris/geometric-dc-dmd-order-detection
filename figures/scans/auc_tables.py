#!/usr/bin/env python3
"""
Compute normalized AUCs from results.csv and print LaTeX tables.

Usage:
    python auc_tables.py results.csv > auc_tables.tex
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import fire
import numpy as np
import pandas as pd


# Parameters we want to summarize
PARAM_COLS = ["snr_db", "freq_sep", "eig_mag", "top_amplitude"]
PARAM_LATEX = {
    "snr_db": "SNR",
    "freq_sep": "\\Delta f",
    "eig_mag": "r",
    "top_amplitude": "\\kappa_b",
}

# Desired method order in the tables
METHOD_ORDER = [
    "BIC",
    "GAP",
    "STC",
    "ESL-Norm",
    "NestedDMD",
    "FixedEigenvalueBVFit",
    "NestedDMD+ESL",
    "FixedEigenvalueBVFit+ESL",
]

# Methods we never want to show
EXCLUDE_METHODS = {"AIC", "ExactModeNorm"}


def compute_normalized_auc(df: pd.DataFrame) -> Dict[Tuple[int, str, str], float]:
    """
    Compute normalized AUC of order_hit_prob for each (num_modes, method, param).

    Returns:
        dict keyed by (num_modes, method, param_col) -> auc_norm
    """
    if "order_hit_prob" not in df.columns:
        raise ValueError("Expected column 'order_hit_prob' in results CSV")

    auc_dict: Dict[Tuple[int, str, str], float] = {}

    for param in PARAM_COLS:
        if param not in df.columns:
            continue

        for m in sorted(df["num_modes"].unique()):
            df_m = df[df["num_modes"] == m]

            for method in sorted(df_m["method"].unique()):
                if method in EXCLUDE_METHODS:
                    continue

                df_mm = df_m[df_m["method"] == method].copy()
                if df_mm.empty:
                    continue

                # average over replicates at each parameter value
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

                auc = float(np.trapz(y, x))
                auc_norm = auc / span

                auc_dict[(m, method, param)] = auc_norm

    return auc_dict


def format_latex_tables(
    auc_dict: Dict[Tuple[int, str, str], float], df: pd.DataFrame
) -> str:
    """
    Build LaTeX tables (one per num_modes) as a single string.
    """
    lines: list[str] = []

    all_ms = sorted(df["num_modes"].unique())
    # Only keep m that actually appear in auc_dict
    all_ms = [m for m in all_ms if any(key[0] == m for key in auc_dict.keys())]

    for m in all_ms:
        # Collect methods present for this m and in desired order
        methods_for_m = sorted(
            {
                method
                for (m_key, method, _param) in auc_dict.keys()
                if m_key == m and method not in EXCLUDE_METHODS
            }
        )
        methods_for_m = [mtd for mtd in METHOD_ORDER if mtd in methods_for_m]

        if not methods_for_m:
            continue

        # Determine best AUC per parameter (for boldface)
        best_per_param: Dict[str, float] = {}
        for param in PARAM_COLS:
            vals = [
                auc_dict[(m, method, param)]
                for method in methods_for_m
                if (m, method, param) in auc_dict
            ]
            if vals:
                best_per_param[param] = max(vals)

        # Start table
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append("\\setlength{\\tabcolsep}{4pt}")
        lines.append("\\begin{tabular}{lcccc}")
        lines.append("\\toprule")
        lines.append(f"& \\multicolumn{{4}}{{c}}{{$m = {m}$}} \\\\")
        lines.append("\\cmidrule(lr){2-5}")
        header_cols = [PARAM_LATEX[p] for p in PARAM_COLS]
        lines.append("Method & " + " & ".join(header_cols) + " \\\\")
        lines.append("\\midrule")

        for method in methods_for_m:
            row_vals = []
            for param in PARAM_COLS:
                key = (m, method, param)
                if key not in auc_dict:
                    row_vals.append("--")
                    continue
                v = auc_dict[key]
                v_str = f"{v:.3f}"
                if param in best_per_param and np.isclose(v, best_per_param[param]):
                    v_str = f"\\textbf{{{v_str}}}"
                row_vals.append(v_str)

            line = method + " & " + " & ".join(row_vals) + " \\\\"
            lines.append(line)

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append(
            f"\\caption{{Normalized AUC for one-dimensional sweeps with $m={m}$. Columns correspond to SNR, frequency separation $\\Delta f$, damping coefficient $r$, and amplitude ratio $\\kappa_b = b_{{\\max}}/b_{{\\min}}$.}}"
        )
        lines.append(f"\\label{{{{tab:auc_m{m}}}}}")
        lines.append("\\end{table}")
        lines.append("")  # blank line between tables

    return "\n".join(lines)


def main(csv_path: str) -> None:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Restrict to Gaussian noise if available
    if "noise_mode" in df.columns:
        df = df[df["noise_mode"] == "gaussian"].copy()

    auc_dict = compute_normalized_auc(df)
    latex = format_latex_tables(auc_dict, df)
    print(latex)


if __name__ == "__main__":
    fire.Fire(main)
