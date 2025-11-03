"""
plotting.py
===========

Utilities to create:
  • working-point table   (baseline_table_fig)
  • 1-D sweep plots   (make_sweep_plots)
  • 2-D heat-maps     (make_heatmaps)


All heavy lifting is done once; no external patching needed.
"""

from __future__ import annotations

import itertools
from numbers import Number
from typing import Dict, List, Mapping, Sequence, Tuple, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


HIGHER_IS_BETTER = {
    # § classification / detection
    "order_hit_prob": True,
    "accuracy": True,
    "precision": True,
    "recall": True,
    "f1": True,
    "f1_score": True,
    # § error metrics
    "abs_diff_mean": False,
    "abs_diff_std": False,
    "order_diff_mean": False,
    "order_diff_std": False,
}


# ────────────────────────────────────────────────────────────────────────
# Typed style configuration
# ────────────────────────────────────────────────────────────────────────
class StyleConf(TypedDict, total=False):
    markers: bool  # per-method markers on/off
    line_width: float
    font_scale: float
    sweep_height: float  # inches
    sweep_aspect: float  # width / height
    auc_table_fontsize: float  # default 8
    auc_table_colwidth: float  # figure-fraction; default 0.8
    auc_table_gap: float  # extra bottom margin; default 0.15
    auc_table_rowheight: 0.25  # inches per method row
    auc_table_pad: 0.15  # inches gap under table


# ────────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────────
def _closest_numeric(series: pd.Series, target: float) -> float:
    uniq = series.unique().astype(float)
    return float(uniq[np.argmin(np.abs(uniq - target))])


def _freeze(
    df: pd.DataFrame,
    baseline: Mapping[str, object],
    drop_cols: Sequence[str],
    *,
    closest: bool,
) -> pd.DataFrame:
    """Return view where every non-dropped column matches the baseline."""
    frozen = df
    for col, val in baseline.items():
        if col in drop_cols or col not in frozen.columns:
            continue
        if val in frozen[col].unique():
            frozen = frozen[frozen[col] == val]
        elif closest and isinstance(val, Number):
            nearest = _closest_numeric(frozen[col], float(val))
            print(f"[freeze] {col}: {val} → closest {nearest}")
            frozen = frozen[frozen[col] == nearest]
        else:
            frozen = frozen[frozen[col] == val]
    return frozen


def _auc(grp: pd.DataFrame, x: str, y: str, normalize: bool = False) -> float | None:
    if grp.empty:
        return None

    g = grp.sort_values(x)
    area = float(np.trapz(g[y].to_numpy(float), g[x].to_numpy(float)))

    if not normalize:
        return area

    span = g[x].iloc[-1] - g[x].iloc[0]
    return None if span == 0 else area / span


def _get_auc_dict(
    sub: pd.DataFrame, xcol: str, metric: str, normalize: bool = False
) -> Dict[tuple, float]:
    auc: Dict[Tuple[str, str], float] = {}
    for (noise, method), grp in sub.groupby(["noise_mode", "method"]):
        val = _auc(grp, xcol, metric, normalize=normalize)
        if val is not None:
            auc[(noise, method)] = val
    return auc


def _get_auc_table(
    auc: Dict[Tuple[str, str], float],
    methods: List[str],
    noises: List[str],
    fmt: str = ".3f",
) -> Tuple[List[List[str]], List[str], List[str]]:
    """
    Convert the {(noise, method): value} dict into lists ready for
    `ax.table(..)`:
      • cell_text  – rows=methods, cols=noises
      • row_labels – methods
      • col_labels – noises
    """
    cell_text: List[List[str]] = []
    for m in methods:
        row = []
        for n in noises:
            v = auc.get((n, m))
            row.append(f"{v:{fmt}}" if v is not None else "–")
        cell_text.append(row)
    return cell_text, methods, noises


def _add_auc_table(
    grid: sns.axisgrid.FacetGrid,
    auc: Dict[Tuple[str, str], float],
    methods: List[str],
    noises: List[str],
    style: StyleConf,
    high_is_good: bool = True,
) -> None:
    """
    Enlarge `grid.fig`, keep subplot sizes unchanged, and insert
    a centred AUC table underneath.  Also bold the max AUC in each
    noise-mode column.
    """
    # ----- style knobs (inches) --------------------------------------
    row_h = float(style.get("auc_table_rowheight", 0.25))
    pad_in = float(style.get("auc_table_pad", 0.05))  # below table
    gap_in = float(style.get("auc_table_gap", 0.15))  # above table
    col_w = float(style.get("auc_table_colwidth", 0.8))
    font_sz = float(style.get("auc_table_fontsize", 8))

    # ----- convert dict → cell text ---------------------------------
    cell_text, row_lbls, col_lbls = _get_auc_table(auc, methods, noises)

    # ----- figure resize & subplot shift ----------------------------
    fig_w, fig_h = grid.fig.get_size_inches()
    table_h_in = row_h * len(methods)
    extra_h = pad_in + gap_in + table_h_in

    grid.fig.set_size_inches(fig_w, fig_h + extra_h, forward=True)
    new_fig_h = fig_h + extra_h
    bottom_margin = extra_h / new_fig_h
    grid.fig.subplots_adjust(bottom=bottom_margin)

    # ----- table Axes ------------------------------------------------
    ax_tbl = grid.fig.add_axes(
        [
            (1 - col_w) / 2,  # centred
            pad_in / new_fig_h,  # bottom offset
            col_w,
            table_h_in / new_fig_h,
        ],
        frameon=False,
    )
    ax_tbl.axis("off")
    ax_tbl.set_title("AUC (normalized by support)")
    tbl = ax_tbl.table(
        cellText=cell_text,
        rowLabels=row_lbls,
        colLabels=col_lbls,
        loc="center",
        cellLoc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_sz)

    # ----- bold max in each column ----------------------------------
    tie_eps = float(style.get("auc_table_tie_eps", 1e-4))  # tolerance for ties
    num_rows = len(methods)
    num_cols = len(noises)
    for j in range(num_cols):  # loop over noise columns
        vals = []
        for i in range(num_rows):
            val_str = cell_text[i][j]
            if val_str != "–":
                vals.append((i, float(val_str)))

        if not vals:
            continue

        best_val = (max if high_is_good else min)(v for _, v in vals)  # pick max or min
        for row_idx, v in vals:
            if abs(v - best_val) <= tie_eps:
                cell_key = (row_idx + 1, j)  # +1 → skip header row
                if cell_key in tbl._cells:
                    tbl._cells[cell_key].get_text().set_weight("bold")

    highlight_methods = {"ModeMatDMD", "ModeMatFeatures"}

    for i, m in enumerate(methods):
        if m in highlight_methods:
            row = i + 1  # +1 → skip header row
            for col in range(len(noises) + 1):  # +1 for row-label column
                cell = tbl._cells.get((row, col))
                if cell:
                    cell.set_facecolor("#fff9c4")  # light yellow


# ────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────
# ─────────────────── baseline table figure ────────────────────────────
def baseline_table_fig(baseline: Mapping[str, object]) -> Figure:
    """
    Return a Figure containing a 2-column table of the baseline values.
    """
    rows = [[k, v] for k, v in baseline.items()]
    fig, ax = plt.subplots(figsize=(5, 0.35 * len(rows) + 1))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=["Parameter", "Value"],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.2)

    ax.set_title("Working Point (Baseline)", pad=6, fontsize=12)
    return fig


def make_sweep_plots(
    df: pd.DataFrame,
    metric: str,
    baseline: Mapping[str, object],
    ci: bool,
    *,
    vary_cols: Sequence[str] | None = None,
    report_auc: bool = False,
    auc_metrics: Sequence[str] = ("order_hit_prob",),
    closest_match: bool = False,
    style: StyleConf | None = None,
    axis_scale: Mapping[str, str] | None = None,
) -> List[Figure]:
    """
    Create one Figure per swept parameter; all noise-modes sit side-by-side,
    and ONE legend is placed on the figure’s right.
    """
    style = style or StyleConf()
    axis_scale = axis_scale or {}

    sns.set_theme(font_scale=float(style.get("font_scale", 0.9)))

    height = float(style.get("sweep_height", 5))
    aspect = float(style.get("sweep_aspect", 1.6))
    markers = bool(style.get("markers", False))
    lw = float(style.get("line_width", 1.5))

    palette_map: dict = style.get("palette", {})
    dash_map: dict = style.get("dashes", {})
    default_col = palette_map.get("default", None)
    default_dash = dash_map.get("default", "--")

    vary_cols = (
        list(vary_cols)
        if vary_cols is not None
        else [c for c in baseline if c != "noise_mode"]
    )

    figs: List[Figure] = []

    for xcol in vary_cols:
        sub = _freeze(df, baseline, [xcol], closest=closest_match)

        # nearest displayed baseline x-value
        target = float(baseline[xcol])
        vals = sub[xcol].unique().astype(float)
        x_baseline = float(vals[np.argmin(np.abs(vals - target))])

        # -------- optional AUC per (noise, method) -------------------
        auc = (
            _get_auc_dict(sub, xcol, metric, normalize=True)
            if report_auc and metric in auc_metrics
            else {}
        )

        methods = list(sub["method"].unique())

        # palette  -----------------------------------------------------------
        color_cycle = itertools.cycle(sns.color_palette("dark", 8))
        palette = {}
        for m in methods:
            palette[m] = palette_map.get(m, default_col or next(color_cycle))

        # dashes  ------------------------------------------------------------
        dashes = [dash_map.get(m, default_dash) for m in methods]

        # -------- seaborn grid (legend suppressed) -------------------
        errorbar = (
            "sd"
            if ci
            and metric
            in {"order_diff_mean", "order_diff_std", "abs_diff_mean", "abs_diff_std"}
            else None
        )

        g = sns.relplot(
            data=sub,
            x=xcol,
            y=metric,
            hue="method",
            style="method",
            col="noise_mode",
            kind="line",
            height=height,
            aspect=aspect,
            errorbar=errorbar,
            markers=markers,
            linewidth=lw,
            legend=False,
            facet_kws={"sharey": False},
            palette=palette,
            dashes=dashes,
        )

        # -------- per-subplot tweaks (baseline + log scales) ---------
        for ax in g.axes.flatten():
            # baseline line
            if style.get("show_baseline", True):
                ax.axvline(
                    x_baseline,
                    color="orange",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.8,
                    zorder=10,
                )

            # axis scales
            if axis_scale.get(xcol, "lin") == "log":
                ax.set_xscale("log")
            if axis_scale.get(metric, "lin") == "log":
                ax.set_yscale("log")

        plotted = [
            m  # keep the order we want
            for m in methods  # full hue_order list
            if sub.loc[sub["method"] == m, metric].notna().any()
        ]

        # -------- single legend for the figure --------------------------------
        first_ax = g.axes.flatten()[0]
        lines = first_ax.get_lines()

        handles, labels = [], []
        for ln, method in zip(lines, plotted):
            ln.set_label(method)
            handles.append(ln)
            labels.append(method)

        # scrub any auto legends inside sub-plots
        for ax in g.axes.flatten():
            if ax.legend_:
                ax.legend_.remove()

        # place the single legend to the right of the whole grid
        g.fig.legend(
            handles,
            labels,
            title="Method",
            loc="center left",
            bbox_to_anchor=(1, 0.5),  # bump up against the figure edge
            borderaxespad=0.0,
            frameon=False,
        )

        if auc:
            is_high_good = HIGHER_IS_BETTER.get(metric, True)

            _add_auc_table(
                g,
                auc,
                methods,
                noises=list(sub["noise_mode"].unique()),
                style=style,
                high_is_good=is_high_good,  # ← new argument
            )

        g.fig.suptitle(f"{metric} vs {xcol}", y=1.02, fontsize=13)
        figs.append(g.fig)

    return figs


def make_heatmaps(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    metric: str,
    baseline: Mapping[str, object],
    *,
    cmap: str = "viridis",
    closest_match: bool = False,
    style: StyleConf | None = None,
    axis_scale: Mapping[str, str] | None = None,
) -> List[Figure]:
    """
    Return list[Figure] – one heat-map per (noise_mode, method).
    """
    style = style or StyleConf()
    axis_scale = axis_scale or {}

    sns.set_theme(font_scale=float(style.get("font_scale", 0.9)))

    height = float(style.get("sweep_height", 5))
    aspect = float(style.get("sweep_aspect", 1.2))
    width = height * aspect

    sub = _freeze(df, baseline, [xcol, ycol], closest=closest_match)
    figs: List[Figure] = []

    for (noise, method), grp in sub.groupby(["noise_mode", "method"]):
        pivot = (
            grp.pivot(index=ycol, columns=xcol, values=metric)
            .sort_index(axis=0)
            .sort_index(axis=1)
        )

        fig, ax = plt.subplots(figsize=(width, height))
        sns.heatmap(pivot, ax=ax, cmap=cmap, cbar_kws={"label": metric}, annot=False)
        ax.set_title(f"{metric} — {method} — {noise}")
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)

        if axis_scale.get(xcol, "lin") == "log":
            ax.set_xscale("log")
        if axis_scale.get(ycol, "lin") == "log":
            ax.set_yscale("log")

        figs.append(fig)

    return figs
