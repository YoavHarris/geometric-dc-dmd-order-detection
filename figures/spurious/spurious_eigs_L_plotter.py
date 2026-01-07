"""
Comprehensive plotter for spurious eigenvalue experiment results.

Generates multiple figures:
1. CDF per L - Shows how eigenvalue magnitude distributions change with L
2. rho_min vs L - Minimum eigenvalue magnitude trends (mixture only)
3. Combined CDFs - Overlay of mixture vs noise for both all magnitudes and rho_min
4. CDF + rho_min combined - Single-column figure with CDF and rho_min side-by-side
5. mu_L and nu_L vs L - Spectral norms with theoretical reference

Usage:
    python spurious_eigs_L_plotter.py config.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fire
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from figures.common import plotting_common


# =============================================================================
# Data loading and validation
# =============================================================================


def validate_and_load_data(config: dict[str, Any], project_root: Path) -> pd.DataFrame:
    """Load and validate CSV data."""
    # Get paths from config
    try:
        plot_cfg = config["plotting"]
        csv_rel_path = plot_cfg["data_csv_path"]
    except KeyError as e:
        raise KeyError(f"Config missing required key: {e}")
    
    csv_path = plotting_common.resolve_path(project_root, csv_rel_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    # Load data
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")
    
    # Validate required columns
    required_cols = ["L", "eigenvalue_magnitude"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    
    # Check for mixture/noise comparison data
    if "setting" in df.columns and "trial_id" in df.columns:
        unique_settings = set(df["setting"].unique())
        print(f"Settings found: {sorted(unique_settings)}")
        
        expected = {"mixture_spurious", "noise_only"}
        if expected.issubset(unique_settings):
            print("[OK] Data includes mixture vs noise comparison")
            df_filtered = df[df["setting"].isin(expected)].copy()
            
            # Print setting counts
            setting_counts = df_filtered["setting"].value_counts()
            print("Rows per setting:")
            for setting, count in setting_counts.items():
                print(f"  {setting}: {count}")
        else:
            print("Warning: Data does not include mixture vs noise comparison")
            df_filtered = df.copy()
    else:
        print("Warning: No 'setting' column - assuming single-mode data (mixture only)")
        df_filtered = df.copy()
    
    print(f"Unique L values ({len(df_filtered['L'].unique())}): {sorted(df_filtered['L'].unique())}")
    print()
    
    return df_filtered


# =============================================================================
# Helpers
# =============================================================================


def _resolve_L_values(config: dict[str, Any], available_L: list[int]) -> list[int]:
    """
    Resolve which L values to plot based on config.
    Returns sorted list of L values.
    """
    requested = config["plotting"].get("L_values_to_plot")
    if not requested:
        return sorted(available_L)
        
    available_set = set(available_L)
    valid = sorted([L for L in requested if L in available_set])
    
    if len(valid) != len(requested):
        print(f"Warning: Requested L values missing from data: {set(requested) - available_set}")
        
    if not valid:
        print("Warning: No valid L values found in config. Using all available.")
        return sorted(available_L)
        
    return valid


# =============================================================================
# Figure 1: CDF per L
# =============================================================================


def plot_cdf_per_L(
    df: pd.DataFrame, 
    config: dict[str, Any], 
    output_path: Path,
    aggregation_method: str = 'pooled'
) -> None:
    """
    Plot empirical CDFs of eigenvalue magnitudes for each L.
    Shows how distributions evolve with embedding length.
    
    Args:
        df: DataFrame with eigenvalue data
        config: Configuration dict
        output_path: Path to save the figure
        aggregation_method: How to compute CDFs
            - 'pooled': Pool all eigenvalues across trials, compute one CDF (default)
            - 'trial_averaged': Compute CDF per trial, then average the CDFs
    """
    if aggregation_method not in ['pooled', 'trial_averaged']:
        raise ValueError(f"aggregation_method must be 'pooled' or 'trial_averaged', got: {aggregation_method}")
    
    L_values = _resolve_L_values(config, df["L"].unique())
    
    try:
        xmin, xmax = config["plotting"]["xlim"]
        cmap_name = config["plotting"]["colors"]["colormap"]
    except KeyError as e:
        raise KeyError(f"Config missing plotting parameter: {e}")
    
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.0, 1.0, len(L_values)))
    
    fig, ax = plt.subplots()
    
    for L, color in zip(L_values, colors):
        if aggregation_method == 'pooled':
            # Pool all eigenvalues from all trials for this L
            vals = df.loc[df["L"] == L, "eigenvalue_magnitude"].to_numpy()
            xs = np.sort(vals)
            n = xs.size
            ys = np.arange(1, n + 1) / n
            
            # Extend to full axis range
            xs_plot = np.concatenate(([xmin], xs, [xmax]))
            ys_plot = np.concatenate(([0.0], ys, [1.0]))
            
        else:  # trial_averaged
            # Compute CDF per trial, then average
            if "trial_id" not in df.columns:
                raise ValueError("aggregation_method='trial_averaged' requires 'trial_id' column")
            
            df_L = df[df["L"] == L]
            trial_ids = df_L["trial_id"].unique()
            
            # Define common evaluation points
            all_vals = df_L["eigenvalue_magnitude"].to_numpy()
            x_eval = np.linspace(all_vals.min(), all_vals.max(), 500)
            
            # Compute CDF for each trial and average
            cdfs = []
            for trial_id in trial_ids:
                trial_vals = df_L[df_L["trial_id"] == trial_id]["eigenvalue_magnitude"].to_numpy()
                trial_vals_sorted = np.sort(trial_vals)
                # Interpolate CDF at common evaluation points
                cdf_interp = np.searchsorted(trial_vals_sorted, x_eval, side='right') / len(trial_vals_sorted)
                cdfs.append(cdf_interp)
            
            # Average CDFs across trials
            avg_cdf = np.mean(cdfs, axis=0)
            
            # Extend to full axis range
            xs_plot = np.concatenate(([xmin], x_eval, [xmax]))
            ys_plot = np.concatenate(([0.0], avg_cdf, [1.0]))
        
        ax.step(xs_plot, ys_plot, where="post", label=f"L={L}", color=color, alpha=0.8)
    
    # Reference line at |lambda| = 1
    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.7, label="Unit circle")
    
    ax.set_xlabel(r"Eigenvalue magnitude $|\lambda|$")
    ax.set_ylabel("Cumulative probability")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([0.0, 1.2])
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.9)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")



# =============================================================================
# Figure 2: rho_min vs L
# =============================================================================


def plot_rho_min_vs_L(df: pd.DataFrame, config: dict[str, Any], output_path: Path) -> None:
    """
    Plot minimum eigenvalue magnitude vs L.
    Shows median with percentile band.
    Requires 'trial_id' column for computing per-trial minima.
    """
    if "trial_id" not in df.columns:
        print("Warning: Skipping rho_min vs L plot - 'trial_id' column not found")
        return
    
    # Filter to mixture only if available
    if "setting" in df.columns:
        df_plot = df[df["setting"] == "mixture_spurious"].copy()
    else:
        df_plot = df.copy()
    
    # Compute rho_min per (L, trial_id)
    rho_min = df_plot.groupby(["L", "trial_id"])["eigenvalue_magnitude"].min().reset_index()
    rho_min.rename(columns={"eigenvalue_magnitude": "rho_min"}, inplace=True)
    
    # Get percentiles from config
    try:
        p_low, p_high = config["plotting"]["confidence_percentiles"]
    except KeyError:
        p_low, p_high = 5, 95  # Default
    
    # Aggregate per L
    L_values = []
    medians = []
    means = []
    lowers = []
    uppers = []
    
    for L in sorted(rho_min["L"].unique()):
        vals = rho_min[rho_min["L"] == L]["rho_min"]
        L_values.append(L)
        medians.append(vals.median())
        means.append(vals.mean())
        lowers.append(np.percentile(vals, p_low))
        uppers.append(np.percentile(vals, p_high))
    
    # Plot
    fig, ax = plt.subplots()
    
    ax.plot(L_values, medians, marker="o", linestyle="-", label=r"Median($\rho_{\min}$)")
    ax.plot(L_values, means, marker="", linestyle="--", linewidth=1.0, label=r"Mean($\rho_{\min}$)")
    
    # Format percentiles
    p_low_str = f"{p_low:.1f}".rstrip('0').rstrip('.') if p_low % 1 else f"{int(p_low)}"
    p_high_str = f"{p_high:.1f}".rstrip('0').rstrip('.') if p_high % 1 else f"{int(p_high)}"
    ax.fill_between(L_values, lowers, uppers, alpha=0.3, 
                     label=f"{p_low_str}–{p_high_str}% band")
    
    ax.set_xlabel(r"Embedding length $L$")
    ax.set_ylabel(r"$\rho_{\min}$")
    ax.set_xlim(left=1)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# =============================================================================
# Figure 4: Combined CDFs (mixture vs noise)
# =============================================================================


def plot_combined_cdf_comparison(df: pd.DataFrame, config: dict[str, Any], output_path: Path) -> None:
    """
    Plot both rho (all magnitudes) and rho_min CDFs comparing mixture vs noise.
    Left: CDF of all eigenvalue magnitudes
    Right: CDF of minimum eigenvalue per trial
    Requires 'setting' and 'trial_id' columns.
    """
    if "setting" not in df.columns or "trial_id" not in df.columns:
        print("Warning: Skipping combined CDF comparison - requires 'setting' and 'trial_id' columns")
        return
    
    # Prepare rho data (all eigenvalue magnitudes)
    rho_mix = df[df["setting"] == "mixture_spurious"]["eigenvalue_magnitude"].to_numpy()
    rho_noise = df[df["setting"] == "noise_only"]["eigenvalue_magnitude"].to_numpy()
    ks_rho, _ = stats.ks_2samp(rho_mix, rho_noise)
    
    # Prepare rho_min data
    rho_min = df.groupby(["L", "trial_id", "setting"])["eigenvalue_magnitude"].min().reset_index()
    rho_min.rename(columns={"eigenvalue_magnitude": "rho_min"}, inplace=True)
    rho_min_mix = rho_min[rho_min["setting"] == "mixture_spurious"]["rho_min"].to_numpy()
    rho_min_noise = rho_min[rho_min["setting"] == "noise_only"]["rho_min"].to_numpy()
    ks_rho_min, _ = stats.ks_2samp(rho_min_mix, rho_min_noise)
    
    # Create side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
    
    # Left panel: rho CDF (all magnitudes)
    xs_mix = np.sort(rho_mix)
    ys_mix = np.arange(1, len(xs_mix) + 1) / len(xs_mix)
    line_mix, = ax1.plot(xs_mix, ys_mix, linestyle="-", linewidth=2)
    
    xs_noise = np.sort(rho_noise)
    ys_noise = np.arange(1, len(xs_noise) + 1) / len(xs_noise)
    line_noise, = ax1.plot(xs_noise, ys_noise, linestyle="--", linewidth=2)
    
    ref_line = ax1.axvline(x=1.0, color="red", linestyle=":", alpha=0.5, linewidth=1.5)
    
    ax1.text(0.02, 0.98, f"KS = {ks_rho:.4f}", 
            transform=ax1.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    ax1.set_xlabel(r"Eigenvalue magnitude")
    ax1.set_ylabel("CDF")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, linestyle=":")
    
    # Right panel: rho_min CDF
    xs_min_mix = np.sort(rho_min_mix)
    ys_min_mix = np.arange(1, len(xs_min_mix) + 1) / len(xs_min_mix)
    ax2.plot(xs_min_mix, ys_min_mix, linestyle="-", linewidth=2)
    
    xs_min_noise = np.sort(rho_min_noise)
    ys_min_noise = np.arange(1, len(xs_min_noise) + 1) / len(xs_min_noise)
    ax2.plot(xs_min_noise, ys_min_noise, linestyle="--", linewidth=2)
    
    ax2.axvline(x=1.0, color="red", linestyle=":", alpha=0.5, linewidth=1.5)
    
    ax2.text(0.02, 0.98, f"KS = {ks_rho_min:.4f}", 
            transform=ax2.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    ax2.set_xlabel(r"Eigenvalue magnitude")
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, linestyle=":")
    
    # Shared legend below figure
    fig.legend(handles=[line_mix, line_noise, ref_line], 
               labels=["Mixture (spurious)", "Noise-only", r"Unit circle ($|\lambda|=1$)"],
               loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, framealpha=0.95)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# =============================================================================
# Figure 5: CDF + rho_min combined (single-column layout)
# =============================================================================


def plot_cdf_and_rho_min_combined(df: pd.DataFrame, config: dict[str, Any], output_path: Path) -> None:
    """
    Plot per-L CDFs on left and rho_min vs L on right.
    Designed for single-column layout.
    Requires 'trial_id' column.
    """
    if "trial_id" not in df.columns:
        print("Warning: Skipping combined CDF+rho_min plot - requires 'trial_id' column")
        return
    
    
    # Create side-by-side subplots with double-column style
    # Use double-column mplstyle for this wider figure
    import matplotlib as mpl
    from pathlib import Path
    
    double_style_path = config["plotting"]["mplstyle_double_path"]
    project_root = Path(__file__).parents[2]
    style_full_path = project_root / double_style_path
    
    with mpl.rc_context(fname=str(style_full_path)):
        fig, (ax_left, ax_right) = plt.subplots(1, 2)
    
    # LEFT PANEL: Per-L CDFs (color-coded by L)
    L_values = _resolve_L_values(config, df["L"].unique())
    
    try:
        xmin, xmax = config["plotting"]["xlim"]
        cmap_name = config["plotting"]["colors"]["colormap"]
    except KeyError as e:
        raise KeyError(f"Config missing plotting parameter: {e}")
    
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.0, 1.0, len(L_values)))
    
    for L, color in zip(L_values, colors):
        vals = df.loc[df["L"] == L, "eigenvalue_magnitude"].to_numpy()
        xs = np.sort(vals)
        n = xs.size
        ys = np.arange(1, n + 1) / n
        
        # Extend to full axis range
        xs_plot = np.concatenate(([xmin], xs, [xmax]))
        ys_plot = np.concatenate(([0.0], ys, [1.0]))
        
        ax_left.step(xs_plot, ys_plot, where="post", label=f"L={L}", color=color, alpha=0.8)
    
    # Reference line at |lambda| = 1
    ax_left.axvline(x=1.0, color="red", linestyle="--", alpha=0.7, label="Unit circle")
    
    ax_left.set_xlabel(r"Eigenvalue magnitude $|\lambda|$")
    ax_left.set_ylabel("Cumulative probability")
    ax_left.set_xlim([xmin, xmax])
    ax_left.set_ylim([0.0, 1.2])
    ax_left.grid(True, alpha=0.3, linestyle=":")
    ax_left.legend(loc="best", fontsize="small", framealpha=0.9)
    
    # RIGHT PANEL: rho_min vs L
    # Filter to mixture only if available
    if "setting" in df.columns:
        df_mix = df[df["setting"] == "mixture_spurious"].copy()
    else:
        df_mix = df.copy()
    
    rho_min = df_mix.groupby(["L", "trial_id"])["eigenvalue_magnitude"].min().reset_index()
    rho_min.rename(columns={"eigenvalue_magnitude": "rho_min"}, inplace=True)
    
    try:
        p_low, p_high = config["plotting"]["confidence_percentiles"]
    except KeyError:
        p_low, p_high = 5, 95
    
    L_values = []
    medians = []
    means = []
    lowers = []
    uppers = []
    
    for L in sorted(rho_min["L"].unique()):
        vals = rho_min[rho_min["L"] == L]["rho_min"]
        L_values.append(L)
        medians.append(vals.median())
        means.append(vals.mean())
        lowers.append(np.percentile(vals, p_low))
        uppers.append(np.percentile(vals, p_high))
    
    ax_right.plot(L_values, medians, marker="o", linestyle="-", markersize=4,
                 label=r"Median($\rho_{\min}$)")
    ax_right.plot(L_values, means, marker="", linestyle="--", linewidth=1.0,
                 label=r"Mean($\rho_{\min}$)")
    
    # Format percentiles
    p_low_str = f"{p_low:.1f}".rstrip('0').rstrip('.') if p_low % 1 else f"{int(p_low)}"
    p_high_str = f"{p_high:.1f}".rstrip('0').rstrip('.') if p_high % 1 else f"{int(p_high)}"
    ax_right.fill_between(L_values, lowers, uppers, alpha=0.3, 
                          label=f"{p_low_str}–{p_high_str}% band")
    
    ax_right.set_xlabel(r"Embedding length $L$")
    ax_right.set_ylabel(r"$\rho_{\min}$")
    ax_right.set_xlim(left=1)
    ax_right.grid(True, alpha=0.3, linestyle=":")
    ax_right.legend(loc="best", fontsize="small", framealpha=0.9)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")



# =============================================================================
# Figure 6: mu_L and nu_L vs L
# =============================================================================


def plot_mu_nu_vs_L(df: pd.DataFrame, config: dict[str, Any], output_path: Path) -> None:
    """
    Plot mu_L and nu_L vs L for mixture_spurious only, side-by-side.
    Includes sqrt((M-m)/L) reference line.
    Requires 'mu_L', 'nu_L', 'trial_id' columns.
    """
    if "mu_L" not in df.columns or "nu_L" not in df.columns or "trial_id" not in df.columns:
        print("Warning: Skipping mu_L/nu_L plot - requires 'mu_L', 'nu_L', and 'trial_id' columns")
        return
    
    # Filter to mixture only if available
    if "setting" in df.columns:
        df_mix = df[df["setting"] == "mixture_spurious"].copy()
    else:
        df_mix = df.copy()
    
    # Extract unique (L, trial_id, mu_L, nu_L) - mu_L and nu_L are duplicated per eigenvalue
    df_unique = df_mix.groupby(["L", "trial_id"]).agg({
        "mu_L": "first",
        "nu_L": "first"
    }).reset_index()
    
    # Infer M-m from eigenvalue counts
    eig_counts = df_mix.groupby(["L", "trial_id"]).size()
    M_minus_m = int(eig_counts.iloc[0])
    
    # Get percentiles from config
    try:
        p_low, p_high = config["plotting"]["confidence_percentiles"]
    except KeyError:
        p_low, p_high = 5, 95
    
    # Get sorted L values
    L_values_sorted = sorted(df_unique["L"].unique())
    
    # Aggregate per L
    L_values = []
    mu_medians, mu_lowers, mu_uppers = [], [], []
    nu_medians, nu_lowers, nu_uppers = [], [], []
    
    for L in L_values_sorted:
        df_L = df_unique[df_unique["L"] == L]
        L_values.append(L)
        
        # mu_L stats
        mu_medians.append(df_L["mu_L"].median())
        mu_lowers.append(np.percentile(df_L["mu_L"], p_low))
        mu_uppers.append(np.percentile(df_L["mu_L"], p_high))
        
        # nu_L stats
        nu_medians.append(df_L["nu_L"].median())
        nu_lowers.append(np.percentile(df_L["nu_L"], p_low))
        nu_uppers.append(np.percentile(df_L["nu_L"], p_high))
    
    # Compute reference line: sqrt((M-m)/L)
    L_array = np.array(L_values)
    reference = np.sqrt(M_minus_m / L_array)
    
    # Format percentiles for legend
    p_low_str = f"{p_low:.1f}".rstrip('0').rstrip('.') if p_low % 1 else f"{int(p_low)}"
    p_high_str = f"{p_high:.1f}".rstrip('0').rstrip('.') if p_high % 1 else f"{int(p_high)}"
    
    # Create side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    
    # Left panel: mu_L
    line_mu, = ax1.plot(L_values, mu_medians, marker="s", linestyle="-", linewidth=2, 
                        color="C0", markersize=3, zorder=10)
    
    ax1.set_xlabel(r"Embedding length $L$")
    ax1.set_ylabel(r"Spectral norm")
    ax1.set_xlim(left=1)
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, linestyle=":")
    
    # Right panel: nu_L
    line_nu, = ax2.plot(L_values, nu_medians, marker="s", linestyle="-", linewidth=2,
                        color="C1", markersize=3, zorder=10)
    band_nu = ax2.fill_between(L_values, nu_lowers, nu_uppers, alpha=0.3, color="C1")
    
    # Reference line on both panels
    ref, = ax1.plot(L_values, reference, linestyle=":", linewidth=2, color="gray")
    ax2.plot(L_values, reference, linestyle=":", linewidth=2, color="gray")
    
    # Legend below figure
    fig.legend(handles=[line_mu, line_nu, band_nu, ref], 
               labels=[r"$\mu_L$ median", r"$\nu_L$ median", 
                      f"{p_low_str}–{p_high_str}% band", rf"$\sqrt{{(M-m)/L}}$"],
               loc='lower center', bbox_to_anchor=(0.5, -0.20), ncol=2, framealpha=0.95)
    
    ax2.set_xlabel(r"Embedding length $L$")
    ax2.set_xlim(left=1)
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, linestyle=":")
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path} (M-m = {M_minus_m})")


# =============================================================================
# Main
# =============================================================================


def main(config_path: str) -> None:
    """
    Generate all spurious eigenvalue figures from experiment results.
    
    Args:
        config_path: Path to YAML config file containing:
            - plotting.mplstyle_path
            - plotting.data_csv_path
            - plotting.xlim
            - plotting.colors.colormap
            - plotting.kde.x_eval_points
            - plotting.kde.bandwidth
            - plotting.confidence_percentiles
            - output.output_dir
    """
    # Load config and resolve paths
    config = plotting_common.load_yaml_config(config_path)
    project_root = plotting_common.resolve_project_root(config, config_path)
    
    # Apply style and load data
    plotting_common.apply_style(config["plotting"], project_root)
    df = validate_and_load_data(config, project_root)
    
    # Get output directory
    try:
        output_rel = config["output"]["output_dir"]
    except KeyError:
        raise KeyError("Config missing output.output_dir")
    
    output_dir = plotting_common.resolve_path(project_root, output_rel)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Generate all figures
    print("Generating figures...")
    
    # Core plots (always generated)
    plot_cdf_per_L(df, config, output_dir / "spurious_eigs_L_cdf.pdf")
    
    # Conditional plots (require specific columns)
    plot_rho_min_vs_L(df, config, output_dir / "spurious_eigs_rho_min_vs_L.pdf")
    plot_combined_cdf_comparison(df, config, output_dir / "spurious_and_noise_cdf_comparison.pdf")
    plot_cdf_and_rho_min_combined(df, config, output_dir / "cdf_and_rho_min_combined.pdf")
    plot_mu_nu_vs_L(df, config, output_dir / "mu_nu_vs_L.pdf")
    
    print("\nDone.")


if __name__ == "__main__":
    fire.Fire(main)
