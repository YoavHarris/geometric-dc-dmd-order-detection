"""
Simplified plotter for spurious eigenvalue experiment results.

Generates 3 figures:
1. Main text: ρ_min vs L (mixture only) with percentile bands
2. Appendix: Combined CDFs of ρ and ρ_min (mixture vs noise) with KS statistics
3. Appendix: μ_L and ν_L vs L with percentile bands

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
import yaml


# =============================================================================
# Config and data loading
# =============================================================================


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_and_load_data(config: dict[str, Any], project_root: Path) -> pd.DataFrame:
    """Load and validate CSV data."""
    # Get paths from config
    try:
        plot_cfg = config["plotting"]
        csv_rel_path = plot_cfg["data_csv_path"]
    except KeyError as e:
        raise KeyError(f"Config missing required key: {e}")
    
    csv_path = project_root / csv_rel_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    # Load data
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")
    
    # Validate required columns
    required_cols = ["L", "trial_id", "setting", "eigenvalue_magnitude", "mu_L", "nu_L"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    
    # Check settings
    unique_settings = set(df["setting"].unique())
    print(f"Settings found: {sorted(unique_settings)}")
    
    expected = {"mixture_spurious", "noise_only"}
    if not expected.issubset(unique_settings):
        missing_set = expected - unique_settings
        raise ValueError(f"CSV missing required settings: {missing_set}")
    
    unexpected = unique_settings - expected
    if unexpected:
        print(f"WARNING: Unexpected settings found (will be ignored): {unexpected}")
    
    # Filter to expected settings
    df_filtered = df[df["setting"].isin(expected)].copy()
    print(f"After filtering: {len(df_filtered)} rows")
    
    # Print setting counts
    setting_counts = df_filtered["setting"].value_counts()
    print("Rows per setting:")
    for setting, count in setting_counts.items():
        print(f"  {setting}: {count}")
    
    # Print eigenvalue counts per (L, trial_id, setting)
    counts = df_filtered.groupby(["L", "trial_id", "setting"]).size()
    print(f"Eigenvalues per (L, trial_id, setting): min={counts.min()}, median={counts.median():.0f}, max={counts.max()}")
    
    print(f"Unique L values ({len(df_filtered['L'].unique())}): {sorted(df_filtered['L'].unique())}")
    print()
    
    return df_filtered


def apply_mplstyle(config: dict[str, Any], project_root: Path) -> None:
    """Apply matplotlib style from config."""
    try:
        style_rel_path = config["plotting"]["mplstyle_path"]
    except KeyError:
        raise KeyError("Config missing plotting.mplstyle_path")
    
    style_path = project_root / style_rel_path
    if not style_path.exists():
        raise FileNotFoundError(f"mplstyle not found: {style_path}")
    
    plt.style.use(str(style_path))
    print(f"Applied style: {style_path}")


def get_output_dir(config: dict[str, Any], project_root: Path) -> Path:
    """Get output directory from config and create if needed."""
    try:
        output_rel = config["output"]["output_dir"]
    except KeyError:
        raise KeyError("Config missing output.output_dir")
    
    output_dir = project_root / output_rel
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    return output_dir


# =============================================================================
# Figure 1: Main text - ρ_min vs L (mixture only)
# =============================================================================


def plot_rho_min_vs_L(df: pd.DataFrame, config: dict[str, Any], output_path: Path) -> None:
    """
    Plot minimum eigenvalue magnitude vs L for mixture_spurious only.
    Shows median with percentile band.
    """
    # Filter to mixture only
    df_mix = df[df["setting"] == "mixture_spurious"].copy()
    
    # Compute rho_min per (L, trial_id)
    rho_min = df_mix.groupby(["L", "trial_id"])["eigenvalue_magnitude"].min().reset_index()
    rho_min.rename(columns={"eigenvalue_magnitude": "rho_min"}, inplace=True)
    
    # Get percentiles from config
    p_low, p_high = config["plotting"]["confidence_percentiles"]
    
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
    # Format percentiles: remove .0 if whole numbers
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
# Figure 2: Appendix - Combined CDFs (ρ and ρ_min)
# =============================================================================


def plot_combined_cdf(df: pd.DataFrame, config: dict[str, Any], output_path: Path) -> None:
    """
    Plot both ρ (all magnitudes) and ρ_min CDFs in horizontal layout with shared legend.
    Left: CDF of all eigenvalue magnitudes
    Right: CDF of minimum eigenvalue per trial
    """
    # Prepare ρ data (all eigenvalue magnitudes)
    rho_mix = df[df["setting"] == "mixture_spurious"]["eigenvalue_magnitude"].to_numpy()
    rho_noise = df[df["setting"] == "noise_only"]["eigenvalue_magnitude"].to_numpy()
    ks_rho, _ = stats.ks_2samp(rho_mix, rho_noise)
    
    # Prepare ρ_min data
    rho_min = df.groupby(["L", "trial_id", "setting"])["eigenvalue_magnitude"].min().reset_index()
    rho_min.rename(columns={"eigenvalue_magnitude": "rho_min"}, inplace=True)
    rho_min_mix = rho_min[rho_min["setting"] == "mixture_spurious"]["rho_min"].to_numpy()
    rho_min_noise = rho_min[rho_min["setting"] == "noise_only"]["rho_min"].to_numpy()
    ks_rho_min, _ = stats.ks_2samp(rho_min_mix, rho_min_noise)
    
    # Create side-by-side subplots with shared x-axis
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
    
    # Left panel: ρ CDF (all magnitudes)
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
    ax1.set_title(r"Pooled $|\lambda|$")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, linestyle=":")
    
    # Right panel: ρ_min CDF
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
    ax2.set_title(r"Pooled $\rho_{\min}$")
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
# Figure 3: Appendix - μ_L and ν_L vs L
# =============================================================================


def plot_mu_nu_vs_L(df: pd.DataFrame, config: dict[str, Any], output_path: Path) -> None:
    """
    Plot μ_L and ν_L vs L for mixture_spurious only, side-by-side.
    Includes sqrt((M-m)/L) reference line.
    """
    # Filter to mixture only
    df_mix = df[df["setting"] == "mixture_spurious"].copy()
    
    # Extract unique (L, trial_id, mu_L, nu_L) - mu_L and nu_L are duplicated per eigenvalue
    df_unique = df_mix.groupby(["L", "trial_id"]).agg({
        "mu_L": "first",
        "nu_L": "first"
    }).reset_index()
    
    # Get M and m from the first row (assumes constant across dataset)
    # We'll infer M-m from the eigenvalue counts
    eig_counts = df_mix.groupby(["L", "trial_id", "setting"]).size()
    M_minus_m = int(eig_counts.iloc[0])  # M-m is the number of spurious eigenvalues per trial
    
    # Get percentiles from config
    p_low, p_high = config["plotting"]["confidence_percentiles"]
    
    # Get sorted L values
    L_values_sorted = sorted(df_unique["L"].unique())
    
    # Aggregate per L
    L_values = []
    mu_medians, mu_lowers, mu_uppers = [], [], []
    nu_medians, nu_lowers, nu_uppers = [], [], []
    
    for L in L_values_sorted:
        df_L = df_unique[df_unique["L"] == L]
        L_values.append(L)
        
        # μ_L stats
        mu_medians.append(df_L["mu_L"].median())
        mu_lowers.append(np.percentile(df_L["mu_L"], p_low))
        mu_uppers.append(np.percentile(df_L["mu_L"], p_high))
        
        # ν_L stats
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
    
    # Left panel: μ_L
    line_mu, = ax1.plot(L_values, mu_medians, marker="s", linestyle="-", linewidth=2, 
                        color="C0", markersize=3, zorder=10)
    
    ax1.set_xlabel(r"Embedding length $L$")
    ax1.set_ylabel(r"Spectral norm")
    ax1.set_title(r"$\mu_L$ (first $D$ rows)")
    ax1.set_xlim(left=1)
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, linestyle=":")
    
    # Right panel: ν_L
    line_nu, = ax2.plot(L_values, nu_medians, marker="s", linestyle="-", linewidth=2,
                        color="C1", markersize=3, zorder=10)
    band_nu = ax2.fill_between(L_values, nu_lowers, nu_uppers, alpha=0.3, color="C1")
    
    # Reference line on both panels
    ref, = ax1.plot(L_values, reference, linestyle=":", linewidth=2, color="gray",
                    label=rf"$\sqrt{{(M-m)/L}}$")
    ax2.plot(L_values, reference, linestyle=":", linewidth=2, color="gray")
    
    # Legend below figure in 2 rows
    fig.legend(handles=[line_mu, line_nu, band_mu, ref], 
               labels=[r"$\mu_L$ median", r"$\nu_L$ median", 
                      f"{p_low_str}–{p_high_str}% band", rf"$\sqrt{{(M-m)/L}}$"],
               loc='lower center', bbox_to_anchor=(0.5, -0.20), ncol=2, framealpha=0.95)
    
    # Common settings
    ax1.set_ylabel(r"Spectral norm")
    ax1.set_title(r"$\mu_L$ (first $D$ rows)")
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, linestyle=":")
    
    ax2.set_xlabel(r"Embedding length $L$")
    ax2.set_title(r"$\nu_L$ (last $D$ rows)")
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
    Generate spurious eigenvalue figures from experiment results.
    
    Args:
        config_path: Path to YAML config file containing:
            - plotting.mplstyle_path
            - plotting.data_csv_path
            - plotting.xlim_rho
            - plotting.confidence_percentiles
            - output.output_dir
    """
    # Load config and setup
    config = load_config(config_path)
    project_root = Path(__file__).parents[2]
    
    apply_mplstyle(config, project_root)
    df = validate_and_load_data(config, project_root)
    output_dir = get_output_dir(config, project_root)
    
    # Generate figures
    print("Generating figures...")
    plot_rho_min_vs_L(df, config, output_dir / "spurious_eigs_rho_min_vs_L_main.pdf")
    plot_combined_cdf(df, config, output_dir / "spurious_and_noise_eig_magnitudes_cdfs.pdf")
    plot_mu_nu_vs_L(df, config, output_dir / "mu_nu_vs_L_mixture_tail.pdf")
    
    print("\nDone.")


if __name__ == "__main__":
    fire.Fire(main)
