"""
Plot spurious eigenvalue magnitude results vs embedding length L.

Simple standalone script to visualize how spurious eigenvalues move toward
the unit circle as L increases.

Usage:
    python spurious_eigs_L_plot_results.py <csv_path> [--output_dir OUTPUT_DIR]
"""

from __future__ import annotations

import fire
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def plot_cdf(df: pd.DataFrame, output_path: str):
    """
    Plot CDFs of spurious eigenvalue magnitudes for each L value.

    Args:
        df: DataFrame with columns: L, eigenvalue_magnitude
        output_path: Path to save figure.
    """
    L_values = sorted(df["L"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(L_values)))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot CDF for each L
    for L, color in zip(L_values, colors):
        L_data = df[df["L"] == L]["eigenvalue_magnitude"].values

        # Compute empirical CDF
        sorted_data = np.sort(L_data)
        n = len(sorted_data)
        y = np.arange(1, n + 1) / n

        ax.plot(
            sorted_data,
            y,
            label=f"L={L}",
            color=color,
            linewidth=2,
            alpha=0.8,
        )

    # Add unit circle reference
    ax.axvline(
        x=1.0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Unit Circle",
    )

    # Formatting
    ax.set_xlabel("Eigenvalue Magnitude |λ|", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Probability", fontsize=12, fontweight="bold")
    ax.set_title(
        "CDF: Spurious Eigenvalues → Unit Circle as L ↑",
        fontsize=13,
        fontweight="bold",
    )

    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="lower right", fontsize=10, ncol=2, framealpha=0.9)
    ax.set_xlim([0.85, 1.02])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"CDF saved to: {output_path}")


def plot_pdf(df: pd.DataFrame, output_path: str):
    """
    Plot smoothed PDFs (KDE) of spurious eigenvalue magnitudes for each L value.

    Args:
        df: DataFrame with columns: L, eigenvalue_magnitude
        output_path: Path to save figure.
    """
    L_values = sorted(df["L"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(L_values)))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Common x-axis for KDE evaluation
    x_eval = np.linspace(0.85, 1.02, 500)

    # Plot KDE for each L
    for L, color in zip(L_values, colors):
        L_data = df[df["L"] == L]["eigenvalue_magnitude"].values

        # Compute KDE
        kde = stats.gaussian_kde(L_data)
        kde.set_bandwidth(bw_method="scott")

        # Evaluate KDE
        pdf = kde(x_eval)
        pdf = np.maximum(pdf, 0)  # Ensure non-negative

        ax.plot(
            x_eval,
            pdf,
            label=f"L={L}",
            color=color,
            linewidth=2,
            alpha=0.8,
        )

    # Add unit circle reference
    ax.axvline(
        x=1.0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Unit Circle",
    )

    # Formatting
    ax.set_xlabel("Eigenvalue Magnitude |λ|", fontsize=12, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=12, fontweight="bold")
    ax.set_title(
        "PDF: Spurious Eigenvalues → Unit Circle as L ↑",
        fontsize=13,
        fontweight="bold",
    )

    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", fontsize=10, ncol=2, framealpha=0.9)
    ax.set_xlim([0.85, 1.02])
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"PDF saved to: {output_path}")


def main(csv_path: str, output_dir: str = "results"):
    """
    Plot spurious eigenvalue magnitude results vs L.

    Args:
        csv_path: Path to CSV file with results.
        output_dir: Output directory for plots (default: results).
    """
    # Load data
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} spurious eigenvalues")
    print(f"L values: {sorted(df['L'].unique())}")
    print()

    # Create output directory
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plot_cdf(df, output_dir_path / "spurious_eigs_L_cdf.png")
    plot_pdf(df, output_dir_path / "spurious_eigs_L_pdf.png")

    print()
    print("Done!")


if __name__ == "__main__":
    fire.Fire(main)
