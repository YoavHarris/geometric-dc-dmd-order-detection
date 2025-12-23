"""bic.py
Bayesian Information Criterion (BIC) for DMD rank selection.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from utils.dmd_utils import fit_dmd


def compute_bic_rank(
    data: NDArray[np.complexfloating],
    max_rank: int,
    num_delays: int = 1,
    min_rank: int = 1,
    plot: bool = False,
) -> int:
    """
    Estimate the optimal DMD rank using the Bayesian Information Criterion (BIC).

    Parameters
    ----------
    data : array (spatial_dim, num_snapshots)
        Snapshot matrix.
    max_rank : int
        Maximum rank to test (inclusive).
    num_delays : int, default 1
        Number of delays for Hankel embedding.
    min_rank : int, default 1
        Minimum rank to test (inclusive).
    plot : bool, default False
        If True, plots the BIC curve.

    Returns
    -------
    int
        The rank that minimizes the BIC.
    """
    spatial_dim, num_snapshots = data.shape
    num_usable_snapshots = num_snapshots - num_delays + 1

    if num_usable_snapshots <= 1:
        raise ValueError("num_delays too large for the number of snapshots")

    ranks = []
    bic_values = []

    for rank in range(min_rank, max_rank + 1):
        # Fit rank-constrained DMD
        dmd = fit_dmd(data, num_delays=num_delays, svd_rank=rank)

        # Reconstruct the data window corresponding to the valid Hankel columns
        reconstruction = dmd.reconstructed_data[
            :, num_delays - 1 : num_delays - 1 + num_usable_snapshots
        ]

        residual = (
            data[:, num_delays - 1 : num_delays - 1 + num_usable_snapshots]
            - reconstruction
        )
        residual_sum_squares = float(np.sum(np.abs(residual) ** 2))

        # Avoid log(0)
        residual_sum_squares = max(residual_sum_squares, np.finfo(float).eps)

        num_observations = spatial_dim * num_usable_snapshots
        num_parameters = (
            2 * rank * (spatial_dim + 1) + 1
        )  # eigenvalues, modes, noise variance

        bic = num_observations * np.log(
            residual_sum_squares / num_observations
        ) + num_parameters * np.log(num_observations)

        ranks.append(rank)
        bic_values.append(bic)

    # Locate minimum
    best_idx = int(np.argmin(bic_values))
    best_rank = ranks[best_idx]

    if plot:
        plt.figure()
        plt.plot(ranks, bic_values, label="BIC/MDL", marker="o")
        plt.axvline(
            best_rank, color="r", linestyle="--", label=f"Best rank: {best_rank}"
        )
        plt.legend()
        plt.title("BIC-based rank selection")
        plt.xlabel("Rank m")
        plt.ylabel("BIC value (lower is better)")
        plt.grid(True, alpha=0.3)
        plt.show()

    return best_rank


def gap_ranks(data: NDArray[np.floating]) -> int:
    """
    Estimate optimal rank via the largest singular value gap.

    Parameters
    ----------
    data : array
        Data matrix to analyze.

    Returns
    -------
    int
        Estimated rank based on the largest singular value gap.
    """
    _, singular_values, _ = np.linalg.svd(data, full_matrices=False)
    gaps = -np.diff(singular_values)
    largest_gap_index = np.argmax(gaps)
    return largest_gap_index + 1
