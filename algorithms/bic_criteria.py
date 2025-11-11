"""bic_criteria.py
Minimal BIC (Bayesian Information Criterion) order selection for DMD.

Provides a streamlined alternative to the full information criteria module,
containing only BIC/MDL for rank selection.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from utils.dmd_utils import fit_dmd


class BICOrderEstimator:
    def __init__(self, num_delays: int) -> None:
        self.num_delays: int = num_delays

        # Filled by ``fit``
        self.ranks: list[int] = []
        self._bic: list[float] = []
        self._best_rank: int | None = None

    def fit(
        self,
        data: NDArray[np.complexfloating],
        max_rank: int,
        min_rank: int = 1,
        plot: bool = False,
    ) -> int:
        """Run the rank scan and return the best rank according to BIC.

        Parameters
        ----------
        data : array (spatial_dim, num_snapshots)
            Snapshot matrix.
        max_rank : int
            Maximum rank to test (inclusive).
        min_rank : int, default 1
            Minimum rank to test (inclusive).
        plot : bool, default False
            Display the BIC curve.

        Returns
        -------
        int
            The rank that minimizes BIC.
        """

        spatial_dim, num_snapshots = data.shape
        num_delays = self.num_delays
        num_usable_snapshots = num_snapshots - num_delays + 1

        if num_usable_snapshots <= 1:
            raise ValueError("num_delays too large for the number of snapshots")

        self.ranks.clear()
        self._bic.clear()

        for rank in range(min_rank, max_rank + 1):
            # Fit rank-constrained DMD
            dmd = fit_dmd(data, num_delays=num_delays, svd_rank=rank)
            reconstruction = dmd.reconstructed_data[
                :, num_delays - 1 : num_delays - 1 + num_usable_snapshots
            ]

            residual = (
                data[:, num_delays - 1 : num_delays - 1 + num_usable_snapshots]
                - reconstruction
            )
            residual_sum_squares = float(np.sum(np.abs(residual) ** 2))

            num_observations = spatial_dim * num_usable_snapshots
            num_parameters = (
                2 * rank * (spatial_dim + 1) + 1
            )  # eigenvalues, modes, noise variance

            bic = self._compute_bic(
                residual_sum_squares, num_observations, num_parameters
            )

            self.ranks.append(rank)
            self._bic.append(bic)

        # Locate minimum
        bic_best_idx = int(np.argmin(self._bic))
        self._best_rank = self.ranks[bic_best_idx]

        if plot:
            self._plot_bic()

        return self._best_rank

    @property
    def best_rank(self) -> int:
        """Return the cached best rank (call :py:meth:`fit` first)."""
        if self._best_rank is None:
            raise RuntimeError("Call 'fit' before requesting results.")
        return self._best_rank

    @staticmethod
    def _compute_bic(
        residual_sum_squares: float,
        num_observations: int,
        num_parameters: int,
    ) -> float:
        """Compute BIC given residuals, observations, and parameters."""
        # Avoid log(0)
        residual_sum_squares = max(residual_sum_squares, np.finfo(float).eps)

        bic = num_observations * np.log(
            residual_sum_squares / num_observations
        ) + num_parameters * np.log(num_observations)

        return bic

    def _plot_bic(self) -> None:
        """Plot the BIC curve."""
        plt.plot(self.ranks, self._bic, label="BIC/MDL", marker="o")
        plt.axvline(
            self._best_rank, color="red", linestyle="--", alpha=0.7, label="Best rank"
        )
        plt.legend()
        plt.title("BIC rank selection")
        plt.xlabel("Rank m")
        plt.ylabel("BIC (lower is better)")
        plt.grid(True, alpha=0.3)
        plt.show()
