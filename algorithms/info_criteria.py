"""info_criteria_dmd.py
Automatic rank (order) selection for Dynamic Mode Decomposition using
AIC, AICc (finite‑sample correction), and BIC/MDL — wrapped in a single
object‑oriented helper class ``InformationCriteriaOrderEstimator``.

All bookkeeping is carried out in the *physical* state space of dimension
``D``; delay embedding only reduces the number of usable snapshots from
``N`` to ``M = N - L + 1``.
"""

from __future__ import annotations


import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from utils.dmd_utils import fit_dmd


class InformationCriteriaOrderEstimator:
    def __init__(self, num_delays: int) -> None:
        self.num_delays: int = num_delays

        # Filled by ``fit``
        self.ranks: list[int] = []
        self._aic: list[float] = []
        self._aicc: list[float] = []
        self._bic: list[float] = []
        self._best: dict[str, int] = {}

    def fit(
        self,
        data: NDArray[np.complexfloating],
        max_rank: int,
        min_rank: int = 1,
        plot: bool = False,
    ) -> dict[str, int]:
        """Run the rank scan and return the best rank for each criterion.

        Parameters
        ----------
        data : array (spatial_dim, num_snapshots)
            Snapshot matrix.
        max_rank : int
            Maximum rank to test (inclusive).
        min_rank : int, default 1
            Minimum rank to test (inclusive).
        plot : bool, default False
            Display the criterion curves.

        Returns
        -------
        dict
            {"AIC": rank_aic, "AICc": rank_aicc, "BIC": rank_bic} — the rank
            that minimizes each information criterion.
        """

        spatial_dim, num_snapshots = data.shape
        num_delays = self.num_delays
        num_usable_snapshots = num_snapshots - num_delays + 1

        if num_usable_snapshots <= 1:
            raise ValueError("num_delays too large for the number of snapshots")

        self.ranks.clear()
        self._aic.clear()
        self._aicc.clear()
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

            aic, aicc, bic = self._compute_information_criteria(
                residual_sum_squares, num_observations, num_parameters
            )

            self.ranks.append(rank)
            self._aic.append(aic)
            self._aicc.append(aicc)
            self._bic.append(bic)

        # Locate minima ---------------------------------------------------
        aic_best_idx = int(np.argmin(self._aic))
        aicc_best_idx = int(np.argmin(self._aicc))
        bic_best_idx = int(np.argmin(self._bic))

        self._best = {
            "AIC": self.ranks[aic_best_idx],
            "AICc": self.ranks[aicc_best_idx],
            "BIC": self.ranks[bic_best_idx],
        }

        if plot:
            self._plot_criteria()

        return self._best

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def best_ranks(self) -> dict[str, int]:
        """Return the cached best ranks (call :py:meth:`fit` first)."""
        if not self._best:
            raise RuntimeError("Call 'fit' before requesting results.")
        return self._best

    # ------------------------------------------------------------------
    # Implementation details
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_information_criteria(
        residual_sum_squares: float,
        num_observations: int,
        num_parameters: int,
    ) -> tuple[float, float, float]:
        """Return (AIC, AICc, BIC) given residuals, observations, and parameters."""
        # Avoid log(0)
        residual_sum_squares = max(residual_sum_squares, np.finfo(float).eps)

        aic = (
            2 * num_observations * np.log(residual_sum_squares / num_observations)
            + 2 * num_parameters
        )

        if num_observations > num_parameters + 1:
            aicc = aic + 2 * num_parameters * (num_parameters + 1) / (
                num_observations - num_parameters - 1
            )
        else:
            aicc = np.inf

        bic = num_observations * np.log(
            residual_sum_squares / num_observations
        ) + num_parameters * np.log(num_observations)

        return aic, aicc, bic

    # ------------------------------------------------------------------
    # Optional plotting
    # ------------------------------------------------------------------

    def _plot_criteria(self) -> None:
        plt.plot(self.ranks, self._aic, label="AIC")
        plt.plot(self.ranks, self._aicc, label="AICc")
        plt.plot(self.ranks, self._bic, label="BIC/MDL")
        plt.legend()
        plt.title("Information‑criterion rank selection")
        plt.xlabel("Rank m")
        plt.ylabel("Criterion value (lower is better)")
        plt.show()


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
