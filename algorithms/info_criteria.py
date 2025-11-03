"""info_criteria_dmd.py
Automatic rank (order) selection for Dynamic Mode Decomposition using
AIC, AICc (finite‑sample correction), and BIC/MDL — wrapped in a single
object‑oriented helper class ``InformationCriteriaOrderEstimator``.

All bookkeeping is carried out in the *physical* state space of dimension
``D``; delay embedding only reduces the number of usable snapshots from
``N`` to ``M = N - L + 1``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from utils.dmd_utils import fit_dmd


class InformationCriteriaOrderEstimator:
    def __init__(self, num_delays: int) -> None:
        self.num_delays: int = num_delays

        # Filled by ``fit``
        self.ranks: List[int] = []
        self._aic: List[float] = []
        self._aicc: List[float] = []
        self._bic: List[float] = []
        self._best: Dict[str, int] = {}

    def fit(
        self,
        X: NDArray[np.complexfloating],
        max_rank: int,
        min_rank: int = 1,
        plot: bool = False,
    ) -> Dict[str, int]:
        """Run the rank scan and return the best rank for each criterion.

        Parameters
        ----------
        X : ndarray of shape (D, N)
            Snapshot matrix (*D* state variables over *N* time steps).
        max_rank : int
            Maximum rank *m* to test (inclusive).
        min_rank : int, default 1
            Minimum rank *m* to test (inclusive).
        plot : bool, default False
            Display the criterion curves.

        Returns
        -------
        dict
            ``{"AIC": m_aic, "AICc": m_aicc, "BIC": m_bic}`` — the rank
            that minimises each information criterion.
        """

        D, N_total = X.shape
        L = self.num_delays
        N_tilde = N_total - L + 1  # usable snapshots after delay embedding
        if N_tilde <= 1:
            raise ValueError("num_delays too large for the number of snapshots")

        self.ranks.clear()
        self._aic.clear()
        self._aicc.clear()
        self._bic.clear()

        for r in range(min_rank, max_rank + 1):
            # --- Fit rank‑m DMD (user must supply 'fit_dmd') -------------
            dmd = fit_dmd(X, num_delays=L, svd_rank=r)
            X_hat = dmd.reconstructed_data[:, L - 1 : L - 1 + N_tilde]

            residual = X[:, L - 1 : L - 1 + N_tilde] - X_hat
            rss = float(np.sum(np.abs(residual) ** 2))

            n_obs = D * N_tilde  # complex observations treated as units
            n_params = 2 * r * (D + 1) + 1  # eigenvalues, modes, noise variance

            aic, aicc, bic = self._compute_information_criteria(rss, n_obs, n_params)

            self.ranks.append(r)
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
    def best_ranks(self) -> Dict[str, int]:
        """Return the cached best ranks (call :py:meth:`fit` first)."""
        if not self._best:
            raise RuntimeError("Call 'fit' before requesting results.")
        return self._best

    # ------------------------------------------------------------------
    # Implementation details
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_information_criteria(
        rss: float,
        n_obs: int,
        n_params: int,
    ) -> Tuple[float, float, float]:
        """Return (AIC, AICc, BIC) given residuals, data count, parameter count."""
        rss = max(rss, np.finfo(float).tiny)  # avoid log(0)

        aic = 2 * n_obs * np.log(rss / n_obs) + 2 * n_params
        if n_obs > n_params + 1:
            aicc = aic + 2 * n_params * (n_params + 1) / (n_obs - n_params - 1)
        else:
            aicc = np.inf
        bic = n_obs * np.log(rss / n_obs) + n_params * np.log(n_obs)
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


def gap_ranks(X: NDArray[np.floating]) -> int:
    """
    Estimate optimal rank via the largest singular value gap.

    Returns:
        Estimated rank based on the largest singular value gap.
    """
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    gaps = -np.diff(s)
    largest_gap_index = np.argmax(gaps)
    return largest_gap_index + 1
