import numpy as np
from typing import Tuple, Iterator, Dict

from matplotlib import pyplot as plt
from numpy.typing import NDArray

from utils.visualizations import scatter_scores_1d, scatter_scores_2d


class DmdResiduals:
    @staticmethod
    def compute_residuals(
        X: NDArray[np.complexfloating],
        Y: NDArray[np.complexfloating],
        U_M: NDArray[np.complexfloating],
        s_M: NDArray[np.floating],
        V_M: NDArray[np.complexfloating],
    ) -> Tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
        """Compute residuals using SVD-derived propagator."""
        Sigma_M_inv = np.diag(1.0 / s_M)
        A_M = U_M.conj().T @ Y @ V_M @ Sigma_M_inv
        return DmdResiduals.compute_residuals_from_propagator(X, Y, U_M, s_M, V_M, A_M)

    @staticmethod
    def compute_residuals_from_propagator(
        X: NDArray[np.complexfloating],
        Y: NDArray[np.complexfloating],
        U_M: NDArray[np.complexfloating],
        s_M: NDArray[np.floating],
        V_M: NDArray[np.complexfloating],
        A_M: NDArray[np.complexfloating],
    ) -> Tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
        """Compute residuals from a given low-rank propagator A_M."""
        _, W = np.linalg.eig(A_M)
        z = Y @ V_M @ np.diag(1.0 / s_M) @ W

        Q, _ = np.linalg.qr(X, mode="reduced")
        Pz = Q @ (Q.conj().T @ z)  # Projection onto data subspace
        P_M_z = U_M @ (U_M.conj().T @ z)  # Projection onto mode subspace

        R_out = z - Pz
        R_drop = z - P_M_z - R_out
        return R_out, R_drop

    @staticmethod
    def residuals_to_scores(
        residuals: NDArray[np.complexfloating],
    ) -> NDArray[np.floating]:
        """Convert residual norm to score using -log(||·|| + eps)."""
        return -np.log(np.linalg.norm(residuals, axis=0) + 1e-15)

    @classmethod
    def compute_score_dict(
        cls,
        X: NDArray[np.complexfloating],
        Y: NDArray[np.complexfloating],
        U_M: NDArray[np.complexfloating],
        s_M: NDArray[np.floating],
        V_M: NDArray[np.complexfloating],
        plot: bool = False,
    ) -> Dict[str, NDArray[np.floating]]:
        R_out, R_drop = cls.compute_residuals(X, Y, U_M, s_M, V_M)
        R_tot = R_out + R_drop
        score_dict = dict(
            R_out=cls.residuals_to_scores(R_out),
            R_drop=cls.residuals_to_scores(R_drop),
            R_tot=cls.residuals_to_scores(R_tot),
        )
        if plot:
            scatter_scores_1d(
                score_dict["R_tot"], "R_tot", "Total eigen-residual", show_id=True
            )
            scatter_scores_2d(
                np.stack((score_dict["R_out"], score_dict["R_drop"]), axis=1),
                ["R_out", "R_drop"],
                "Eigen Residuals",
                show_id=True,
            )
        return score_dict


class TruncatedPropagatorIterator(Iterator[NDArray[np.complexfloating]]):
    """
    Iterator that yields propagator matrices A_r for r = min_rank ... max_rank
    via rank-one updates to A_r = U_r^H Y V_r Σ_r^{-1}.
    """

    def __init__(
        self,
        Y: NDArray[np.complexfloating],
        U_max: NDArray[np.complexfloating],
        V_max: NDArray[np.complexfloating],
        s_max: NDArray[np.floating],
        min_rank: int = 1,
    ) -> None:
        self.Y = Y
        self.U_max = U_max
        self.V_max = V_max
        self.s_max = s_max
        self.min_rank = min_rank
        self.max_rank = int(s_max.shape[0])

        self._current_M = min_rank
        self._initial_yielded = False

        U_M, V_M, s_M = self._get_current_truncation()
        self.A_current = U_M.conj().T @ Y @ V_M @ np.diag(1 / s_M)

    def __iter__(self) -> "TruncatedPropagatorIterator":
        return self

    def __next__(self) -> NDArray[np.complexfloating]:
        if not self._initial_yielded:
            self._initial_yielded = True
            return self.A_current

        if self._current_M >= self.max_rank:
            raise StopIteration

        idx = self._current_M
        next_u, next_sigma, next_v = self.get_singular_tuple(idx)
        U_M, V_M, s_M = self._get_current_truncation()

        Y_v_over_sigma = self.Y @ next_v / next_sigma
        top_right_correction = U_M.conj().T @ Y_v_over_sigma
        V_Sigma_inv_next = np.column_stack(
            (V_M @ np.diag(1 / s_M), next_v / next_sigma)
        )
        bottom_correction = next_u.conj().T @ self.Y @ V_Sigma_inv_next

        M = self._current_M
        assert self.A_current.shape == (M, M)
        assert top_right_correction.shape == (M,)
        assert bottom_correction.shape == (M + 1,)

        next_A = np.hstack((self.A_current, top_right_correction[:, None]))
        next_A = np.vstack((next_A, bottom_correction[None, :]))

        self._current_M += 1
        self.A_current = next_A
        return next_A

    def _get_current_truncation(self):
        M = self._current_M
        return self.U_max[:, :M], self.V_max[:, :M], self.s_max[:M]

    def get_singular_tuple(self, idx):
        return self.U_max[:, idx], self.s_max[idx], self.V_max[:, idx]
