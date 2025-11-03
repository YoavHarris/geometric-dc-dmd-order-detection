import numpy as np
from numpy.typing import NDArray
from utils.visualizations import scatter_scores_1d

class EstimatedSubspaceLeakage:
    """
    Efficient Estimated Subspace Leakage (ESL) computation for DMD/EDMD.
    Given exact modes (D*L, M) and eigenvalues (M,),
    computes per-mode leakage as:
      ESL_m^2 = ||mode_m||^2 - |lambda_m|^2
    and outputs score = -log(ESL_m + epsilon)
    """
    def __init__(self, epsilon: float = 1e-12):
        self.epsilon = epsilon

    @staticmethod
    def _compute_esl_squared(
        exact_modes: NDArray[np.complexfloating],
        eigenvalues: NDArray[np.complexfloating],
    ) -> NDArray[np.floating]:
        # norm squared of each mode
        mode_norm2 = np.sum(np.abs(exact_modes) ** 2, axis=0)
        eigval_abs2 = np.abs(eigenvalues) ** 2
        esl_sq = mode_norm2 - eigval_abs2
        # Negative? Clip at zero for physical meaningfulness
        esl_sq = np.maximum(esl_sq, 0.0)
        return esl_sq

    def compute_features(
        self,
        exact_modes: NDArray[np.complexfloating],  # shape (D*L, M)
        eigenvalues: NDArray[np.complexfloating],  # shape (M,)
        plot: bool = False,
    ) -> dict[str, NDArray[np.floating]]:
        esl_sq = self._compute_esl_squared(exact_modes, eigenvalues)
        esl_score = -np.log(esl_sq + self.epsilon)
        if plot:
            scatter_scores_1d(esl_score, "Estimated-Subspace-Leakage", title="ESL Scores", show_id=True)
        return {"Estimated-Subspace-Leakage": esl_score.astype(np.float32)}
