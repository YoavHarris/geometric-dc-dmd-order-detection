import numpy as np
from numpy.typing import NDArray

class STC:
    """
    Spatiotemporal Coupling (STC) quotient-consistency feature extractor.
    For each mode, constructs per-delay quotients relative to the first sub-mode,
    estimates an effective eigenvalue via geometric means across valid lags, and
    measures its consistency w  ith the given eigenvalue.
    The resulting feature is returned as a score:
        score = -log(|lambda_hat - lambda_true| + epsilon)
    Higher is better.
    """
    def __init__(
        self,
        num_delays: int,
        dt: float = 1.0,
        epsilon: float = 1e-12,
        numerical_inf: float = 1e12,
    ) -> None:
        self.num_delays = num_delays
        self.dt = dt
        self.epsilon = epsilon
        self.numerical_inf = numerical_inf
        self.omega_s = 2 * np.pi / dt

    def compute_features(
        self,
        eigenvalues: NDArray[np.complexfloating],  # shape (M,)
        modes: NDArray[np.complexfloating],        # shape (D*L, M)
        plot: bool = False,
    ) -> dict[str, NDArray[np.floating]]:
        M = eigenvalues.shape[0]
        L = self.num_delays
        D = modes.shape[0] // L
        # Reshape to (M, L, D)
        modes_batch = modes.T.reshape(M, L, D)
        submodes_0 = modes_batch[:, 0, :]  # (M, D)
        # Quotients w.r.t. first sub-mode, for lags 1..L-1 -> shape (M, L-1, D)
        with np.errstate(divide="ignore", invalid="ignore"):
            divided_by_first = modes_batch[:, 1:, :] / submodes_0[:, None, :]
        exponents = 1.0 / (np.arange(1, L)[None, :, None])  # (1, L-1, 1)
        lambda_tilde_batch = divided_by_first ** exponents  # (M, L-1, D)
        # Bounds per mode (Nyquist-like)
        oscillations = np.abs(np.log(eigenvalues)) / self.dt
        bounds = self._compute_bounds(oscillations)  # int array, in [0, L-1]
        lambda_hat = np.empty(M, dtype=eigenvalues.dtype)
        eps_err = np.empty(M, dtype=np.float64)
        for i in range(M):
            b = int(bounds[i])
            if b == 0:
                lambda_hat[i] = 0.0
                eps_err[i] = self.numerical_inf
                continue
            vals = lambda_tilde_batch[i, :b, :]  # (b, D)
            lambda_hat[i] = np.mean(vals)
            eps_err[i] = np.abs(lambda_hat[i] - eigenvalues[i])
        # Convert to scores: higher is better
        scores = -np.log(eps_err + self.epsilon)
        if plot:
            from utils.visualizations import scatter_scores_1d
            scatter_scores_1d(scores, "STC")

        return {"STC": scores.astype(np.float32)}

    def _compute_bounds(self, oscillations: NDArray[np.floating]) -> NDArray[np.int32]:
        bounds = np.minimum(self.num_delays - 1, np.floor(0.5 * self.omega_s / oscillations))
        bounds = np.where(oscillations == 0, 0, bounds)
        return bounds.astype(int)
