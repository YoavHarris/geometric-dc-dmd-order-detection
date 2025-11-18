import numpy as np
from numpy.typing import NDArray


class STC:
    """
    Spatiotemporal Coupling (STC) quotient-consistency feature extractor.

    Based on Bronstein et al., Chaos 32, 123127 (2022), included for comparison with the new BV-fit methods
    (NestedDMD, FixedEigenvalueBVFit).

    For each mode, constructs per-delay quotients relative to the first sub-mode,
    estimates an effective eigenvalue via geometric means across valid lags, and
    measures its consistency with the given eigenvalue.

    The resulting feature is returned as a score:
        score = -log10(|estimated_eigenvalue - true_eigenvalue| + epsilon)

    Higher score = better consistency = more likely a true mode.
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
        self.use_nyquist_cap = True

    def compute_features(
        self,
        eigenvalues: NDArray[np.complexfloating],  # shape (M,)
        modes: NDArray[np.complexfloating],  # shape (D*L, M)
        plot: bool = False,
    ) -> dict[str, NDArray[np.floating]]:
        """
        Compute STC consistency scores for each mode.

        Parameters
        ----------
        eigenvalues : array (M,)
            DMD eigenvalues.
        modes : array (D*L, M)
            DMD modes (typically projected modes).
        plot : bool
            Whether to plot scores.

        Returns
        -------
        dict with "STC" -> scores (M,)
        """
        num_modes = eigenvalues.shape[0]
        num_delays = self.num_delays
        spatial_dim = modes.shape[0] // num_delays

        # Reshape modes to (num_modes, num_delays, spatial_dim)
        mode_matrices = modes.T.reshape(num_modes, num_delays, spatial_dim)

        # Extract first delay (reference) for each mode
        first_delay_modes = mode_matrices[:, 0, :]  # (num_modes, spatial_dim)

        # Compute quotients w.r.t. first delay for lags 1..L-1
        # Shape: (num_modes, num_delays-1, spatial_dim)
        with np.errstate(divide="ignore", invalid="ignore"):
            delay_quotients = mode_matrices[:, 1:, :] / first_delay_modes[:, None, :]

        # Apply fractional exponents to extract eigenvalue estimates from each lag
        # For lag k, we take the k-th root: quotient^(1/k)
        lag_exponents = 1.0 / (np.arange(1, num_delays)[None, :, None])  # (1, L-1, 1)
        quotient_eigenvalues = delay_quotients**lag_exponents  # (M, L-1, D)

        # Compute Nyquist-like bounds per mode (how many lags are valid)
        # Digital angular frequency per sample in [0, pi]
        oscillation_rates = np.abs(np.angle(eigenvalues))
        max_valid_lags = self._compute_max_valid_lags(oscillation_rates)

        # For each mode, average the quotient-derived eigenvalues and measure consistency
        estimated_eigenvalues = np.empty(num_modes, dtype=eigenvalues.dtype)
        consistency_errors = np.empty(num_modes, dtype=np.float64)

        for mode_idx in range(num_modes):
            max_lag = int(max_valid_lags[mode_idx])

            if max_lag == 0:
                # No valid lags for this mode
                estimated_eigenvalues[mode_idx] = 0.0
                consistency_errors[mode_idx] = self.numerical_inf
                continue

            # Average across valid lags and spatial dimensions
            valid_estimates = quotient_eigenvalues[mode_idx, :max_lag, :]
            estimated_eigenvalues[mode_idx] = np.mean(valid_estimates)

            # Measure consistency with external eigenvalue
            consistency_errors[mode_idx] = np.abs(
                estimated_eigenvalues[mode_idx] - eigenvalues[mode_idx]
            )

        # Convert errors to scores (larger = better)
        # Use log10 to match original paper (Bronstein et al., Chaos 2022)
        scores = -np.log10(consistency_errors + self.epsilon)

        if plot:
            from utils.visualizations import scatter_scores_1d

            scatter_scores_1d(
                scores, "STC", title="STC Consistency Scores", show_id=True
            )

        return {"STC": scores.astype(np.float32)}

    def _compute_max_valid_lags(
        self, oscillation_rates: NDArray[np.floating]
    ) -> NDArray[np.int32]:
        """
        Compute maximum valid lag for each mode based on Nyquist criterion.

        High-frequency modes (large oscillation rates) need fewer lags
        to avoid aliasing in the quotient estimates.
        """
        if not self.use_nyquist_cap:
            return np.full_like(
                oscillation_rates, fill_value=self.num_delays - 1, dtype=np.int32
            )

        # Nyquist-like bound in samples:
        safe_rates = np.maximum(oscillation_rates, self.epsilon)  # avoid divide-by-zero
        max_lags = np.minimum(
            self.num_delays - 1,
            np.floor(np.pi / safe_rates),
        )
        # Zero oscillation rate means DC component (no valid quotients)
        max_lags = np.where(oscillation_rates == 0, 0, max_lags)
        return max_lags.astype(int)
