from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Literal, Optional

# Optional plotting hooks (no-ops if unavailable)
try:
    from utils.visualizations import plot_matrices_list, scatter_scores_2d
except Exception:  # pragma: no cover
    def plot_matrices_list(*args, **kwargs):  # type: ignore
        pass
    def scatter_scores_2d(*args, **kwargs):  # type: ignore
        pass

from utils.dmd_utils import fit_dmd

ModeName = Literal["nested_dmd", "febvf"]


# -------------------------- helpers -------------------------- #

def _wrap_angle(delta: NDArray[np.floating]) -> NDArray[np.floating]:
    """Wrap angles to (-pi, pi]."""
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def _relative_fro_error(X: NDArray[np.complexfloating], Xhat: NDArray[np.complexfloating]) -> float:
    """Relative Frobenius: ||X - Xhat||_F / ||X||_F, with a well-defined zero case."""
    num = np.linalg.norm((X - Xhat).ravel())
    den = np.linalg.norm(X.ravel())
    if den == 0.0:
        return 0.0 if np.linalg.norm(Xhat.ravel()) == 0.0 else 1.0
    return float(num / den)


def _eig_distance(ref: NDArray[np.complexfloating],
                  est: NDArray[np.complexfloating]) -> NDArray[np.floating]:
    """
    Distance(ref, est) = normalized_angle_distance + 2*|mag1 - mag2|/(mag1 + mag2).

    - normalized_angle_distance = |wrap(arg(ref) - arg(est))| / pi  ∈ [0, 1]
    - magnitude term ∈ [0, 2]; we handle zero-denominator safely.
    """
    ref = ref.astype(np.complex128)
    est = est.astype(np.complex128)

    # angle term in [0,1]
    delta = _wrap_angle(np.angle(ref) - np.angle(est))
    angle_term = np.abs(delta) / np.pi

    # magnitude term with safe division
    mag1 = np.abs(ref)
    mag2 = np.abs(est)
    num = 2.0 * np.abs(mag1 - mag2)
    den = mag1 + mag2
    # where den==0: if num==0 -> 0 else -> inf
    mag_term = np.divide(num, den, out=np.full_like(num, np.inf, dtype=np.float64), where=(den != 0))
    mag_term = np.where((den == 0) & (num == 0), 0.0, mag_term)

    return (angle_term + mag_term).astype(np.float64)


def _febvf_w(eigenvalue_est: complex, L: int) -> NDArray[np.complexfloating]:
    """w = [1, conj(λ), conj(λ)^2, ..., conj(λ)^{L-1}]^T."""
    c = np.conjugate(eigenvalue_est)
    return np.array([c**k for k in range(L)], dtype=np.complex128)


def _geom_series_abs_sq_sum(mag: float, L: int) -> float:
    """Sum_{ell=0}^{L-1} mag^{2*ell}, closed-form."""
    r2 = mag * mag
    if r2 == 1.0:
        return float(L)
    return float((1.0 - r2**L) / (1.0 - r2))


# ------------------------ main class ------------------------- #

class ModeNestedDMD:
    """
    Two operation modes:

    - mode="nested_dmd":
        * rank-1 DMD per mode matrix (D x L)
        * features:
            - Reconstruction: ||X - Xhat||_F / ||X||_F  (then scored by -log(.+epsilon))
            - Eigenvalue-Consistency: normalized angle distance + magnitude ratio term

    - mode="febvf":
        * Fixed–Eigenvalue BV Fit:
            residual = 1 - || Phi_j @ w(eigenvalue_est_j) ||_2^2 / sum_{ell=0}^{L-1} |eigenvalue_est_j|^{2*ell}
        * feature "BV-Conformity": -log(residual + epsilon)

    No magic numbers: only explicit epsilon (default: machine epsilon for float64).
    """

    def __init__(
        self,
        num_delays: int,
        spatial_dim: int,
        epsilon: float = float(np.finfo(np.float64).eps),
        mode: ModeName = "nested_dmd",
    ):
        self.L = int(num_delays)
        self.D = int(spatial_dim)
        if not (np.isfinite(epsilon) and epsilon > 0.0):
            raise ValueError("epsilon must be a finite positive float.")
        self.epsilon = float(epsilon)
        self.mode = mode

    def compute_features(
        self,
        modes: NDArray[np.complexfloating],       # (D*L, M)
        eigenvalues: NDArray[np.complexfloating], # (M,)
        plot: bool = False,
        **fit_dmd_kwargs,
    ) -> Dict[str, NDArray[np.floating]]:
        """
        Returns:
          - nested_dmd:
              "Reconstruction", "Eigenvalue-Consistency",
              "Reconstruction_raw", "Eigenvalue-Consistency_raw"
          - febvf:
              "BV-Conformity", "BV-Conformity_raw"
        """
        D, L = self.D, self.L
        if modes.shape[0] != D * L:
            raise ValueError(f"`modes` must be (D*L, M) with D={D}, L={L}.")
        if eigenvalues.ndim != 1 or eigenvalues.shape[0] != modes.shape[1]:
            raise ValueError("`eigenvalues` must be shape (M,) matching number of modes.")
        M = modes.shape[1]

        # (DL, M) -> (M, D, L)
        mode_mats = modes.T.reshape(M, L, -1).transpose(0, 2, 1).astype(np.complex128)

        if plot:
            plot_matrices_list(mode_mats, title="Mode Sub-Matrices",
                               x_label="Delay Index", y_label="Spatial Dimension")
            baseline = mode_mats[:, :, [0]]
            mask = np.abs(baseline) != 0.0
            normalized = np.where(mask, mode_mats / baseline, mode_mats)
            plot_matrices_list(normalized, title="Mode Sub-Matrices (Normalized by first delay where defined)",
                               x_label="Delay Index", y_label="Spatial Dimension")

        if self.mode == "nested_dmd":
            return self._compute_nested_dmd_features(mode_mats, eigenvalues, **fit_dmd_kwargs)
        elif self.mode == "febvf":
            return self._compute_febvf_features(mode_mats, eigenvalues)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    # ------------------ mode implementations ------------------ #

    def _compute_nested_dmd_features(
        self,
        mode_mats: NDArray[np.complexfloating],      # (M, D, L)
        eigenvalues: NDArray[np.complexfloating],    # (M,)
        **fit_dmd_kwargs,
    ) -> Dict[str, NDArray[np.floating]]:
        # Fit rank-1 DMD per mode
        fitted = [fit_dmd(X, svd_rank=1, **fit_dmd_kwargs) for X in mode_mats]

        # Reconstruction errors
        recon_err = np.array(
            [_relative_fro_error(mode_mats[i], f.reconstructed_data) if hasattr(f, "reconstructed_data") else np.nan
             for i, f in enumerate(fitted)],
            dtype=np.float64,
        )

        # First eigenvalue from each fit
        eigenvalues_est = np.array(
            [f.eigs[0] if hasattr(f, "eigs") and len(getattr(f, "eigs", [])) >= 1 else np.nan + 1j*np.nan
             for f in fitted],
            dtype=np.complex128,
        )

        # Consistency distance (your requested formula)
        eig_err = _eig_distance(eigenvalues.astype(np.complex128), eigenvalues_est)

        eps = self.epsilon
        return {
            "Reconstruction": (-np.log(recon_err + eps)).astype(np.float32),
            "Eigenvalue-Consistency": (-np.log(eig_err + eps)).astype(np.float32),
            "Reconstruction_raw": recon_err.astype(np.float32),
            "Eigenvalue-Consistency_raw": eig_err.astype(np.float32),
        }

    def _compute_febvf_features(
        self,
        mode_mats: NDArray[np.complexfloating],      # (M, D, L)
        eigenvalues: NDArray[np.complexfloating],    # (M,)
    ) -> Dict[str, NDArray[np.floating]]:
        M, _, L = mode_mats.shape
        eigenvalues = eigenvalues.astype(np.complex128)

        # w vectors and numerators || Phi_j @ w ||^2
        w_list = [_febvf_w(eigenvalues[j], L) for j in range(M)]                # (L,) each
        num_sq = np.array([np.linalg.norm(mode_mats[j] @ w_list[j])**2 for j in range(M)], dtype=np.float64)

        # Denominator sum_{ell=0}^{L-1} |λ|^{2 ell}
        magnitudes = np.abs(eigenvalues)
        den = np.array([_geom_series_abs_sq_sum(magnitudes[j], L) for j in range(M)], dtype=np.float64)

        residual = 1.0 - num_sq / den
        residual = np.maximum(residual, 0.0)  # clip tiny negative due to numeric roundoff

        eps = self.epsilon
        score = -np.log(residual + eps)

        return {
            "BV-Conformity": score.astype(np.float32),
            "BV-Conformity_raw": residual.astype(np.float32),
        }
