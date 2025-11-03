from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Literal, Optional

import numpy as np
from numpy.typing import NDArray

# Optional plotting utilities (kept as hooks; no-op if not provided by the caller)
try:
    from utils.visualizations import plot_matrices_list, scatter_scores_2d
except Exception:  # pragma: no cover
    def plot_matrices_list(*args, **kwargs):  # type: ignore
        pass
    def scatter_scores_2d(*args, **kwargs):  # type: ignore
        pass

# Required: a DMD fitter that returns an object with attributes:
#   - reconstructed_data : NDArray[np.complexfloating] of shape (D, L)
#   - eigs               : NDArray[np.complexfloating]  (length >= 1)
from utils.dmd_utils import fit_dmd


# ------------------------------- helpers ------------------------------------ #

def wrap_angle(delta: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Wrap angles to (-pi, pi].
    """
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def eig_distance(
    ref: NDArray[np.complexfloating],
    est: NDArray[np.complexfloating],
    *,
    metric: Literal["log", "chordal", "angle", "modulus", "weighted"] = "log",
    w_alpha: float = 1.0,
    w_omega: float = 1.0,
) -> NDArray[np.floating]:
    """
    Eigenvalue distance between ref and est (vectorized over 1-D arrays).

    Parameters
    ----------
    ref, est : complex ndarray, shape (M,)
        Reference and estimated eigenvalues.

    metric : {"log", "chordal", "angle", "modulus", "weighted"}
        - "log": geodesic in C^* via log-space (default).
        - "chordal": |lambda_ref - lambda_est|.
        - "angle": |Δ arg| (wrapped).
        - "modulus": |Δ log |lambda||.
        - "weighted": sqrt( (w_alpha Δlog|λ|)^2 + (w_omega Δarg)^2 ).

    w_alpha, w_omega : float
        Weights for the "weighted" metric.

    Returns
    -------
    d : float ndarray, shape (M,)
    """
    ref = np.asarray(ref, dtype=np.complex128)
    est = np.asarray(est, dtype=np.complex128)

    if metric == "chordal":
        return np.abs(ref - est).astype(np.float64)

    dalpha = np.log(np.abs(ref)) - np.log(np.abs(est))
    domega = wrap_angle(np.angle(ref) - np.angle(est))

    if metric == "angle":
        return np.abs(domega).astype(np.float64)
    if metric == "modulus":
        return np.abs(dalpha).astype(np.float64)
    if metric == "weighted":
        return np.sqrt((w_alpha * dalpha) ** 2 + (w_omega * domega) ** 2).astype(np.float64)

    # default: "log"
    return np.hypot(dalpha, domega).astype(np.float64)


def vandermonde_scale(alpha: NDArray[np.floating], L: int) -> NDArray[np.floating]:
    """
    S(alpha) = sum_{k=1}^{L-1} k^2 * e^{2k alpha}.
    Vectorized over alpha.

    This is the conformal factor of the induced metric on the Vandermonde
    manifold under the ambient Euclidean metric for ψ_L(λ) with λ=e^{α+iω}.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    if L <= 1:
        return np.zeros_like(alpha)
    k = np.arange(1, L, dtype=np.float64)  # 1,...,L-1
    # shape broadcasting: (K, 1) + (1, A) -> (K, A)
    exps = np.exp(np.outer(2.0 * k, alpha))
    return (k[:, None] ** 2 * exps).sum(axis=0)


# ------------------------------- main class --------------------------------- #

@dataclass
class ModeNestedDMD:
    """
    Publishable-quality implementation of per-mode nested DMD features.

    Given a set of delay-embedded modes φ ∈ C^{DL×M}, this class:
      1) reshapes each mode to a (D×L) mode-matrix,
      2) fits rank-1 DMD to each mode-matrix,
      3) computes two per-mode features:
         - Reconstruction error:  ||X - X̂||_F / ||X||_F
         - Eigenvalue consistency: distance(λ_ext, λ_nested) (configurable metric)

    Scores are produced as:  score = -log(distance + epsilon),
    which converts small errors to large positive scores in a numerically
    controlled way.

    Design goals:
      - No ad-hoc "magic numbers": use machine epsilons and explicit parameters.
      - Clear metrics with principled geometry (log-space by default).
      - Deterministic, vectorized where appropriate, robust to fit failures.

    Parameters
    ----------
    num_delays : int
        L, the number of delays (columns of each mode-matrix).
    spatial_dim : int
        D, the spatial dimension (rows of each mode-matrix).
    epsilon : float, default np.finfo(np.float64).eps
        Positive offset inside -log(⋅) scoring; must be > 0.
    eig_metric : {"log", "chordal", "angle", "modulus", "weighted"}, default "log"
        Metric used for eigenvalue consistency.
    eig_metric_weights : Optional[dict], default None
        Optional {"w_alpha": float, "w_omega": float} if eig_metric="weighted".
    vandermonde_weighting : Optional[Literal["none","Savg"]], default "none"
        If "Savg", multiply the eigenvalue distance by sqrt(S(ᾱ)) where
        ᾱ = 0.5[log|λ_ext| + log|λ_nested|]. Use "none" to remain scale-neutral.
    failure_policy : {"nan", "raise"}, default "nan"
        How to handle DMD fit failures for a mode.

    Notes
    -----
    - The class does not assume any global normalization of input modes.
    - Plotting hooks are optional and skipped if the utilities are unavailable.
    """

    num_delays: int
    spatial_dim: int
    epsilon: float = field(default_factory=lambda: float(np.finfo(np.float64).eps))
    eig_metric: Literal["log", "chordal", "angle", "modulus", "weighted"] = "log"
    eig_metric_weights: Optional[Dict[str, float]] = None
    vandermonde_weighting: Literal["none", "Savg"] = "none"
    failure_policy: Literal["nan", "raise"] = "nan"

    # --------------------------- API ---------------------------------------- #

    def compute_features(
        self,
        modes: NDArray[np.complexfloating],       # shape (D*L, M)
        eigenvalues: NDArray[np.complexfloating], # shape (M,)
        *,
        plot: bool = False,
        fit_dmd_kwargs: Optional[dict] = None,
        plotting_normalization: Optional[Literal["first_delay"]] = "first_delay",
    ) -> Dict[str, NDArray[np.floating]]:
        """
        Compute per-mode features and their scores.

        Returns
        -------
        dict with keys:
            - "Reconstruction"                : float32 scores (size M)
            - "Eigenvalue-Consistency"        : float32 scores (size M)
            - "Reconstruction_raw"            : float32 raw errors (size M)
            - "Eigenvalue-Consistency_raw"    : float32 raw distances (size M)
        """
        L, D = self.num_delays, self.spatial_dim
        if modes.ndim != 2 or modes.shape[0] != D * L:
            raise ValueError(f"`modes` expected shape (D*L, M) with D={D}, L={L}; got {modes.shape}.")
        if eigenvalues.ndim != 1 or eigenvalues.shape[0] != modes.shape[1]:
            raise ValueError("`eigenvalues` must be shape (M,) matching modes' column count.")

        M = modes.shape[1]
        fit_dmd_kwargs = fit_dmd_kwargs or {}

        # Reshape (DL, M) -> (M, D, L)
        mode_mats = modes.T.reshape(M, L, D).transpose(0, 2, 1).astype(np.complex128)  # (M, D, L)

        # Optional visualization (no arbitrary constants; use data-aware checks)
        if plot:
            plot_matrices_list(
                mode_mats,
                title="Mode Sub-Matrices",
                x_label="Delay Index",
                y_label="Spatial Dimension",
            )
            if plotting_normalization == "first_delay":
                baseline = mode_mats[:, :, [0]]  # (M, D, 1)
                # Normalize only where baseline is non-zero; else leave unchanged
                mask = np.abs(baseline) > 0.0
                normalized = np.where(mask, mode_mats / baseline, mode_mats)
                plot_matrices_list(
                    normalized,
                    title="Mode Sub-Matrices (Normalized by first delay where defined)",
                    x_label="Delay Index",
                    y_label="Spatial Dimension",
                )

        # Fit rank-1 DMD per mode-matrix
        fitted = []
        for i in range(M):
            try:
                fitted.append(fit_dmd(mode_mats[i], svd_rank=1, **fit_dmd_kwargs))
            except Exception as exc:
                if self.failure_policy == "raise":
                    raise
                fitted.append(None)

        # Reconstruction error (relative Frobenius)
        X = mode_mats.reshape(M, -1)
        norms = np.linalg.norm(X, axis=1)  # ||X||_F per mode
        # Define zero-norm cases: relative error is 0 if both X and reconstruction are zero; else 1
        recon_err = np.full(M, np.nan, dtype=np.float64)
        for i in range(M):
            f = fitted[i]
            if f is None or not hasattr(f, "reconstructed_data"):
                recon_err[i] = np.nan
                continue
            R = np.asarray(f.reconstructed_data, dtype=np.complex128)
            num = np.linalg.norm((mode_mats[i] - R).ravel())
            if norms[i] == 0.0:
                recon_err[i] = 0.0 if np.linalg.norm(R.ravel()) == 0.0 else 1.0
            else:
                recon_err[i] = num / norms[i]

        # Eigenvalue consistency (configurable metric)
        est_eigs = np.array(
            [
                (getattr(f, "eigs")[0] if (f is not None and hasattr(f, "eigs") and len(getattr(f, "eigs")) >= 1)
                 else np.nan + 1j * np.nan)
                for f in fitted
            ],
            dtype=np.complex128,
        )

        metric_kwargs = self.eig_metric_weights or {}
        eig_err = eig_distance(
            eigenvalues.astype(np.complex128),
            est_eigs,
            metric=self.eig_metric,
            **metric_kwargs,
        )  # shape (M,)

        # Optional Vandermonde-aware scaling (scale-sensitive; set "none" to avoid)
        if self.vandermonde_weighting == "Savg":
            alpha_bar = 0.5 * (np.log(np.abs(eigenvalues)) + np.log(np.abs(est_eigs)))
            # If either eigenvalue is nan/inf, alpha_bar becomes nan; keep propagation
            scale = np.sqrt(vandermonde_scale(alpha_bar, L))
            eig_err = scale * eig_err

        # Convert raw errors to scores via -log(⋅ + epsilon).
        # epsilon is explicit (no magic constants). NaNs remain NaNs.
        eps = float(self.epsilon)
        if not (np.isfinite(eps) and eps > 0.0):
            raise ValueError("`epsilon` must be a finite positive float.")
        recon_score = -np.log(recon_err + eps)
        eig_score = -np.log(eig_err + eps)

        # Return float32 to keep downstream memory light
        return {
            "Reconstruction": recon_score.astype(np.float32),
            "Eigenvalue-Consistency": eig_score.astype(np.float32),
            "Reconstruction_raw": recon_err.astype(np.float32),
            "Eigenvalue-Consistency_raw": eig_err.astype(np.float32),
        }
