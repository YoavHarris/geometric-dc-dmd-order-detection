from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Literal
from utils.visualizations import (
    plot_matrices_list,
    scatter_scores_1d,
    scatter_scores_2d,
)
from utils.dmd_utils import fit_dmd


ModeName = Literal["nested_dmd", "febvf"]


# -------------------------- helpers -------------------------- #


def _wrap_angle(delta: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Wrap angle differences to the range (-pi, pi].
    """
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def _relative_fro_error(
    X: NDArray[np.complexfloating], Xhat: NDArray[np.complexfloating]
) -> float:
    """
    Relative Frobenius-type error between two (D x L) complex arrays.
    """
    num = np.linalg.norm((X - Xhat).ravel())
    den = np.linalg.norm(X.ravel())
    if den == 0.0:
        return 0.0 if np.linalg.norm(Xhat.ravel()) == 0.0 else 1.0
    return float(num / den)


def _eig_distance_simple(
    ref: NDArray[np.complexfloating], est: NDArray[np.complexfloating]
) -> NDArray[np.floating]:
    """
    Eigenvalue distance designed for stability and interpretability:

      distance = normalized_angle_difference + magnitude_ratio_term

    - normalized_angle_difference:
        absolute wrapped phase difference divided by pi, so it is in [0, 1].
    - magnitude_ratio_term:
        2 * |mag1 - mag2| / (mag1 + mag2), which is scale-free and in [0, 2].
        Division is done safely; when both magnitudes are zero, the term is 0.

    Works elementwise on 1-D arrays of the same length.
    """
    ref = ref.astype(np.complex128)
    est = est.astype(np.complex128)

    # angle term in [0, 1]
    delta = _wrap_angle(np.angle(ref) - np.angle(est))
    angle_term = np.abs(delta) / np.pi

    # magnitude term with safe division
    mag1 = np.abs(ref)
    mag2 = np.abs(est)
    num = 2.0 * np.abs(mag1 - mag2)
    den = mag1 + mag2
    mag_term = np.divide(
        num, den, out=np.full_like(num, np.inf, dtype=np.float64), where=(den != 0)
    )
    mag_term = np.where((den == 0) & (num == 0), 0.0, mag_term)

    return (angle_term + mag_term).astype(np.float64)


# ------------------------ main class ------------------------- #


class ModeNestedDMD:
    """
    Feature extractor with two operation modes that share the same inputs/outputs:

    1) mode="nested_dmd"
       - Treat each mode vector (flattened length D*L) as a (D x L) "mode matrix".
       - Fit a rank-1 DMD to that (interpreting columns as short lagged samples).
       - Produce two per-mode features:
           a) Reconstruction: relative error between the mode matrix and its DMD-based reconstruction.
           b) Eigenvalue-Consistency: distance between the external eigenvalue and the one
              estimated by the inner DMD, using:
                  normalized wrapped phase difference (0..1)
                + scale-free magnitude difference term (0..2).
       - Scores are -log(feature + epsilon) for numerical stability and easier clustering.

    2) mode="febvf"
       - Fixed-Eigenvalue BV Fit (FEBVF).
       - Hold the external eigenvalue fixed and check how well the mode matrix follows
         a block-Vandermonde evolution consistent with that eigenvalue.
       - The fit reduces to one weighted summation across delays per mode, plus a closed-form
         normalization depending only on the eigenvalue magnitude and number of delays.
       - Output is a single per-mode residual and its score (-log(residual + epsilon)).

    - Plotting calls are optional and only for visualization.
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
        modes: NDArray[np.complexfloating],  # shape (D*L, M)
        eigenvalues: NDArray[np.complexfloating],  # shape (M,)
        plot: bool = False,
        **fit_dmd_kwargs,
    ) -> Dict[str, NDArray[np.floating]]:
        """
        Compute per-mode features and return them as a dictionary.

        Returns:
          - For mode="nested_dmd":
              {
                "Reconstruction",            # score
                "Eigenvalue-Consistency",    # score
                "Reconstruction_raw",        # raw metric
                "Eigenvalue-Consistency_raw" # raw metric
              }

          - For mode="febvf":
              {
                "BV-Fit",     # score
                "BV-Fit_raw"  # raw residual
              }
        """
        D, L = self.D, self.L
        if modes.ndim != 2 or modes.shape[0] != D * L:
            raise ValueError(f"`modes` must be (D*L, M) with D={D}, L={L}.")
        if eigenvalues.ndim != 1 or eigenvalues.shape[0] != modes.shape[1]:
            raise ValueError(
                "`eigenvalues` must be shape (M,) matching number of modes."
            )
        M = modes.shape[1]

        # Reshape (DL, M) -> (M, D, L)
        mode_mats = modes.T.reshape(M, L, -1).transpose(0, 2, 1).astype(np.complex128)

        if plot:
            plot_matrices_list(
                mode_mats,
                title="Mode Matrices",
                x_label="Delay Index",
                y_label="Spatial Dimension",
            )
            # Divide by the first delay where it is non-zero; otherwise leave as-is
            baseline = mode_mats[:, :, [0]]
            mask = np.abs(baseline) != 0.0
            normalized = np.where(mask, mode_mats / baseline, mode_mats)
            plot_matrices_list(
                normalized,
                title="Mode Matrices (Normalized by first delay where defined)",
                x_label="Delay Index",
                y_label="Spatial Dimension",
            )

        if self.mode == "nested_dmd":
            features = self._compute_nested(mode_mats, eigenvalues, **fit_dmd_kwargs)
            if plot:
                # stack two 1-D arrays into an (M, 2) matrix for 2D scatter
                vals = np.stack(
                    [features["Reconstruction"], features["Eigenvalue-Consistency"]],
                    axis=1,
                )
                scatter_scores_2d(
                    vals,
                    score_names=["Reconstruction", "Eigenvalue-Consistency"],
                    title="Mode-Nested-DMD Scores",
                    show_id=True,
                )

        elif self.mode == "febvf":
            features = self._compute_febvf(mode_mats, eigenvalues)
            if plot:
                scatter_scores_1d(
                    features["BV-Fit"],
                    "BV-Fit-Score",
                    title="Fixed-Eigenvalue BV Fit",
                    show_id=True,
                )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return features

    # ------------------ mode implementations ------------------ #

    def _compute_nested(
        self,
        mode_mats: NDArray[np.complexfloating],  # (M, D, L)
        eigenvalues: NDArray[np.complexfloating],  # (M,)
        **fit_dmd_kwargs,
    ) -> Dict[str, NDArray[np.floating]]:
        """
        Nested-DMD path:
        - Fit rank-1 DMD to each mode matrix independently.
        - Compute relative reconstruction error per mode.
        - Extract the first eigenvalue estimated by the fit and compare it
          to the external eigenvalue using the simple distance defined above.
        """
        fitted = [fit_dmd(X, svd_rank=1, **fit_dmd_kwargs) for X in mode_mats]

        # Reconstruction errors (vectorized post-fit where possible)
        recon_err = np.array(
            [
                (
                    _relative_fro_error(mode_mats[i], f.reconstructed_data)
                    if hasattr(f, "reconstructed_data")
                    else np.nan
                )
                for i, f in enumerate(fitted)
            ],
            dtype=np.float64,
        )

        # Estimated eigenvalues (first from each fit)
        eigenvalues_est = np.array(
            [
                (
                    f.eigs[0]
                    if hasattr(f, "eigs") and len(getattr(f, "eigs", [])) >= 1
                    else np.nan + 1j * np.nan
                )
                for f in fitted
            ],
            dtype=np.complex128,
        )

        # Consistency distance (vectorized)
        eig_err = _eig_distance_simple(
            eigenvalues.astype(np.complex128), eigenvalues_est
        )

        eps = self.epsilon
        return {
            "Reconstruction": (-np.log(recon_err + eps)).astype(np.float32),
            "Eigenvalue-Consistency": (-np.log(eig_err + eps)).astype(np.float32),
            "Reconstruction_raw": recon_err.astype(np.float32),
            "Eigenvalue-Consistency_raw": eig_err.astype(np.float32),
        }

    def _compute_febvf(
        self,
        mode_mats: NDArray[np.complexfloating],  # (M, D, L)
        eigenvalues: NDArray[np.complexfloating],  # (M,)
    ) -> Dict[str, NDArray[np.floating]]:
        """
        Fixed-Eigenvalue BV Fit (vectorized across all modes):

        For each mode matrix:
        - Build a weight vector from the conjugate eigenvalue powers along the delay axis.
        - Multiply the mode matrix by this weight vector (one weighted sum across delays).
        - Compare the squared norm of the result with a geometric-series normalization
          that depends only on the eigenvalue magnitude and the number of delays.
        - Residual is 1 minus this normalized energy; score is -log(residual + epsilon).
        """
        M, D, L = mode_mats.shape
        eigenvalues = eigenvalues.astype(np.complex128)

        # Weighted sum across delays using conjugate eigenvalue powers
        weights = np.conjugate(eigenvalues)[:, None] ** np.arange(L)[None, :]  # (M, L)
        summed = (mode_mats @ weights[..., None])[..., 0]  # (M, D)

        # Numerator: squared norms of the weighted sums (per mode)
        num_sq = np.sum(np.abs(summed) ** 2, axis=1).astype(np.float64)  # (M,)

        # Denominator: geometric series in the squared magnitude
        r2 = np.abs(eigenvalues) ** 2  # (M,)
        den = np.where(
            r2 == 1.0,
            float(L),
            (1.0 - r2**L) / (1.0 - r2),
        ).astype(
            np.float64
        )  # (M,)

        residual = 1.0 - num_sq / den
        residual = np.maximum(residual, 0.0)  # clip tiny negatives

        eps = self.epsilon
        score = -np.log(residual + eps)

        return {
            "BV-Fit": score.astype(np.float32),
            "BV-Fit_raw": residual.astype(np.float32),
        }
