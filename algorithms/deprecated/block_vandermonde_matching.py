from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from utils.dmd_utils import fit_dmd
from utils.visualizations import (
    scatter_scores_2d,
    plot_matrices_list,
)


def block_vandermonde_mode_rel_rms(mode_matrices, eigenvalue_ests, tol=1e-12):
    """
    Compute deviation of mode matrices from ideal block-Vandermonde form.

    Parameters
    ----------
    mode_matrices : ndarray, shape (M, D, L)
        Each (D, L) matrix is a mode candidate composed of L blocks of dimension D.
    eigenvalue_ests : array-like, shape (M,)
        The expected geometric ratio λ for each mode (complex allowed).
    tol : float
        Threshold to guard against degenerate inputs.

    Returns
    -------
    dev : ndarray, shape (M,)
        Relative RMS deviation from ideal form.
        0.0 == perfect match, higher == worse, np.inf == degenerate.
    """
    M, D, L = mode_matrices.shape
    ratios = np.asarray(eigenvalue_ests, dtype=mode_matrices.dtype)

    powers = np.arange(L)
    V = ratios[:, None] ** powers  # shape (M, L)
    denom = np.sum(np.abs(V) ** 2, axis=1)  # shape (M,)

    # Least-squares estimate of q: (M, D)
    numer = np.einsum("ml,mdl->md", V.conj(), mode_matrices)
    q_hat = np.where(denom[:, None] > tol, numer / denom[:, None], 0.0)

    # Reconstruct ideal block-Vandermonde matrices: (M, D, L)
    recon = q_hat[:, :, None] * V[:, None, :]

    # Compute residuals
    resid = mode_matrices - recon
    err2 = np.sum(np.abs(resid) ** 2, axis=(1, 2))
    norm2 = np.sum(np.abs(mode_matrices) ** 2, axis=(1, 2))

    dev = np.where(norm2 > tol, np.sqrt(err2 / norm2), np.inf)
    return dev, q_hat, recon


class StructureModeEvaluator:
    def __init__(
        self,
        num_delays: int,
        spatial_dim: int,
    ) -> None:
        self.L = num_delays  # delays
        self.D = spatial_dim

    def compute_scores(
        self,
        modes: NDArray[np.complexfloating],  # shape (D·L, M)
        eigenvalues: NDArray[np.complexfloating],
        singular_values_for_scaling: NDArray[np.floating] = None,
        plot: bool = False,
        force_nyquist_validity: bool = False,
        **fit_dmd_kwargs,
    ) -> dict[str, NDArray[np.floating]]:
        M = modes.shape[1]
        if singular_values_for_scaling is not None:
            num_sing_vals = singular_values_for_scaling.shape[0]
            assert (
                num_sing_vals == M
            ), f"{M} modes given, but {num_sing_vals} leading singular values given."
            modes = modes @ np.diag(singular_values_for_scaling)

        mode_matrices = modes.T.reshape(M, self.L, -1)  # (M, L, D)
        mode_matrices = mode_matrices.transpose(0, 2, 1)  # (M, D, L)

        if plot:
            plot_matrices_list(
                mode_matrices,
                title="Mode Sub-Matrices",
                x_label="Delay Index",
                y_label="Spatial Dimension",
            )
            plot_matrices_list(
                mode_matrices / mode_matrices[:, :, 0, None],
                title="Mode Sub-Matrices (Normalized)",
                x_label="Delay Index",
                y_label="Spatial Dimension",
            )

        block_vandermonde_rel_rms, _, closest_bv_modes = block_vandermonde_mode_rel_rms(
            mode_matrices=mode_matrices, eigenvalue_ests=eigenvalues
        )
        block_vandermonde_rms_score = -np.log(block_vandermonde_rel_rms + 1e-15)

        # Mode Mat DMD
        nyquist_bounds = self._nyquist_bounds(eigenvalues)
        mode_matrices = (
            [mode_mat[:, : b + 1] for mode_mat, b in zip(mode_matrices, nyquist_bounds)]
            if force_nyquist_validity
            else mode_matrices
        )

        fitted_dmds = [
            fit_dmd(mode_mat, svd_rank=1, **fit_dmd_kwargs)
            for mode_mat in mode_matrices
        ]
        dot = np.sum(
            (closest_bv_modes.conj() * mode_matrices).real, axis=(1, 2)
        )  # shape (M,)
        mode_norms = np.linalg.norm(mode_matrices, axis=(1, 2))
        closest_bv_norms = np.linalg.norm(closest_bv_modes, axis=(1, 2))
        cos_dist = 0.5 * (1.0 - dot / (mode_norms * closest_bv_norms + 1e-15))
        block_vandermonde_cos_score = -np.log(cos_dist + 1e-15)

        recon_error = np.array(
            [
                np.mean(np.abs(mode_matrices[i] - fitted_dmds[i].reconstructed_data))
                for i in range(M)
            ]
        )

        eig_diff = np.array(
            [np.abs(eigenvalues[i] - fitted_dmds[i].eigs[0]) for i in range(M)]
        )

        recon_score = -np.log(recon_error + 1e-15)
        eig_score = -np.log(eig_diff + 1e-15)

        if plot:
            scatter_scores_2d(
                np.stack((recon_score, eig_score), axis=1),
                ["Reconstruction", "Eigenvalue Consistency"],
                title="ModeMatDMD Scores",
                show_id=True,
            )
            scatter_scores_2d(
                np.stack(
                    (block_vandermonde_rms_score, block_vandermonde_cos_score), axis=1
                ),
                ["BV-RMS-Score", "BV-Cosine-Score"],
                title="Block-Vandermonmde Template Matching",
                show_id=True,
            )

        score_dict = {
            "Block-Vandermonde-RMS": block_vandermonde_rms_score,
            "Block-Vandermonde-Cosine": block_vandermonde_cos_score,
            "Reconstruction": recon_score.astype(np.float32),
            "Eigenvalue-Consistency": eig_score.astype(np.float32),
        }

        return score_dict

    def _nyquist_bounds(
        self, eigenvalues: np.ndarray, eps: float = 1e-12
    ) -> NDArray[int]:
        """
        Returns per-mode largest usable lag index B_i for each eigenvalue λ_i.
        Use columns [:B_i+1] for the residual fit of mode i.
        """
        theta = np.abs(np.angle(eigenvalues))  # discrete angles in [0, pi]
        safe_theta = np.maximum(theta, eps)  # avoid div-by-zero.
        bounds = np.floor(np.pi / safe_theta).astype(int)  # per-mode Nyquist bounds
        bounds = np.where(theta < eps, self.L - 1, bounds)
        return np.clip(bounds, 0, self.L - 1)
