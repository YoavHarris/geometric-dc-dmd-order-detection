"""
ResDMD residual score for order detection.

Implements the per-eigenpair residual score from Residual DMD
(Colbrook, Ayton, Szoke, 2023) in the linear-dictionary case, adapted
to the same column-snapshot convention used elsewhere in this codebase.

For an eigenpair (lambda_j, g_j) of A_M^H (i.e., a left eigenvector of
the reduced DMD propagator A_M), the score is

    r_j = || X1^H g_j - lambda_j X0^H g_j ||^2  /  || X0^H g_j ||^2

Equivalently, in reduced coordinates with g_j = U_M * g_tilde_j,

    r_j = ( g_tilde_j^H Q g_tilde_j ) / ( g_tilde_j^H G0_tilde g_tilde_j )

where
    G0_tilde = Sigma_M^2
    G1_tilde = Sigma_M V_M^H X1^H U_M
    G2_tilde = U_M^H X1 X1^H U_M
    Q        = G2_tilde - lambda * G1_tilde^H - conj(lambda) * G1_tilde
                + |lambda|^2 * G0_tilde

See `notes/resdmd_mapping.pdf` for the derivation linking ResDMD's
formulation to the standard DMD pipeline used here.
"""

import numpy as np

from utils.delay_embedding import DelayEmbedding
from utils.visualizations import scatter_scores_1d


class ResDMDResidual:
    """
    Computes the ResDMD residual score per DMD eigenpair.

    Unlike post-hoc scores (e.g. ESR, eigenvalue magnitude) that only
    consume DMD outputs, this score requires access to the snapshot
    matrices X0 and X1, because it needs the third Galerkin matrix
    G2_tilde = U_M^H X1 X1^H U_M, which is not part of standard DMD
    output.

    The reduced DMD propagator A_M is taken directly from the fitted
    DMD object via `exact_dmd.operator.as_numpy_array`. Left eigenvectors
    of A_M (the objects ResDMD's score is defined on) are obtained as
    right eigenvectors of A_M^H.

    Parameters
    ----------
    num_delays : int
        Delay-embedding length used in the surrounding DMD pipeline.
        Must match the value used to fit the DMD object whose operator
        is passed to `compute_features`.
    eps : float
        Floor on the denominator to avoid division by zero in degenerate
        cases.

    Notes
    -----
    The eigenvalues returned by the internal eig solve match
    `exact_dmd.eigs` as a set (modulo conjugation for complex spectra),
    but their ordering is determined by `np.linalg.eig` and is not
    guaranteed to align entry-wise with `exact_dmd.eigs`. For order
    detection via clustering this is irrelevant, since clustering
    operates on the score values alone. If a per-eigenvalue alignment
    against another method's ordering is needed (e.g. for diagnostic
    plots), match via `scipy.optimize.linear_sum_assignment` on
    |lambda_resdmd - lambda_other|.
    """

    def __init__(self, num_delays: int = 1, eps: float = 1e-12):
        self.num_delays = num_delays
        self.eps = eps

    def compute_features(
        self,
        signal: np.ndarray,
        exact_dmd,
        plot: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Compute the ResDMD residual score for each DMD eigenpair.

        Parameters
        ----------
        signal : np.ndarray
            Raw multivariate signal of shape (D, N). Will be delay-
            embedded internally if `num_delays > 1`.
        exact_dmd : DMD object
            Fitted DMD object exposing `operator.as_numpy_array`
            (the reduced propagator A_M).
        plot : bool
            If True, produce a 1-D scatter plot of the scores.

        Returns
        -------
        dict[str, np.ndarray]
            {"ResDMDResidual": residuals}, where `residuals` has length M
            (the number of DMD eigenpairs / truncation rank).
        """
        # 1. Delay-embed if needed; build snapshot matrices X0, X1.
        if self.num_delays > 1:
            embedded = DelayEmbedding(num_delays=self.num_delays).transform(signal)
        else:
            embedded = signal
        X0 = embedded[:, :-1]
        X1 = embedded[:, 1:]

        # 2. Reduced propagator from the fitted DMD object.
        A_M = exact_dmd.operator.as_numpy_array
        M = A_M.shape[0]

        # 3. Truncated SVD of X0 at rank M, to recover U_M, Sigma_M, V_M.
        #    PyDMD performs this internally; recomputing here keeps the
        #    class self-contained and removes any coupling to library
        #    internals.
        U, s, Vh = np.linalg.svd(X0, full_matrices=False)
        if len(s) < M:
            raise ValueError(
                f"Truncation rank M={M} exceeds available singular values "
                f"({len(s)}). Check that exact_dmd was fit with svd_rank "
                f"compatible with this signal."
            )
        U_M = U[:, :M]
        S_M = s[:M]
        V_M = Vh[:M, :].conj().T  # shape (N_eff, M)

        # 4. Reduced Galerkin matrices (all M x M).
        # G0_tilde = Sigma_M^2
        G0_tilde = np.diag(S_M**2)
        # G1_tilde = Sigma_M V_M^H X1^H U_M
        G1_tilde = np.diag(S_M) @ (V_M.conj().T @ X1.conj().T @ U_M)
        # G2_tilde = U_M^H X1 X1^H U_M
        UM_H_X1 = U_M.conj().T @ X1  # shape (M, N_eff)
        G2_tilde = UM_H_X1 @ UM_H_X1.conj().T

        # 5. Left eigenvectors of A_M = right eigenvectors of A_M^H.
        eigvals, eigvecs = np.linalg.eig(A_M.conj().T)

        # 6. Per-eigenpair residual, vectorized across all eigenpairs.
        # For each j, we need q_A[j] = eigvecs[:, j]^H @ A @ eigvecs[:, j]
        # for A in {G0_tilde, G1_tilde, G2_tilde}.
        #
        # Trick: if E is the (M, M) matrix of eigenvectors-as-columns,
        # then A @ E has shape (M, M) and its j-th column is A @ eigvecs[:, j].
        # The j-th quadratic form is the inner product of conj(eigvecs[:, j])
        # with the j-th column of A @ E. Doing this for all j at once is
        # an elementwise multiply followed by a column-wise sum.
        def column_quadratic_forms(A, E):
            """Return length-M vector with entries E[:, j]^H @ A @ E[:, j]."""
            AE = A @ E  # shape (M, M)
            return (E.conj() * AE).sum(axis=0)  # shape (M,)

        qG0 = column_quadratic_forms(G0_tilde, eigvecs)
        qG1 = column_quadratic_forms(G1_tilde, eigvecs)
        qG2 = column_quadratic_forms(G2_tilde, eigvecs)
        # qG1H[j] = eigvecs[:, j]^H @ G1_tilde^H @ eigvecs[:, j]
        #         = conj(eigvecs[:, j]^H @ G1_tilde @ eigvecs[:, j])
        #         = qG1[j].conj()  --> avoids a fourth matmul.

        num = np.real(
            qG2
            - eigvals * qG1.conj()
            - np.conj(eigvals) * qG1
            + (np.abs(eigvals) ** 2) * qG0
        )
        den = np.real(qG0)

        # Numerator can pick up a tiny negative value from roundoff;
        # clamp at zero. Denominator is floor-protected by eps.
        residuals = np.maximum(num, 0.0) / np.maximum(den, self.eps)

        scores = -np.log(residuals + self.eps)

        if plot:
            scatter_scores_1d(
                scores,
                "ResDMD-scores",
                "ResDMDResidual",
                show_id=True,
            )

        return {"ResDMDResidual": scores.astype(np.float32)}
