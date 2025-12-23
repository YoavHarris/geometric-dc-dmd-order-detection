# leakage_separation_utils.py

from __future__ import annotations
import warnings
import numpy as np
from numpy.typing import NDArray


def compute_leakage_projector(
    orthonormal_basis: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """
    Compute the leakage projector: I - P where P projects onto span(basis).

    Args:
        orthonormal_basis: Orthonormal basis matrix (columns are basis vectors), shape (DL, rank).

    Returns:
        Orthogonal complement projector I - basis @ basis^H, shape (DL, DL).

    Raises:
        ValueError: If the input basis is not orthonormal.
    """
    DL, rank = orthonormal_basis.shape

    # Verify orthonormality
    gram = orthonormal_basis.conj().T @ orthonormal_basis
    if not np.allclose(gram, np.eye(rank), atol=1e-10):
        raise ValueError("Basis must be orthonormal for P = BB^H formula.")

    projector = orthonormal_basis @ orthonormal_basis.conj().T
    leakage_projector = np.eye(DL, dtype=complex) - projector
    return leakage_projector


def _compute_subspace_leakage(
    mode: NDArray[np.complexfloating],
    orthonormal_basis: NDArray[np.complexfloating],
) -> float:
    """
    Compute subspace leakage given an orthonormal basis.

    Measures how much of the mode lies outside the given subspace.
    This is the squared norm of the component of the mode orthogonal to
    the subspace.

    Args:
        mode: DMD mode vector, shape (DL,).
        orthonormal_basis: Orthonormal basis for subspace, shape (DL, rank).
            MUST be orthonormal (not checked for performance).

    Returns:
        Subspace leakage: ||(I - P) @ mode||^2, where P projects onto the subspace.
    """
    # Compute leakage projector: I - P
    leakage_projector = compute_leakage_projector(orthonormal_basis)

    # Project mode onto subspace complement and compute squared norm
    leakage = leakage_projector @ mode
    return float(np.sum(np.abs(leakage) ** 2))


def compute_ssl(
    mode: NDArray[np.complexfloating],
    signal_basis: NDArray[np.complexfloating],
) -> float:
    """
    Compute Signal Subspace Leakage (SSL).

    SSL measures how much of the mode lies outside the true signal subspace.
    It is the squared norm of the component of the mode orthogonal to the
    signal subspace.

    Args:
        mode: DMD mode vector, shape (DL,).
        signal_basis: Basis for true signal subspace, shape (DL, m).
            Will be orthonormalized internally.

    Returns:
        SSL value: ||（I - P_signal) @ mode||^2, where P_signal projects onto
        the signal subspace.
    """
    # Orthonormalize signal basis (may not be orthonormal initially)
    signal_basis_orthonormal, _ = np.linalg.qr(signal_basis)

    return _compute_subspace_leakage(mode, signal_basis_orthonormal)


def compute_esl(
    mode: NDArray[np.complexfloating],
    estimated_basis: NDArray[np.complexfloating],
) -> float:
    """
    Compute Estimated Subspace Leakage (ESL).

    ESL measures how much of the mode lies outside the estimated subspace
    (computed from noisy data). It is the squared norm of the component of
    the mode orthogonal to the estimated subspace.

    Args:
        mode: DMD mode vector, shape (DL,).
        estimated_basis: Orthonormal basis for estimated subspace, shape (DL, M).
            Should already be orthonormal from SVD (not re-orthonormalized for efficiency).

    Returns:
        ESL value: ||(I - P_est) @ mode||^2, where P_est projects onto
        the estimated subspace.
    """
    # Estimated basis from SVD is already orthonormal, use directly
    return _compute_subspace_leakage(mode, estimated_basis)


def compute_exact_mode_norm(mode: NDArray[np.complexfloating]) -> float:
    """
    Compute the squared norm of a mode vector.

    Args:
        mode: Mode vector, shape (DL,).

    Returns:
        Squared norm: ||mode||^2.
    """
    return float(np.sum(np.abs(mode) ** 2))


# =============================================================================
# Subspace gap and bound computation
# =============================================================================


def compute_directed_gap(
    basis_A: NDArray[np.complexfloating],
    basis_B: NDArray[np.complexfloating],
) -> float:
    """
    Compute directed gap delta(A,B) = ||(I - P_B)P_A||_2.

    This measures how much of subspace A is not captured by subspace B.

    Args:
        basis_A: Basis matrix for subspace A, shape (DL, rank_A).
        basis_B: Basis matrix for subspace B, shape (DL, rank_B).

    Returns:
        Directed gap: largest singular value of (I - P_B)P_A.
    """
    r_A = basis_A.shape[1]
    r_B = basis_B.shape[1]

    # Degenerate regime: theoretical delta(A,B) = 1 exactly
    if r_A > r_B:
        warnings.warn(
            f"compute_directed_gap: dim(A)={r_A} > dim(B)={r_B}. "
            "Returning delta(A,B)=1.0 (degenerate direction).",
            RuntimeWarning,
        )
        return 1.0

    # Orthonormalize both bases
    Q_A, _ = np.linalg.qr(basis_A)
    Q_B, _ = np.linalg.qr(basis_B)

    # Compute residual: A - B(B^H A)
    # This is equivalent to (I - P_B)P_A applied to basis A
    residual = Q_A - Q_B @ (Q_B.conj().T @ Q_A)

    # Return largest singular value
    sigma = np.linalg.svd(residual, compute_uv=False)
    return float(sigma[0]) if len(sigma) > 0 else 0.0


def compute_delta_tail(
    signal_basis: NDArray[np.complexfloating],
    basis_m: NDArray[np.complexfloating],
    basis_M: NDArray[np.complexfloating],
) -> float:
    """
    Compute delta_tail(M) = ||(I - P_S) P_{U_tail}||_2.

    This measures the tail overestimation factor when using rank M > m.

    Args:
        signal_basis: Orthonormal basis for true signal subspace, shape (DL, m).
        basis_m: Rank-m estimated basis, shape (DL, m).
        basis_M: Rank-M estimated basis (M > m), shape (DL, M).

    Returns:
        Tail overestimation factor.
    """
    # Orthonormalize all bases
    Q_S, _ = np.linalg.qr(signal_basis)
    Q_m, _ = np.linalg.qr(basis_m)
    Q_M, _ = np.linalg.qr(basis_M)

    # Extract U_tail = orthonormal_basis(U_M - U_m(U_m^H U_M))
    # This is the component of U_M orthogonal to U_m
    U_tail_unnormalized = Q_M - Q_m @ (Q_m.conj().T @ Q_M)

    # Check if U_tail is effectively zero
    if np.linalg.norm(U_tail_unnormalized) < 1e-14:
        return 0.0

    # Orthonormalize U_tail
    Q_tail, _ = np.linalg.qr(U_tail_unnormalized)

    # Compute residual: U_tail - S(S^H U_tail) = (I - P_S) U_tail
    residual = Q_tail - Q_S @ (Q_S.conj().T @ Q_tail)

    # Return largest singular value
    sigma = np.linalg.svd(residual, compute_uv=False)
    return float(sigma[0]) if len(sigma) > 0 else 0.0


# =============================================================================
# Subspace basis computation helpers
# =============================================================================


def make_vandermonde_matrix(
    eigenvalues: NDArray[np.complexfloating], num_delays: int
) -> NDArray[np.complexfloating]:
    """
    Matrix where each column is a Vandermonde vector for the corresponding eigenvalue.
    Example column: [1, eigenvalues[0], eigenvalues[0]^2, ..., eigenvalues[0]^(num_delays-1)]
    """
    exponents = np.arange(num_delays, dtype=np.int64)  # (num_delays,)
    return (
        eigenvalues[:, None] ** exponents[None, :]
    ).transpose()  # (num_delays, num_modes)


def orthonormal_basis(
    matrix: NDArray[np.complexfloating], rank: int | None = None
) -> NDArray[np.complexfloating]:
    """
    Compute orthonormal basis for col(matrix) via reduced QR.
    Optionally trim to specified rank.
    """
    if matrix.size == 0:
        raise ValueError("Matrix is empty.")
    col_norms = np.linalg.norm(matrix, axis=0)
    keep = col_norms > (1e-14 * np.max(col_norms))
    mat = matrix[:, keep] if np.any(keep) else matrix
    Q, _ = np.linalg.qr(mat, mode="reduced")
    if rank is not None:
        rank = min(rank, Q.shape[1])
        Q = Q[:, :rank]
    return Q


def compute_estimated_basis(
    embedded_data: NDArray[np.complexfloating], num_modes: int
) -> NDArray[np.complexfloating]:
    """
    Compute estimated subspace basis from SVD of noisy embedded data.
    Returns first num_modes left singular vectors.
    """
    U, _, _ = np.linalg.svd(embedded_data, full_matrices=False)
    return U[:, :num_modes]


def compute_practical_basis(
    embedded_noise: NDArray[np.complexfloating],  # (LD, T)
    clean_bv_modes: NDArray[np.complexfloating],  # (LD, m)
    eigenvalues: NDArray[np.complexfloating],  # (m,)
    num_modes: int,
) -> NDArray[np.complexfloating]:
    """
    Compute practical BV basis following the paper's formula (equations 31-36).
    Absorbs noise that's aligned with the exponential temporal structures of the signal into the modes.

    Args:
        embedded_noise: Delay-embedded noise, shape (LD, T)
        clean_bv_modes: Clean block-Vandermonde modes, shape (LD, m)
        eigenvalues: True eigenvalues, shape (m,)
        num_modes: Number of modes

    Returns:
        Orthonormal basis for the perturbed signal subspace, shape (LD, num_modes)
    """
    LD, T = embedded_noise.shape

    # Vandermonde matrix in TIME: (T, m)
    Psi = make_vandermonde_matrix(eigenvalues, T)

    # Least-squares projection of noise onto temporal basis: (LD, m)
    Q = embedded_noise @ Psi.conj() @ np.linalg.inv(Psi.T @ Psi.conj())

    # Perturbed modes: add noise perturbation
    perturbed_bv = clean_bv_modes + Q  # (LD, m)

    # Normalize each column (vectorized)
    norms = np.linalg.norm(perturbed_bv, axis=0, keepdims=True) + 1e-16
    practical_bv = perturbed_bv / norms

    return orthonormal_basis(practical_bv, num_modes)


# =============================================================================
# Proxy signal subspace selection
# =============================================================================


def select_proxy_signal_subspace(
    signal_subspace_basis: NDArray[np.complexfloating],
    estimated_subspace_basis: NDArray[np.complexfloating],
    *,
    assume_orthonormal: bool = True,
    return_distance: bool = True,
) -> dict[str, any]:
    """
    Select m columns from estimated M-column basis that best align with signal subspace.
    
    Uses exhaustive search over all C(M, m) combinations to minimize spectral
    projector distance ||P_S - P_{U_J}||_2 between signal subspace and selected
    columns.
    
    Efficient scoring per subset uses:
        C = S^T U  (m x M)
        For J: C_J = C[:, J] (m x m)
               G_J = C_J^T C_J
        sigma_min(C_J)^2 = lambda_min(G_J)
    Minimize distance <=> maximize sigma_min(C_J).
    
    Args:
        signal_subspace_basis: Basis for true signal subspace, shape (DL, m).
        estimated_subspace_basis: Basis for estimated subspace, shape (DL, M).
        assume_orthonormal: If True, assume inputs are already orthonormal.
                           If False, orthonormalize internally.
        return_distance: If True, include spectral projector distance in output.
    
    Returns:
        Dictionary with:
            - 'selected_indices': tuple of m column indices from estimated basis
            - 'proxy_basis': selected m columns, shape (DL, m)
            - 'tail_basis': remaining M-m columns, shape (DL, M-m)
            - 'sigma_min': smallest singular value of overlap matrix for best subset
            - 'distance': spectral projector distance (if return_distance=True)
    """
    from itertools import combinations
    
    S = np.asarray(signal_subspace_basis, dtype=complex)
    U = np.asarray(estimated_subspace_basis, dtype=complex)
    
    if S.ndim != 2 or U.ndim != 2:
        raise ValueError("Inputs must be 2D arrays.")
    DL_s, m = S.shape
    DL_u, M = U.shape
    if DL_s != DL_u:
        raise ValueError(f"Row mismatch: S is {DL_s}x{m}, U is {DL_u}x{M}.")
    if m > M:
        raise ValueError(f"Need m <= M, got m={m}, M={M}.")
    
    if not assume_orthonormal:
        S, _ = np.linalg.qr(S)
        U, _ = np.linalg.qr(U)
    
    # Precompute overlap matrix (m x M)
    C = S.conj().T @ U
    
    best_indices = None
    best_lambda_min = -np.inf  # maximize
    
    # Iterate all subsets of size m
    for J in combinations(range(M), m):
        C_J = C[:, J]                 # (m, m)
        G_J = C_J.conj().T @ C_J      # (m, m), symmetric PSD
        # Smallest eigenvalue is sigma_min(C_J)^2
        eigvals = np.linalg.eigvalsh(G_J)
        lam_min = float(eigvals[0])
        if lam_min > best_lambda_min:
            best_lambda_min = lam_min
            best_indices = J
    
    # Recover sigma_min and build output
    sigma_min = float(np.sqrt(max(best_lambda_min, 0.0)))
    
    # Extract selected columns and tail columns
    all_indices = set(range(M))
    tail_indices = sorted(all_indices - set(best_indices))
    
    proxy_basis = U[:, best_indices]
    tail_basis = U[:, tail_indices] if tail_indices else np.zeros((DL_u, 0), dtype=complex)
    
    result = {
        "selected_indices": best_indices,
        "proxy_basis": proxy_basis,
        "tail_basis": tail_basis,
        "sigma_min": sigma_min,
    }
    
    if return_distance:
        # distance = sqrt(1 - sigma_min^2) = sin(theta_max)
        dist = float(np.sqrt(max(0.0, 1.0 - sigma_min * sigma_min)))
        result["distance"] = dist
    
    return result
