# leakage_separation_utils.py

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def compute_leakage_projector(
    basis: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """
    Compute the leakage projector: I - P where P projects onto span(basis).

    Args:
        basis: Basis matrix (columns are basis vectors), shape (DL, rank).

    Returns:
        Orthogonal complement projector I - basis @ basis^H, shape (DL, DL).
    """
    DL = basis.shape[0]
    projector = basis @ basis.conj().T
    leakage_projector = np.eye(DL, dtype=complex) - projector
    return leakage_projector


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
        signal_basis: Orthonormal basis for true signal subspace, shape (DL, m).

    Returns:
        SSL value: ||（I - P_signal) @ mode||^2, where P_signal projects onto
        the signal subspace.
    """
    # Ensure signal basis is orthonormal
    signal_basis_orthonormal, _ = np.linalg.qr(signal_basis)

    # Compute leakage projector: I - P_signal
    leakage_projector = compute_leakage_projector(signal_basis_orthonormal)

    # Project mode onto signal complement and compute squared norm
    leakage = leakage_projector @ mode
    ssl = float(np.sum(np.abs(leakage) ** 2))

    return ssl


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

    Returns:
        ESL value: ||(I - P_est) @ mode||^2, where P_est projects onto
        the estimated subspace.
    """
    # Ensure estimated basis is orthonormal
    estimated_basis_orthonormal, _ = np.linalg.qr(estimated_basis)

    # Compute leakage projector: I - P_estimated
    leakage_projector = compute_leakage_projector(estimated_basis_orthonormal)

    # Project mode onto estimated complement and compute squared norm
    leakage = leakage_projector @ mode
    esl = float(np.sum(np.abs(leakage) ** 2))

    return esl


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
