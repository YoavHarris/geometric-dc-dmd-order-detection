# subspace_proximity_analysis.py

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def make_vandermonde_matrix(eigenvalues: NDArray[np.complex128], num_delays: int) -> NDArray[np.complex128]:
    """
    Matrix where each column is a Vandermonde vector for the corresponding eigenvalue.
    Example column: [1, eigenvalues[0], eigenvalues[0]^2, ..., eigenvalues[0]^(num_delays-1)]
    """
    exponents = np.arange(num_delays, dtype=np.int64)  # (num_delays,)
    return (eigenvalues[:, None] ** exponents[None, :]).transpose()  # (num_delays, num_modes)

def build_block_vandermonde_modes(
    spatial_modes: NDArray[np.complex128],
    eigenvalues: NDArray[np.complex128],
    num_delays: int
) -> NDArray[np.complex128]:
    """
    Constructs the block-Vandermonde modes for the given spatial modes and eigenvalues.
    """
    spatial_dim, num_modes = spatial_modes.shape
    
    # Vandermonde matrix: (num_delays, num_modes)
    vandermonde_matrix = make_vandermonde_matrix(eigenvalues, num_delays)
    
    # Batched outer product: for each mode k, compute outer(V[:,k], spatial[:,k])
    # Result shape: (num_delays, spatial_dim, num_modes)
    outer_stack = vandermonde_matrix[:, None, :] * spatial_modes[None, :, :]
    
    # Reshape to (num_delays * spatial_dim, num_modes)
    return outer_stack.reshape(num_delays * spatial_dim, num_modes)

def orthonormal_basis(matrix: NDArray[np.complex128], rank: int | None = None) -> NDArray[np.complex128]:
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


def principal_angles_between(U: NDArray[np.complex128], V: NDArray[np.complex128]) -> NDArray[np.float64]:
    """
    Compute principal angles (radians) between span(U) and span(V).
    U and V should have orthonormal columns.
    Returns angles in ascending order.
    """
    M = U.conj().T @ V
    singular_values = np.linalg.svd(M, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    return np.sort(np.arccos(singular_values))


# ---------------------------------------------------------------------
# Subspace basis computation
# ---------------------------------------------------------------------

def compute_estimated_basis(
    embedded_data: NDArray[np.complex128],
    num_modes: int
) -> NDArray[np.complex128]:
    """
    Compute estimated subspace basis from SVD of noisy embedded data.
    Returns first num_modes left singular vectors.
    """
    U, _, _ = np.linalg.svd(embedded_data, full_matrices=False)
    return U[:, :num_modes]


def compute_practical_basis(
    embedded_noise: NDArray[np.complex128],  # (LD, T)
    clean_bv_modes: NDArray[np.complex128],  # (LD, m)
    eigenvalues: NDArray[np.complex128],     # (m,)
    num_modes: int
) -> NDArray[np.complex128]:
    """
    Compute practical BV basis following the paper's formula (equations 31-36).
    
    Q = E_embedded @ conj(Psi_T) @ (Psi_T^T @ conj(Psi_T))^{-1}
    practical_mode_j = normalize(clean_bv_mode_j + q_j)
    """
    LD, T = embedded_noise.shape
    
    # Vandermonde matrix in TIME: (T, m)
    Psi_T = make_vandermonde_matrix(eigenvalues, T).T
    
    # Least-squares projection of noise onto temporal basis: (LD, m)
    Q = embedded_noise @ np.conj(Psi_T) @ np.linalg.inv(Psi_T.T @ np.conj(Psi_T))
    
    # Perturbed modes: add noise perturbation
    perturbed_bv = clean_bv_modes + Q  # (LD, m)
    
    # Normalize each column (vectorized)
    norms = np.linalg.norm(perturbed_bv, axis=0, keepdims=True) + 1e-16
    practical_bv = perturbed_bv / norms
    
    return orthonormal_basis(practical_bv, num_modes)


# ---------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------

def compute_subspace_principal_angles(
    embedded_data: NDArray[np.complex128],
    true_modes: NDArray[np.complex128],
    true_eigenvalues: NDArray[np.complex128],
    num_modes: int,
    embedded_noise: NDArray[np.complex128] | None = None,
    embedded_signal: NDArray[np.complex128] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """
    Compute principal angles between estimated subspace and two reference subspaces.
    
    Parameters
    ----------
    embedded_data : (num_delays * spatial_dim, num_samples)
        Noisy delay-embedded data.
    true_modes : (spatial_dim, num_modes)
        True spatial modes (columns).
    true_eigenvalues : (num_modes,)
        True eigenvalues.
    num_modes : int
        Number of signal modes.
    embedded_noise : (num_delays * spatial_dim, num_samples), optional
        Delay-embedded noise. If not provided, must supply embedded_signal.
    embedded_signal : (num_delays * spatial_dim, num_samples), optional
        Delay-embedded clean signal. Used to derive noise if embedded_noise not given.
    
    Returns
    -------
    dict with keys:
        'clean_subspace_angles' : principal angles to clean reference (radians, ascending)
        'practical_subspace_angles' : principal angles to practical reference (radians, ascending)
    
    Notes
    -----
    - Clean reference: block-Vandermonde span of true noise-free modes.
    - Practical reference: block-Vandermonde span with spatial modes adjusted by
      absorbing the component of noise aligned with each mode's temporal structure.
    """
    # Derive embedded noise if needed
    if embedded_noise is None:
        if embedded_signal is None:
            raise ValueError("Must provide either embedded_noise or embedded_signal.")
        if embedded_signal.shape != embedded_data.shape:
            raise ValueError("embedded_signal must have same shape as embedded_data.")
        embedded_noise = embedded_data - embedded_signal
    
    # Infer dimensions
    total_dim, num_samples = embedded_data.shape
    spatial_dim = true_modes.shape[0]
    if total_dim % spatial_dim != 0:
        raise ValueError(f"embedded_data rows ({total_dim}) must be multiple of spatial_dim ({spatial_dim}).")
    num_delays = total_dim // spatial_dim
    
    if true_modes.shape[1] != num_modes or true_eigenvalues.shape[0] != num_modes:
        raise ValueError(f"true_modes must be ({spatial_dim}, {num_modes}), "
                        f"true_eigenvalues must have length {num_modes}.")
    
    # Compute bases
    estimated_basis = compute_estimated_basis(embedded_data, num_modes)
    
    # Build clean BV modes (needed for both clean and practical bases)
    clean_bv_modes = build_block_vandermonde_modes(true_modes, true_eigenvalues, num_delays)
    clean_basis = orthonormal_basis(clean_bv_modes, num_modes)
    
    # Practical basis: clean BV modes + noise perturbation
    practical_basis = compute_practical_basis(
        embedded_noise, clean_bv_modes, true_eigenvalues, num_modes
    )
    
    # Compute principal angles
    clean_angles = principal_angles_between(estimated_basis, clean_basis)
    practical_angles = principal_angles_between(estimated_basis, practical_basis)
    
    return {
        "clean_subspace_angles": clean_angles,
        "practical_subspace_angles": practical_angles,
    }
