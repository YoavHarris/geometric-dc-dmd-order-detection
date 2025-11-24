# leakage_separation_tools.py

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
