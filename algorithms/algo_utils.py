"""
algo_utils.py
=============

Utility functions for DMD algorithms.
"""

from typing import Dict
import numpy as np
from numpy.typing import NDArray


def subspace_stats(
    signal_basis: NDArray,
    estimated_basis: NDArray,
    num_components: int,
    eps: float = 1e-12
) -> Dict[str, np.ndarray | float]:
    """
    Compute subspace similarity statistics using principal angles.
    
    Parameters
    ----------
    signal_basis : array (n, k_signal)
        True signal subspace basis (not necessarily orthonormal).
    estimated_basis : array (n, k_estimated)
        Estimated subspace basis from SVD (orthonormal).
    num_components : int
        Number of true components (dimension of signal subspace).
    eps : float
        Tolerance for rank checks.
        
    Returns
    -------
    dict with:
        theta : array (num_components,)
            Principal angles in radians.
        overlap : float
            Mean of cos^2(theta), in [0, 1]. Higher = better overlap.
        chordal_distance : float
            ||sin(theta)||_2. Lower = better overlap.
        max_sine : float
            max(sin(theta)). Worst-case misalignment.
    """
    # Take only the relevant columns
    signal_k = signal_basis[:, :num_components]
    estimated_k = estimated_basis[:, :num_components]
    
    # Orthonormalize signal basis
    Q_signal, R = np.linalg.qr(signal_k)
    
    # Check for rank deficiency
    if np.abs(np.diag(R)).min() < eps:
        raise ValueError(
            "Signal basis is numerically rank-deficient; cannot compute subspace stats"
        )
    
    # Compute principal angles via SVD
    _, singular_values, _ = np.linalg.svd(Q_signal.conj().T @ estimated_k)
    cosines = np.clip(singular_values, 0.0, 1.0)
    angles = np.arccos(cosines)
    sines = np.sqrt(1.0 - cosines ** 2)
    
    return {
        "theta": angles,
        "overlap": float(np.mean(cosines ** 2)),
        "chordal_distance": float(np.linalg.norm(sines)),
        "max_sine": float(sines.max()),
    }
