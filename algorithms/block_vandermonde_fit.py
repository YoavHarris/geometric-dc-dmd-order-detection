from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Dict
from abc import ABC, abstractmethod
from utils.visualizations import plot_matrices_list, scatter_scores_1d, scatter_scores_2d
from utils.dmd_utils import fit_dmd


# Helper functions
def wrap_angle(angles: NDArray[np.floating]) -> NDArray[np.floating]:
    """Wrap angles to (-pi, pi]."""
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


def eigenvalue_distance(
    reference: NDArray[np.complexfloating],
    estimate: NDArray[np.complexfloating]
) -> NDArray[np.floating]:
    """
    Compute distance between eigenvalues (element-wise).
    
    Distance = normalized_angle_difference + magnitude_ratio
    
    - Angle difference: |angle(ref) - angle(est)| / pi  in [0, 1]
    - Magnitude ratio: 2 * ||mag(ref)| - |mag(est)|| / (|mag(ref)| + |mag(est)|)  in [0, 2]
    """
    reference = reference.astype(np.complex128)
    estimate = estimate.astype(np.complex128)
    
    # Angle term (normalized to [0, 1])
    angle_diff = wrap_angle(np.angle(reference) - np.angle(estimate))
    angle_term = np.abs(angle_diff) / np.pi
    
    # Magnitude term (scale-free, in [0, 2])
    mag_ref = np.abs(reference)
    mag_est = np.abs(estimate)
    numerator = 2.0 * np.abs(mag_ref - mag_est)
    denominator = mag_ref + mag_est
    
    # Safe division
    mag_term = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=(denominator != 0)
    )
    
    return (angle_term + mag_term).astype(np.float64)


# Base class for shared functionality
class BlockVandermondeFit(ABC):
    """
    Base class for Block-Vandermonde fitting strategies.
    
    Each DMD mode is reshaped into a (spatial_dim x num_delays) matrix
    and analyzed for block-Vandermonde structure along the delay axis.
    
    Parameters
    ----------
    num_delays : int
        L, number of delays (columns in mode matrix).
    spatial_dim : int
        D, spatial dimension (rows in mode matrix).
    epsilon : float, default machine epsilon
        Regularization for log scores.
    """
    
    def __init__(
        self,
        num_delays: int,
        spatial_dim: int,
        epsilon: float = float(np.finfo(np.float64).eps),
    ):
        self.num_delays = int(num_delays)
        self.spatial_dim = int(spatial_dim)
        
        if epsilon <= 0 or not np.isfinite(epsilon):
            raise ValueError("epsilon must be positive and finite")
        self.epsilon = float(epsilon)
    
    def compute_features(
        self,
        modes: NDArray[np.complexfloating],  # shape (D*L, M)
        eigenvalues: NDArray[np.complexfloating],  # shape (M,)
        plot: bool = False,
        **kwargs,
    ) -> Dict[str, NDArray[np.floating]]:
        """
        Compute per-mode BV-fit features.
        
        Parameters
        ----------
        modes : array (D*L, M)
            DMD modes as columns.
        eigenvalues : array (M,)
            Corresponding eigenvalues.
        plot : bool
            Whether to plot results.
        **kwargs
            Additional arguments passed to strategy-specific methods.
            
        Returns
        -------
        Dictionary of feature_name -> scores (M,)
        """
        # Validate inputs
        if modes.ndim != 2 or modes.shape[0] != self.spatial_dim * self.num_delays:
            raise ValueError(
                f"modes must be (D*L, M) with D={self.spatial_dim}, L={self.num_delays}"
            )
        if eigenvalues.ndim != 1 or eigenvalues.shape[0] != modes.shape[1]:
            raise ValueError(
                f"eigenvalues must be shape (M,) matching number of modes"
            )
        
        # Reshape modes to matrices
        mode_matrices = self._reshape_modes_to_matrices(modes)
        
        if plot:
            self._plot_mode_matrices(mode_matrices)
        
        # Strategy-specific computation
        features = self._compute_strategy(mode_matrices, eigenvalues, **kwargs)
        
        if plot:
            self._plot_features(features)
        
        return features
    
    def _reshape_modes_to_matrices(
        self, modes: NDArray[np.complexfloating]
    ) -> NDArray[np.complexfloating]:
        """
        Reshape (D*L, M) modes to (M, D, L) mode matrices.
        
        Each mode becomes a (D x L) matrix for analysis along delay axis.
        """
        num_modes = modes.shape[1]
        return (
            modes.T
            .reshape(num_modes, self.num_delays, self.spatial_dim)
            .transpose(0, 2, 1)
            .astype(np.complex128)
        )
    
    @abstractmethod
    def _compute_strategy(
        self,
        mode_matrices: NDArray[np.complexfloating],  # (M, D, L)
        eigenvalues: NDArray[np.complexfloating],  # (M,)
        **kwargs,
    ) -> Dict[str, NDArray[np.floating]]:
        """Strategy-specific computation. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _plot_features(self, features: Dict[str, NDArray]):
        """Strategy-specific plotting. Must be implemented by subclasses."""
        pass
    
    # Shared plotting helper
    def _plot_mode_matrices(self, mode_matrices: NDArray[np.complexfloating]):
        """Plot mode matrices as heatmaps."""
        plot_matrices_list(
            mode_matrices,
            title="Mode Matrices",
            x_label="Delay Index",
            y_label="Spatial Dimension",
        )
        
        # Plot normalized version
        first_delay = mode_matrices[:, :, [0]]
        mask = np.abs(first_delay) != 0.0
        normalized = np.where(mask, mode_matrices / first_delay, mode_matrices)
        
        plot_matrices_list(
            normalized,
            title="Mode Matrices (normalized by first delay)",
            x_label="Delay Index",
            y_label="Spatial Dimension",
        )


# Concrete strategy 1: Nested DMD
class NestedDMD(BlockVandermondeFit):
    """
    Nested Rank-1 DMD strategy for Block-Vandermonde fitting.
    
    Fits rank-1 DMD to each mode matrix independently and computes:
    - Reconstruction error: how well rank-1 DMD captures the mode
    - Eigenvalue consistency: agreement between external and nested eigenvalue
    
    Implements Algorithm "BV-fit via nested rank-1 DMD" from Section 4.3
    (bv_structure_and_nested_DMD.tex, lines 360-385).
    """
    
    def _compute_strategy(
        self,
        mode_matrices: NDArray[np.complexfloating],  # (M, D, L)
        eigenvalues: NDArray[np.complexfloating],  # (M,)
        **fit_dmd_kwargs,
    ) -> Dict[str, NDArray[np.floating]]:
        """Fit rank-1 DMD to each mode matrix and extract features."""
        num_modes = mode_matrices.shape[0]
        
        # Fit DMD to each mode
        fits = [
            fit_dmd(mat, svd_rank=1, **fit_dmd_kwargs)
            for mat in mode_matrices
        ]
        
        # Extract reconstruction errors (MSE)
        recon_errors = np.array([
            np.mean(np.abs(mode_matrices[i] - fit.reconstructed_data) ** 2)
            if hasattr(fit, "reconstructed_data")
            else np.nan
            for i, fit in enumerate(fits)
        ], dtype=np.float64)
        
        # Extract nested eigenvalues
        nested_eigenvalues = np.array([
            fit.eigs[0] if hasattr(fit, "eigs") and len(fit.eigs) >= 1
            else np.nan + 1j * np.nan
            for fit in fits
        ], dtype=np.complex128)
        
        # Compute eigenvalue consistency
        consistency_errors = eigenvalue_distance(eigenvalues, nested_eigenvalues)
        
        # Convert to scores (larger = better)
        # Use 0.5 factor for MSE to effectively score by log(RMSE)
        recon_scores = -0.5 * np.log(recon_errors + self.epsilon)
        consistency_scores = -np.log(consistency_errors + self.epsilon)
        
        return {
            "Reconstruction": recon_scores.astype(np.float32),
            "Eigenvalue-Consistency": consistency_scores.astype(np.float32),
            "Reconstruction_raw": recon_errors.astype(np.float32),
            "Eigenvalue-Consistency_raw": consistency_errors.astype(np.float32),
        }
    
    def _plot_features(self, features: Dict[str, NDArray]):
        """Plot 2D scatter of nested DMD scores."""
        scores = np.stack([
            features["Reconstruction"],
            features["Eigenvalue-Consistency"]
        ], axis=1)
        
        scatter_scores_2d(
            scores,
            score_names=["Reconstruction", "Eigenvalue-Consistency"],
            title="Nested DMD Scores",
            show_id=True,
        )


# Concrete strategy 2: Fixed-Eigenvalue BV Fit
class FixedEigenvalueBVFit(BlockVandermondeFit):
    """
    Fixed-Eigenvalue Block-Vandermonde Fit (FEBVF).
    
    Uses external eigenvalue directly to compute BV structure fit.
    Single closed-form score based on weighted sum across delays.
    
    Faster than Nested DMD: O(D*L) vs O(D*L*min(D,L)).
    
    Implements simplified FEBVF from Section 4.3
    (bv_structure_and_nested_DMD.tex, lines 416-421).
    """
    
    def _compute_strategy(
        self,
        mode_matrices: NDArray[np.complexfloating],  # (M, D, L)
        eigenvalues: NDArray[np.complexfloating],  # (M,)
        **kwargs,
    ) -> Dict[str, NDArray[np.floating]]:
        """Closed-form BV fit using external eigenvalues."""
        num_modes, spatial_dim, num_delays = mode_matrices.shape
        eigenvalues = eigenvalues.astype(np.complex128)
        
        # Weight vector: conjugate(eigenvalue)^k for k=0..L-1
        delay_indices = np.arange(num_delays)
        weights = np.conjugate(eigenvalues)[:, None] ** delay_indices[None, :]
        
        # Weighted sum across delays
        weighted_sum = (mode_matrices @ weights[..., None]).squeeze(-1)
        
        # Numerator: ||weighted_sum||^2 for each mode
        numerator = np.sum(np.abs(weighted_sum) ** 2, axis=1)
        
        # Denominator: geometric series sum for |eigenvalue|^2
        magnitude_squared = np.abs(eigenvalues) ** 2
        denominator = np.where(
            magnitude_squared == 1.0,
            float(num_delays),
            (1.0 - magnitude_squared ** num_delays) / (1.0 - magnitude_squared)
        )
        
        # Residual: 1 - (energy captured / total energy)
        residuals = 1.0 - numerator / denominator
        residuals = np.maximum(residuals, 0.0)  # clip numerical negatives
        
        # Convert to scores
        scores = -np.log(residuals + self.epsilon)
        
        return {
            "BV-Fit": scores.astype(np.float32),
            "BV-Fit_raw": residuals.astype(np.float32),
        }
    
    def _plot_features(self, features: Dict[str, NDArray]):
        """Plot 1D scatter of FEBVF scores."""
        scatter_scores_1d(
            features["BV-Fit"],
            "BV-Fit-Score",
            title="Fixed-Eigenvalue BV Fit",
            show_id=True,
        )

