import numpy as np
from numpy.typing import NDArray
from utils.visualizations import scatter_scores_1d


class EstimatedSubspaceLeakage:
    """
    Estimated Subspace Leakage (ESL) for DMD mode selection.
    
    Computes how much each mode "leaks" outside the estimated subspace.
    Formula: leakage_squared = ||mode||^2 - |eigenvalue|^2
    Score: -log(leakage_squared + epsilon)  [larger = better]
    
    True modes have small leakage, spurious modes have large leakage.
    
    Parameters
    ----------
    epsilon : float, default 1e-12
        Small positive value for numerical stability in log.
    """
    
    def __init__(self, epsilon: float = 1e-12):
        if epsilon <= 0 or not np.isfinite(epsilon):
            raise ValueError(f"epsilon must be positive and finite, got {epsilon}")
        self.epsilon = float(epsilon)
    
    def compute_features(
        self,
        exact_modes: NDArray[np.complexfloating],  # shape (D*L, M)
        eigenvalues: NDArray[np.complexfloating],  # shape (M,)
        plot: bool = False,
    ) -> dict[str, NDArray[np.floating]]:
        """
        Compute leakage scores for each mode.
        
        Parameters
        ----------
        exact_modes : array (D*L, M)
            Exact DMD modes (columns are modes).
        eigenvalues : array (M,)
            Corresponding eigenvalues.
        plot : bool
            Whether to plot scores.
            
        Returns
        -------
        dict with "Estimated-Subspace-Leakage" -> scores of shape (M,)
        """
        # Validate shapes
        if exact_modes.ndim != 2:
            raise ValueError(f"exact_modes must be 2D, got shape {exact_modes.shape}")
        if eigenvalues.ndim != 1:
            raise ValueError(f"eigenvalues must be 1D, got shape {eigenvalues.shape}")
        
        num_modes = exact_modes.shape[1]
        if eigenvalues.shape[0] != num_modes:
            raise ValueError(
                f"Shape mismatch: {num_modes} modes but {eigenvalues.shape[0]} eigenvalues"
            )
        
        # Compute leakage: ||mode||^2 - |eigenvalue|^2
        mode_norm_squared = np.sum(np.abs(exact_modes) ** 2, axis=0)
        eigenvalue_magnitude_squared = np.abs(eigenvalues) ** 2
        leakage_squared = mode_norm_squared - eigenvalue_magnitude_squared
        
        # Clip to avoid numerical negatives
        leakage_squared = np.maximum(leakage_squared, 0.0)
        
        # Convert to score: larger is better
        scores = -np.log(leakage_squared + self.epsilon)
        
        if plot:
            scatter_scores_1d(
                scores,
                "Estimated-Subspace-Leakage",
                title="ESL Scores (larger = more true-like)",
                show_id=True,
            )
        
        return {"Estimated-Subspace-Leakage": scores.astype(np.float32)}
