from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from numpy._typing import NDArray
from scipy import linalg as linalg


class DMD:
    """
    Dynamic Mode Decomposition (Exact or Projected variants).

    Parameters
    ----------
    variant : {"projected", "exact"}, default "projected"
        - "projected": Modes lie in the left singular vector space
        - "exact": Modes computed via pseudoinverse (higher-dimensional)
    svd_rank : int or None, default None
        - None: Use numeric rank of the data
        - int: Keep this many singular values (capped at numeric rank)
    scale_mode_by_eigenvalue : bool, default False
        (Exact variant only) Divide each mode by its eigenvalue.
    phase_align : bool, default False
        Make amplitudes real/positive by absorbing phase into modes.
    """

    def __init__(
        self,
        variant: str = "projected",
        svd_rank: int | None = None,
        scale_mode_by_eigenvalue: bool = False,
        phase_align: bool = True,
    ):
        if variant not in {"projected", "exact"}:
            raise ValueError("variant must be 'projected' or 'exact'")

        self.variant = variant
        self.svd_rank = svd_rank
        self.scale = scale_mode_by_eigenvalue
        self.phase_align = phase_align

        # Attributes set by fit()
        self.eigs: NDArray | None = None
        self.modes: NDArray | None = None
        self.amplitudes: NDArray | None = None
        self._num_snapshots: int = 0

        # SVD components (for inspection)
        self.U = None
        self.singular_values = None
        self.Vh = None

    def fit(self, data: NDArray):
        """
        Learn DMD from snapshot matrix.

        Parameters
        ----------
        data : array (spatial_dim, num_snapshots)
            Snapshot matrix (real or complex).

        Returns
        -------
        self
        """
        past_snapshots = data[:, :-1]
        future_snapshots = data[:, 1:]
        self._num_snapshots = data.shape[1]

        # Economy SVD
        left_vectors, singular_vals, right_vectors_conj_T = linalg.svd(
            past_snapshots, full_matrices=False
        )
        self.U = left_vectors
        self.singular_values = singular_vals
        self.Vh = right_vectors_conj_T

        # Determine numeric rank using same tolerance as np.linalg.matrix_rank
        epsilon = np.finfo(singular_vals.dtype).eps
        tolerance = epsilon * max(past_snapshots.shape) * singular_vals[0]
        numeric_rank = int(np.sum(singular_vals > tolerance))

        # Choose truncation rank
        if self.svd_rank is None:
            truncation_rank = numeric_rank
        else:
            if self.svd_rank > numeric_rank:
                warnings.warn(
                    f"Requested svd_rank={self.svd_rank} exceeds numeric rank {numeric_rank}; "
                    "using numeric rank instead.",
                    RuntimeWarning,
                )
                truncation_rank = numeric_rank
            else:
                truncation_rank = self.svd_rank

        # Truncate SVD components
        left_trunc = left_vectors[:, :truncation_rank]
        singular_trunc = singular_vals[:truncation_rank]
        right_trunc = right_vectors_conj_T.conj().T[:, :truncation_rank]
        singular_inv = np.diag(1 / singular_trunc)

        # Reduced operator (truncation_rank x truncation_rank)
        reduced_operator = (
            left_trunc.conj().T @ future_snapshots @ right_trunc @ singular_inv
        )

        # Eigen-decomposition of reduced operator
        eigenvalues, reduced_eigenvectors = linalg.eig(reduced_operator)

        # Compute full-space modes
        if self.variant == "projected":
            modes = left_trunc @ reduced_eigenvectors
        else:  # exact
            modes = future_snapshots @ right_trunc @ singular_inv @ reduced_eigenvectors
            if self.scale:
                modes = modes / eigenvalues

        # Compute amplitudes from initial condition
        amplitudes = linalg.lstsq(modes, past_snapshots[:, 0])[0]

        # Optional phase alignment
        if self.phase_align:
            phase_factors = np.exp(1j * np.angle(amplitudes))
            modes = modes * phase_factors
            amplitudes = np.abs(amplitudes)

        # Store results
        self.eigs = eigenvalues
        self.modes = modes
        self.amplitudes = amplitudes

        return self

    def reconstruct(self, timesteps: Sequence[int] | NDArray) -> NDArray:
        """
        Reconstruct state(s) at specified integer timesteps.

        Parameters
        ----------
        timesteps : sequence or array of int
            Zero-based timestep indices to reconstruct.

        Returns
        -------
        reconstruction : array (spatial_dim, num_timesteps)
            Reconstructed snapshots at requested timesteps.
        """
        timesteps_array = np.asarray(timesteps)
        # Dynamics: amplitudes * eigenvalues^t for each mode and time
        time_evolution = self.amplitudes[:, None] * (
            self.eigs[:, None] ** timesteps_array
        )
        return self.modes @ time_evolution

    def reconstructed_data(self) -> NDArray:
        """
        Rebuild the full training sequence.

        Returns
        -------
        reconstruction : array (spatial_dim, num_snapshots)
            Reconstructed snapshot matrix matching the training data shape.
        """
        return self.reconstruct(np.arange(self._num_snapshots))
