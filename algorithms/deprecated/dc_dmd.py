from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from algorithms.clustering import ModeClustering


def cluster_modes(score_dict, normalization_mode="min_max", decider="vote", 
                   pilot_feature="", weights=None):
    """Backward compatibility wrapper for deprecated code."""
    clusterer = ModeClustering(
        normalization=normalization_mode,
        strategy=decider,
        pilot_feature=pilot_feature,
        weights=weights or {},
    )
    return clusterer(score_dict)


class DcDMDModeSelector:
    def __init__(self, num_delays: int, dt: float = 1.0) -> None:
        """
        Mode selector leveraging spatiotemporal coupling in delay-coordinates DMD.

        Based on: Bronstein et al., Chaos 32, 123127 (2022)
        DOI: https://doi.org/10.1063/5.0123101
        """
        self.num_delays = num_delays
        self.dt = dt
        self.omega_s = 2 * np.pi / dt

    def label_modes(
        self,
        eigenvalues: NDArray[np.complexfloating],
        modes: NDArray[np.complexfloating],
        numerical_inf: float = 1e12,
    ) -> NDArray[np.bool_]:
        """
        Select true modes based on consistency in their spatiotemporal coupling.

        Returns:
            Boolean array marking true modes.
        """
        spatial_dim = modes.shape[0] // self.num_delays
        modes_batch = modes.T
        total_num_modes = eigenvalues.shape[0]

        oscillations = np.abs(np.log(eigenvalues)) / self.dt
        bounds = self._compute_bounds(oscillations)

        submodes_batch = modes_batch.reshape(
            total_num_modes, self.num_delays, spatial_dim
        )
        submodes_0 = submodes_batch[:, 0, :]
        divided_by_first_sub_mode = submodes_batch[:, 1:, :] / submodes_0[:, None, :]

        exponents = 1 / (np.arange(1, self.num_delays))[None, :, None]
        lambda_tilde_batch = divided_by_first_sub_mode**exponents

        lambda_tilde_averaged = np.empty(total_num_modes, dtype=eigenvalues.dtype)
        epsilon_errors = np.empty(total_num_modes, dtype=np.float32)

        for mode_idx in range(total_num_modes):
            lambda_tilde = lambda_tilde_batch[mode_idx]
            b = bounds[mode_idx]
            if b == 0:
                lambda_tilde_averaged[mode_idx] = 0
                epsilon_errors[mode_idx] = numerical_inf
                continue
            lambda_tilde_averaged[mode_idx] = np.mean(lambda_tilde[:b, :])
            epsilon_errors[mode_idx] = np.abs(
                lambda_tilde_averaged[mode_idx] - eigenvalues[mode_idx]
            )

        mode_labels = cluster_modes(
            {"log_error": -np.log10(epsilon_errors)},
            normalization_mode=None,
        )

        return mode_labels

    def _compute_bounds(self, oscillations: NDArray[np.floating]) -> NDArray[np.int32]:
        bounds = np.minimum(
            self.num_delays - 1, np.floor(0.5 * self.omega_s / oscillations)
        )
        bounds = np.where(oscillations == 0, 0, bounds)
        return bounds.astype(int)
