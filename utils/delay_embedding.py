"""
dmd_tools.py
============

Minimal, transparent implementation of:
  - DelayEmbedding
  - Dynamic Mode Decomposition (DMD)

"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from dmd.custom_impl_dmd import DMD
from utils.data_generation import DMDDataGenerator

try:
    import scipy.linalg as linalg
except ModuleNotFoundError:  # pragma: no cover
    import numpy.linalg as linalg  # type: ignore


def fit_custom_dc_dmd(
    data: NDArray,
    svd_rank: int,
    mode: str = "projected",
    num_delays: int = 1,
    **kwargs,
) -> DMD:
    """
    Fit a Dynamic Mode Decomposition (DMD) model
    with optional delay embedding (Hankel stacking).

    Parameters
    ----------
    data : array (spatial_dim, num_snapshots)
        Snapshot matrix (real or complex).
    svd_rank : int
        Requested rank for the SVD truncation.
    mode : {"exact", "projected"}, default "projected"
        Which DMD algorithm variant to use.
    num_delays : int, default 1
        Number of delays for embedding (1 = no delay embedding).
    **kwargs
        Forwarded to the DMD class constructor
        (e.g. scale_mode_by_eigenvalue, phase_align).

    Returns
    -------
    dmd : DMD
        Fitted DMD object. If num_delays > 1, an attribute
        dmd.delay_embedding is attached so you can invert the
        reconstruction back to the original snapshot space:

            embedded_reconstruction = dmd.reconstructed_data()
            reconstruction = dmd.delay_embedding.inverse_transform(embedded_reconstruction)
    """
    if mode not in {"exact", "projected"}:
        raise ValueError("mode must be 'exact' or 'projected'")

    dmd = DMD(variant=mode, svd_rank=svd_rank, **kwargs)

    # Optional delay embedding
    if num_delays > 1:
        embedding = DelayEmbedding(num_delays)
        data_to_fit = embedding.transform(data)
        dmd.delay_embedding = embedding  # Store for later use
    else:
        data_to_fit = data

    # Suppress INFO logging during fitting
    logging.disable(logging.INFO)
    dmd.fit(data_to_fit)
    logging.disable(logging.NOTSET)

    return dmd


def align_modes_and_amplitudes_phases(
    modes: NDArray, amplitudes: NDArray
) -> tuple[NDArray, NDArray]:
    """
    Make amplitudes real/positive by pushing their phases into the modes.

    Given modes and amplitudes, we produce:
        modes_aligned = modes * diag(exp(i*angle(amplitudes)))
        amplitudes_real = |amplitudes| >= 0

    so that modes @ amplitudes == modes_aligned @ amplitudes_real element-wise.

    Returns
    -------
    modes_aligned : array
        Modes with embedded phase (same shape as input modes).
    amplitudes_real : array
        Real, non-negative amplitudes.
    """
    phase_factors = np.exp(1j * np.angle(amplitudes))
    modes_aligned = modes * phase_factors  # Broadcast across rows
    amplitudes_real = np.abs(amplitudes)

    # Sanity check (optional)
    assert np.allclose(modes @ amplitudes, modes_aligned @ amplitudes_real)

    return modes_aligned, amplitudes_real


def collapse_hankel(
    hankel_matrix: NDArray,
    spatial_dim: int,
    num_delays: int,
    num_snapshots: int,
    *,
    agg: str = "mean",
) -> NDArray:
    """
    Collapse Hankel (delay-embedded) matrix back to original snapshot space.

    Parameters
    ----------
    hankel_matrix : array (num_delays * spatial_dim, num_embedded_snapshots)
        Delay-embedded data matrix.
    spatial_dim : int
        Original spatial dimension.
    num_delays : int
        Number of delays used in embedding.
    num_snapshots : int
        Original number of snapshots (before embedding).
    agg : {"mean", "sum", "first", "last"}, default "mean"
        How to aggregate overlapping entries.

    Returns
    -------
    snapshots : array (spatial_dim, num_snapshots)
        Reconstructed snapshot matrix.
    """
    num_embedded_snapshots = num_snapshots - num_delays + 1

    # Reshape to (num_delays, spatial_dim, num_embedded_snapshots)
    reshaped = hankel_matrix.reshape(num_delays, spatial_dim, num_embedded_snapshots)

    snapshots = np.zeros((spatial_dim, num_snapshots), dtype=hankel_matrix.dtype)

    if agg in ("mean", "sum"):
        counts = np.zeros(num_snapshots, int)
        for delay_idx in range(num_delays):
            end_idx = delay_idx + num_embedded_snapshots
            snapshots[:, delay_idx:end_idx] += reshaped[delay_idx]
            counts[delay_idx:end_idx] += 1
        if agg == "mean":
            snapshots /= counts
        return snapshots

    if agg in ("first", "last"):
        order = range(num_delays) if agg == "first" else range(num_delays - 1, -1, -1)
        filled = np.zeros(num_snapshots, bool)
        for delay_idx in order:
            end_idx = delay_idx + num_embedded_snapshots
            unfilled_mask = ~filled[delay_idx:end_idx]
            if unfilled_mask.any():
                unfilled_indices = np.where(unfilled_mask)[0]
                snapshots[:, delay_idx + unfilled_indices] = reshaped[
                    delay_idx, :, unfilled_mask
                ]
                filled[delay_idx:end_idx][unfilled_mask] = True
        return snapshots

    raise ValueError("agg must be 'mean', 'sum', 'first', or 'last'")


class DelayEmbedding:
    """
    Delay-embedding (Hankel matrix) transformation with reversible inverse.

    Stacks consecutive time delays to create augmented state vectors.
    """

    def __init__(self, num_delays: int):
        if num_delays < 1:
            raise ValueError("num_delays must be >= 1")
        self.num_delays = num_delays
        self._original_shape: tuple[int, int] | None = None

    def transform(self, snapshots: NDArray, *, copy: bool = False) -> NDArray:
        """
        Create delay-embedded matrix (Hankel structure).

        Parameters
        ----------
        snapshots : array (spatial_dim, num_snapshots)
            Original snapshot matrix.
        copy : bool
            If True, return a copy instead of a view.

        Returns
        -------
        embedded : array (num_delays * spatial_dim, num_embedded_snapshots)
            Delay-embedded matrix where num_embedded_snapshots = num_snapshots - num_delays + 1.

        Notes
        -----
        Returns a zero-copy view matching pyDMD's row ordering:
            - Outer axis iterates over delays (0 to num_delays-1)
            - Inner axis iterates over spatial dimensions
        """
        spatial_dim, num_snapshots = snapshots.shape
        num_embedded_snapshots = num_snapshots - self.num_delays + 1

        if num_embedded_snapshots < 1:
            raise ValueError("num_delays cannot exceed num_snapshots")

        # Create strided view for zero-copy Hankel matrix
        embedded_view = np.lib.stride_tricks.as_strided(
            snapshots,
            shape=(self.num_delays, spatial_dim, num_embedded_snapshots),
            strides=(
                snapshots.strides[1],  # Delay advances one time-step
                snapshots.strides[0],  # Spatial axis
                snapshots.strides[1],  # Time advance within column
            ),
            writeable=False,
        ).reshape(self.num_delays * spatial_dim, num_embedded_snapshots)

        self._original_shape = (spatial_dim, num_snapshots)
        return embedded_view.copy() if copy else embedded_view

    def inverse_transform(
        self, embedded_data: NDArray, *, agg: str = "mean"
    ) -> NDArray:
        """
        Collapse delay-embedded data back to original snapshot space.

        Parameters
        ----------
        embedded_data : array (num_delays * spatial_dim, num_embedded_snapshots)
            Delay-embedded matrix to collapse.
        agg : {"mean", "sum", "first", "last"}, default "mean"
            How to aggregate overlapping entries.

        Returns
        -------
        snapshots : array (spatial_dim, num_snapshots)
            Reconstructed snapshot matrix.
        """
        if self._original_shape is None:
            raise RuntimeError("transform() must be called before inverse_transform()")

        spatial_dim, num_snapshots = self._original_shape
        return collapse_hankel(
            embedded_data, spatial_dim, self.num_delays, num_snapshots, agg=agg
        )


def demo_dcdmd_pipeline(
    spatial_dim: int = 6,
    num_snapshots: int = 100,
    true_rank: int = 3,
    num_delays: int = 5,
    svd_rank: int = 5,
):
    """
    Demonstrate DMD pipeline: generate data -> embed -> fit -> reconstruct.

    Parameters
    ----------
    spatial_dim : int
        Spatial dimension of snapshots.
    num_snapshots : int
        Number of time snapshots.
    true_rank : int
        Number of true modes in generated data.
    num_delays : int
        Number of delays for embedding.
    svd_rank : int
        Rank to request from DMD.
    """
    print("\n==== Synthetic data generation ====")
    generator = DMDDataGenerator(
        eigenvalue_magnitude=0.95,
        frequency_separation=0.4,
        snr_db=10.0,
        random_seed=0,
    )
    data, data_clean, true_eigs, true_modes, true_amps = generator.generate(
        spatial_dim, num_snapshots, true_rank
    )

    data_rank = np.linalg.matrix_rank(data)
    print(f"Data shape {data.shape}, numeric rank = {data_rank}")

    # Delay embedding
    embedding = DelayEmbedding(num_delays)
    embedded_data = embedding.transform(data)
    embedded_rank = np.linalg.matrix_rank(embedded_data)
    print(f"Embedded shape {embedded_data.shape}, rank = {embedded_rank}")

    # Fit DMD
    dmd = DMD(
        variant="exact",
        svd_rank=svd_rank,
        scale_mode_by_eigenvalue=True,
        phase_align=True,
    ).fit(embedded_data)

    print("\nDMD learned:")
    print(f"  eigenvalues shape  = {dmd.eigs.shape}")
    print(f"  modes shape        = {dmd.modes.shape}")
    print(f"  amplitudes shape   = {dmd.amplitudes.shape}")

    # Reconstruct
    embedded_reconstruction = dmd.reconstructed_data()
    embedded_recon_rank = np.linalg.matrix_rank(embedded_reconstruction)
    print(
        f"\nReconstructed embedded shape {embedded_reconstruction.shape}, "
        f"rank = {embedded_recon_rank}"
    )

    reconstruction = embedding.inverse_transform(embedded_reconstruction, agg="mean")
    recon_rank = np.linalg.matrix_rank(reconstruction)
    print(f"Reconstructed original shape {reconstruction.shape}, rank = {recon_rank}")

    # Errors
    embedded_mae = np.abs(embedded_reconstruction - embedded_data).mean()
    mae = np.abs(reconstruction - data).mean()
    print(f"\nEmbedded MAE: {embedded_mae:.6f}")
    print(f"Original space MAE: {mae:.6f}")

    return data, embedded_data, embedded_reconstruction, reconstruction


if __name__ == "__main__":
    demo_dcdmd_pipeline()
