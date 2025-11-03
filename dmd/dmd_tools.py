"""dmd_tools.py
=================
Minimal, transparent implementation of
  • DelayEmbedding
  • Dynamic Mode Decomposition (DMD)

MIT License · 2025‑05‑07 · OpenAI o3
"""

from __future__ import annotations

import logging
import warnings
from typing import Sequence, Optional

import numpy as np
from numpy.typing import NDArray

from utils.data_generation import DMDDataGenerator

try:
    import scipy.linalg as la  # preferred backend (fast & numerically robust)
except ModuleNotFoundError:  # pragma: no cover
    import numpy.linalg as la  # type: ignore


def fit_dmd(
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
    data : ndarray, shape (D, N)
        Snapshot matrix (real or complex).
    svd_rank : int
        Requested rank for the SVD truncation.
    mode : {'exact', 'projected'}, default 'projected'
        Which DMD algorithm variant to use.
    num_delays : int, default 1
        L = number of delays (1 ⇒ no delay embedding).
    **kwargs
        Forwarded to the DMD class constructor
        (e.g. scale_mode_by_eigenvalue, phase_align).

    Returns
    -------
    dmd : DMD
        Fitted DMD object.  If ``num_delays > 1`` an attribute
        ``dmd.delay_embedding`` is attached so you can invert the
        reconstruction back to the original snapshot space:

            H_hat = dmd.reconstructed_data()
            X_hat = dmd.delay_embedding.inverse_transform(H_hat)
    """
    assert mode in {"exact", "projected"}, "mode must be 'exact' or 'projected'"

    dmd = DMD(variant=mode, svd_rank=svd_rank, **kwargs)

    # ------------------------------------------------------------------
    # Optional delay embedding
    # ------------------------------------------------------------------
    if num_delays > 1:
        emb = DelayEmbedding(num_delays)
        data_to_fit = emb.transform(data)
        dmd.delay_embedding = emb  # stash for later use
    else:
        data_to_fit = data

    # ------------------------------------------------------------------
    # Suppress INFO chatter during fitting
    # ------------------------------------------------------------------
    logging.disable(logging.INFO)
    dmd.fit(data_to_fit)
    logging.disable(logging.NOTSET)

    return dmd


# ---------------------------------------------------------------------
# Phase-alignment helper
# ---------------------------------------------------------------------
def align_modes_and_amplitudes_phases(
    modes: NDArray, amplitudes: NDArray
) -> tuple[NDArray, NDArray]:
    """
    Make amplitudes real/positive by pushing their phases into the modes.

    Given Φ (modes) and b (amplitudes) we produce
        Φ' = Φ · diag(e^{i·arg(b)})
        b' = |b| ≥ 0
    so that  Φ·b == Φ'·b'  element-wise.

    Returns
    -------
    modes_aligned : NDArray
        Modes with embedded phase (same shape as `modes`).
    amplitudes_fixed : NDArray
        Real, non-negative amplitudes |b|.
    """
    phase = np.exp(1j * np.angle(amplitudes))
    modes_aligned = modes * phase  # broadcast rows × cols
    amplitudes_fixed = np.abs(amplitudes)

    # Sanity check (optional, could comment out for speed)
    assert np.allclose(modes @ amplitudes, modes_aligned @ amplitudes_fixed)

    return modes_aligned, amplitudes_fixed


# ---------------------------------------------------------------------
# Collapse Hankel back to snapshots (stand-alone utility)
# ---------------------------------------------------------------------


def collapse_hankel(
    H: NDArray, D: int, L: int, N: int, *, agg: str = "mean"
) -> NDArray:
    K = N - L + 1
    H3 = H.reshape(L, D, K)
    X = np.zeros((D, N), dtype=H.dtype)

    if agg in ("mean", "sum"):
        counts = np.zeros(N, int)
        for delay in range(L):
            X[:, delay : delay + K] += H3[delay]
            counts[delay : delay + K] += 1
        if agg == "mean":
            X /= counts
        return X

    if agg in ("first", "last"):
        order = range(L) if agg == "first" else range(L - 1, -1, -1)
        filled = np.zeros(N, bool)
        for delay in order:
            mask = ~filled[delay : delay + K]
            if mask.any():
                X[:, delay + np.where(mask)[0]] = H3[delay, :, mask]
                filled[delay : delay + K][mask] = True
        return X

    raise ValueError("agg must be 'mean', 'sum', 'first', or 'last'")


# -------- DelayEmbedding class ---------------------------------------------
class DelayEmbedding:
    """Stacked delay-embedding with reversible inverse_transform."""

    def __init__(self, L: int):
        if L < 1:
            raise ValueError("L must be ≥ 1")
        self.L = L
        self._shape: tuple[int, int] | None = None  # (D, N)

    # ---- forward ----------------------------------------------------------
    def transform(self, X: NDArray, *, copy: bool = False) -> NDArray:
        """
        Returns a zero-copy Hankel view matching pyDMD’s row order:
            – outer axis 0 … L-1  → delay index
            – inner axis 0 … D-1  → spatial index
        Result shape: (L*D, K)  where K = N-L+1
        """
        D, N = X.shape
        K = N - self.L + 1
        if K < 1:
            raise ValueError("L cannot exceed N")

        H_view = np.lib.stride_tricks.as_strided(
            X,
            shape=(self.L, D, K),
            strides=(
                X.strides[1],  # delay advances one time-step → stride in columns
                X.strides[0],  # spatial axis uses row stride
                X.strides[1],
            ),  # time-advance inside each column
            writeable=False,
        ).reshape(self.L * D, K)

        self._shape = (D, N)
        return H_view.copy() if copy else H_view

    # ---- inverse ----------------------------------------------------------
    def inverse_transform(self, H: NDArray, *, agg: str = "mean") -> NDArray:
        """
        Collapse `H` back to snapshots using the stand-alone `collapse_hankel`.
        """
        if self._shape is None:
            raise RuntimeError("transform() must be called first")
        D, N = self._shape
        return collapse_hankel(H, D, self.L, N, agg=agg)


class DMD:
    """
    Dynamic Mode Decomposition (Exact or Projected).

    Parameters
    ----------
    variant : {'projected', 'exact'}, default 'projected'
        Algorithm 1 ('projected') or Algorithm 2 ('exact') from the DMD paper.
    svd_rank : {None, int}, default None
        None  → use numeric (effective) rank of X.
        int   → keep min(svd_rank, numeric_rank); warn if svd_rank > numeric_rank.
    scale_mode_by_eigenvalue : bool, default False
        (Exact variant only) multiply each mode by 1 / eigenvalue.
    phase_align : bool, default False
        Make amplitudes real/positive and absorb phase into the modes.
    """

    # ------------------------------------------------------------------ #
    # constructor
    # ------------------------------------------------------------------ #
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

        # learned quantities
        self.eigs: NDArray | None = None  # eigenvalues λ_i
        self.modes: NDArray | None = None  # modes Φ (D, r)
        self.amplitudes: NDArray | None = None  # amplitudes b_i
        self._n_snapshots: int = 0  # N
        self.U = None
        self.singular_values = None
        self.Vh = None

    # ------------------------------------------------------------------ #
    # core algorithm
    # ------------------------------------------------------------------ #
    def fit(self, X: NDArray):
        """
        Learn DMD from snapshot matrix X (shape D × N).

        After calling, attributes:
            eigs, modes, amplitudes
        """
        x_past, x_future = X[:, :-1], X[:, 1:]  # X0, X1
        self._n_snapshots = X.shape[1]

        # economy SVD of X0
        U, s, Vh = la.svd(x_past, full_matrices=False)
        self.U = U
        self.singular_values = s
        self.Vh = Vh

        # numeric rank m (same rule as la.matrix_rank)
        eps = np.finfo(s.dtype).eps
        tol = eps * max(x_past.shape) * s[0]
        m = int(np.sum(s > tol))

        # choose r according to your rule
        if self.svd_rank is None:
            r = m
        else:
            if self.svd_rank > m:
                warnings.warn(
                    f"Requested svd_rank={self.svd_rank} but numeric rank is {m}; "
                    "using numeric rank instead.",
                    RuntimeWarning,
                )
                r = m
            else:
                r = self.svd_rank

        # truncate SVD
        U_r = U[:, :r]
        s_r = s[:r]
        V_r = Vh.conj().T[:, :r]
        sigma_inv = np.diag(1 / s_r)  # Σ⁻¹

        # reduced operator Â (r × r)
        A_tilde = U_r.conj().T @ x_future @ V_r @ sigma_inv

        # eigen-decomposition
        eigvals, W = la.eig(A_tilde)  # λ_i, W

        if self.variant == "projected":  # Algorithm 1
            Phi = U_r @ W
        else:
            Phi = x_future @ V_r @ sigma_inv @ W
            if self.scale:
                Phi = Phi / eigvals

        amplitudes = la.lstsq(Phi, x_past[:, 0])[0]

        # optional phase alignment
        if self.phase_align:
            phase = np.exp(1j * np.angle(amplitudes))
            Phi *= phase
            amplitudes = np.abs(amplitudes)

        # store results
        self.eigs = eigvals
        self.modes = Phi
        self.amplitudes = amplitudes
        return self

    # ------------------------------------------------------------------ #
    # reconstruction helpers
    # ------------------------------------------------------------------ #
    def reconstruct(self, timesteps: Sequence[int] | NDArray) -> NDArray:
        """
        Reconstruct state(s) at integer timesteps (0-based).
        """
        t = np.asarray(timesteps)
        dynamics = self.amplitudes[:, None] * (self.eigs[:, None] ** t)
        return self.modes @ dynamics

    def reconstructed_data(self) -> NDArray:
        """
        Rebuild the full training sequence with the same shape as X.
        """
        return self.reconstruct(np.arange(self._n_snapshots))


# ----------------------------------------------------------------------
# Demo: low-rank signal → delay-embed → DMD → reconstruct
# ----------------------------------------------------------------------
def demo_dmd_pipeline(
    D: int = 6,  # spatial dim
    N: int = 100,  # timesteps
    r_true: int = 3,  # intrinsic (clean) rank / # modes
    L: int = 5,  # delays
    svd_rank: int = 5,  # what we *ask* DMD for
):
    print("\n==== Synthetic data generation ====")
    gen = DMDDataGenerator(
        eigenvalue_magnitude=0.95,
        frequency_separation=0.4,
        snr_db=10.0,
        random_seed=0,
    )
    X, X_clean, eigs, modes, amps = gen.generate(D, N, r_true)

    rank_X = np.linalg.matrix_rank(X)
    print(f"X shape {X.shape},  numeric rank = {rank_X}")

    # ------------------------------------------------------------------
    emb = DelayEmbedding(L)
    X_delayed = emb.transform(X)
    rank_delayed = np.linalg.matrix_rank(X_delayed)
    print(f"Delay-embedded shape {X_delayed.shape}, rank = {rank_delayed}")

    # ------------------------------------------------------------------
    dmd = DMD(
        variant="exact",
        svd_rank=svd_rank,
        scale_mode_by_eigenvalue=True,
        phase_align=True,
    ).fit(X_delayed)

    print("\nDMD learned:")
    print(f"  eigs.shape        = {dmd.eigs.shape}")
    print(f"  modes.shape       = {dmd.modes.shape}")
    print(f"  amplitudes.shape  = {dmd.amplitudes.shape}")

    # ------------------------------------------------------------------
    X_delayed_hat = dmd.reconstructed_data()
    rank_delayed_hat = np.linalg.matrix_rank(X_delayed_hat)
    print(
        f"\nReconstructed delayed shape {X_delayed_hat.shape}, "
        f"rank = {rank_delayed_hat}"
    )

    X_hat = emb.inverse_transform(X_delayed_hat, agg="mean")
    rank_X_hat = np.linalg.matrix_rank(X_hat)
    print(f"Reconstructed original shape {X_hat.shape},  rank = {rank_X_hat}")

    delayed_mae = np.abs(X_delayed_hat - X_delayed).mean()
    mae = np.abs(X_hat - X).mean()
    print(f"Delayed MAE: {delayed_mae}, MAE: {mae}")

    return X, X_delayed, X_delayed_hat, X_hat


# ----------------------------------------------------------------------
# Run the demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_dmd_pipeline()
