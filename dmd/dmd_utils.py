import logging
import warnings
import numpy as np
from numpy.typing import NDArray
from pydmd import DMD
from pydmd.preprocessing import hankel_preprocessing
from scipy.optimize import linear_sum_assignment

from pydmd.utils import pseudo_hankel_matrix


def fit_dmd(
    data: NDArray[np.complexfloating],
    svd_rank: int,
    mode: str = "projected",
    num_delays: int = 1,
    **kwargs,
) -> DMD:
    """
    Fit a DMD model with optional delay embedding.

    Returns:
        Fitted DMD model object.
    """
    assert mode in [
        "exact",
        "projected",
    ], "DMD mode must be either 'exact' or 'projected'."

    kwargs["exact"] = mode == "exact"
    dmd = DMD(svd_rank=svd_rank, **kwargs)

    if num_delays > 1:
        dmd = hankel_preprocessing(dmd, d=num_delays)

    logging.disable(logging.INFO)
    dmd.fit(data)
    logging.disable(logging.NOTSET)

    return dmd


def warn_if_rank_mismatch(
    X: NDArray,
    requested_rank: int,
    *,
    num_delays: int = 1,
    energy_tol: float = 0.999,
    sv_tol: float = 1e-12,
    verbose: bool = True,
) -> int:
    """
    Decide a *practical* rank for X and warn if `requested_rank` is larger.

    Parameters
    ----------
    X : ndarray, shape (D, N)
        Snapshot matrix (for Exact DMD pass X[:, :-1]).
    requested_rank : int
        The svd_rank you intend to give PyDMD.
    energy_tol : float
        Cumulative energy threshold (Σ σ_i^2 / Σ σ_tot^2) for "practical rank".
    sv_tol : float
        Absolute tolerance for "effectively zero" singular values.
    verbose : bool
        True → emit warnings; False → silent return.

    Returns
    -------
    practical_rank : int
        min(numeric_rank, energy_rank), i.e. what most users would call "true".
    """
    # economy SVD once; NumPy is plenty fast for D ≤ a few thousand
    if num_delays > 1:
        X = pseudo_hankel_matrix(X, d=num_delays)

    _, s, _ = np.linalg.svd(X, full_matrices=False)

    numeric_rank = np.linalg.matrix_rank(X)  # ε·max(m,n) rule
    energy_cumsum = np.cumsum(s**2) / np.sum(s**2)
    energy_rank = int(np.searchsorted(energy_cumsum, energy_tol) + 1)

    practical_rank = min(numeric_rank, energy_rank)

    if requested_rank > practical_rank and verbose:
        warnings.warn(
            f"svd_rank={requested_rank} exceeds practical rank={practical_rank} "
            f"(numeric={numeric_rank}, energy≥{energy_tol:.3f}→{energy_rank}); "
            "PyDMD will pad with zero singular values!",
            RuntimeWarning,
        )
    # Hard corner case: true zeros *inside* the non-padded range
    if s[practical_rank - 1] < sv_tol and verbose:
        warnings.warn(
            "Trailing singular values are below sv_tol "
            f"({sv_tol:g}); expect ill-conditioned eigenvalues.",
            RuntimeWarning,
        )

    return practical_rank


def align_by_eigs(
    values: NDArray,
    source_eigs: NDArray[np.complexfloating],
    target_eigs: NDArray[np.complexfloating],
    *,
    tol: float = 1e-6,
    name: str = "values",
) -> NDArray:
    """
    Reorder a per-eigenpair array so it follows ``target_eigs`` ordering.

    The ``source_eigs`` and ``target_eigs`` are assumed to be the same set of
    eigenvalues (modulo floating-point error and possibly a permutation), and
    ``values`` is keyed to ``source_eigs`` order. Returns ``values`` reordered
    so that ``out[j]`` corresponds to ``target_eigs[j]``.

    Matching is done via Hungarian assignment on ``|source - target|``, which
    is robust to small numerical error and unique up to ties. Raises a
    ``RuntimeWarning`` if the worst matched pair differs by more than
    ``tol * max(|target_eigs|, 1)``, which would indicate that the two sets
    are not actually the same eigenvalues.

    Parameters
    ----------
    values : ndarray, shape (M, ...)
        Per-eigenpair quantity to reorder. The first axis is the eigenpair
        axis; trailing axes are preserved.
    source_eigs : complex ndarray, shape (M,)
        Eigenvalues that ``values`` is currently keyed to.
    target_eigs : complex ndarray, shape (M,)
        Desired eigenvalue ordering.
    tol : float
        Relative tolerance for the alignment-quality warning.
    name : str
        Name of the quantity being aligned (used in the warning message only).

    Returns
    -------
    ndarray
        ``values`` reordered to the order of ``target_eigs``.
    """
    source_eigs = np.asarray(source_eigs)
    target_eigs = np.asarray(target_eigs)
    if source_eigs.shape != target_eigs.shape:
        raise ValueError(
            f"align_by_eigs: source/target shape mismatch "
            f"({source_eigs.shape} vs {target_eigs.shape})"
        )
    if values.shape[0] != source_eigs.shape[0]:
        raise ValueError(
            f"align_by_eigs: leading axis of values ({values.shape[0]}) "
            f"must match source_eigs length ({source_eigs.shape[0]})"
        )

    diff = np.abs(source_eigs[:, None] - target_eigs[None, :])
    row_ind, col_ind = linear_sum_assignment(diff)

    worst = float(diff[row_ind, col_ind].max())
    scale = max(float(np.abs(target_eigs).max()), 1.0)
    if worst > tol * scale:
        warnings.warn(
            f"align_by_eigs: poor alignment for {name!r}: worst matched pair "
            f"differs by {worst:.3e} (tol={tol * scale:.3e}). source and "
            f"target eigenvalues do not appear to be the same set.",
            RuntimeWarning,
        )

    out = np.empty_like(values)
    out[col_ind] = values[row_ind]
    return out


def align_modes_and_amplitudes_phases(
    modes: NDArray[np.complexfloating], amplitudes: NDArray[np.complexfloating]
) -> tuple[NDArray[np.complexfloating], NDArray[np.floating]]:
    """
    Align mode phases to ensure amplitudes are real and positive.

    Returns:
        aligned_modes: Phase-aligned modes.
        fixed_amplitudes: Real, positive amplitudes.
    """
    original_product = modes * amplitudes
    amplitude_angles = np.angle(amplitudes)
    fixed_amplitudes = np.abs(amplitudes)
    aligned_modes = modes * np.exp(1j * amplitude_angles)[None, :]

    aligned_product = aligned_modes * fixed_amplitudes
    assert np.allclose(
        original_product, aligned_product, atol=1e-8
    ), "Product mismatch!"

    return aligned_modes, fixed_amplitudes
