"""
Data collection: Spurious eigenvalue magnitudes vs embedding length L.

This script collects spurious eigenvalue magnitudes for different embedding lengths L
to demonstrate how they move toward the unit circle as L increases.

Design:
- For each L value, run n_mc Monte Carlo iterations
- Generate data with fixed parameters (SNR, freq_sep, eig_mag, D, N, M, m, etc.)
- Run DMD with rank M to get M modes (m true + M-m spurious)
- Identify spurious modes using Signal Subspace Leakage (SSL)
- Collect spurious eigenvalue magnitudes
- Save to CSV with all parameters for each row

The collected data can then be plotted using the plotting tools in
figures/spurious_eigs_L_plotting/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import yaml
import fire
from tqdm import tqdm
from scipy.linalg import qr  # scipy implementation to allow pivoting

from utils.data_generation import DMDDataGenerator
from dmd.dmd_utils import fit_dmd
from utils.delay_embedding import DelayEmbedding


# =============================================================================
# Utility Functions (Pure, Reusable)
# =============================================================================


def classify_modes_by_ssr(
    recovered_modes: NDArray[np.complexfloating],
    signal_basis: NDArray[np.complexfloating],
    num_true_modes: int,
) -> NDArray[np.str_]:
    """
    Classify modes as true or spurious using Signal Subspace Leakage (SSL).

    Modes with lowest SSL (most aligned with signal subspace) are classified as true.
    The remaining modes are classified as spurious.

    Args:
        recovered_modes: All recovered DMD modes, shape (DL, M).
        signal_basis: Basis for true signal subspace (delay-embedded true modes),
            shape (DL, m).
        num_true_modes: Number of true modes (m).

    Returns:
        Mode labels array of shape (M,), with values "true" or "spurious".
    """
    M = recovered_modes.shape[1]
    Q, _ = qr(signal_basis, mode="economic")

    # 1. Total energy of the new modes
    mode_square_energies = np.linalg.norm(recovered_modes, axis=0) ** 2

    # 2. Energy captured by the signal subspace
    in_signal_subspace_energies = (
        np.linalg.norm(Q.conj().T @ recovered_modes, axis=0) ** 2
    )
    # 3. Residual energy (Difference)
    ssr_energies = np.maximum(0, mode_square_energies - in_signal_subspace_energies)

    # Sort modes by SSL (ascending)
    sorted_indices = np.argsort(ssr_energies)

    # First num_true_modes with lowest SSL are true, rest are spurious
    mode_labels = np.array(["spurious"] * M, dtype=object)
    mode_labels[sorted_indices[:num_true_modes]] = "true"

    return mode_labels


def select_subspaces_via_pivoted_qr(
    U_M: NDArray[np.complexfloating],
    true_modes_embedded: NDArray[np.complexfloating],
    m: int,
) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.complexfloating]]:
    """
    Select U_m and U_tail from U_M using pivoted QR.

    Uses pivoted QR on Q^H @ U_M to identify the best m-subset of U_M
    that aligns with the true signal subspace Q. The remaining M-m columns
    form U_tail (the spurious subspace).

    Args:
        U_M: First M left singular vectors of embedded snapshot matrix, shape (DL, M).
        true_modes_embedded: True embedded modes, shape (DL, m).
        m: Number of true modes.

    Returns:
        Tuple of (I_m, I_tail, U_tail) where:
        - I_m: Column indices of U_M that best align with signal subspace, shape (m,).
        - I_tail: Remaining column indices (spurious subspace), shape (M-m,).
        - U_tail: The spurious subspace U_M[:, I_tail], shape (DL, M-m).
    """
    M = U_M.shape[1]

    # Compute Q: orthonormal basis of true signal subspace
    Q, _ = qr(true_modes_embedded, mode="economic")

    # Pivoted QR to select columns
    # G = Q^H @ U_M, shape (m, M)
    G = Q.conj().T @ U_M

    # Pivoted QR on G (not G.T!) to choose informative columns
    # With full mode, piv will be a permutation of all M columns
    _, _, piv = qr(G, pivoting=True, mode="economic")

    # First m indices are I_m (best aligned), rest are I_tail
    I_m = piv[:m]
    I_tail = piv[m:]

    # Extract U_tail
    U_tail = U_M[:, I_tail]

    return I_m, I_tail, U_tail


def compute_subspace_boundary_norms(
    U_subspace: NDArray[np.complexfloating],
    D: int,
    L: int,
) -> tuple[float, float]:
    """
    Compute mu_L and nu_L: spectral norms of first and last D-blocks of a subspace.

    Args:
        U_subspace: Subspace basis matrix, shape (DL, rank).
        D: Spatial dimension.
        L: Embedding length.

    Returns:
        Tuple of (mu_L, nu_L) where:
        - mu_L: Spectral norm of first D rows (U[0:D, :]).
        - nu_L: Spectral norm of last D rows (U[(L-1)*D:L*D, :]).
    """
    U_first = U_subspace[0:D, :]
    U_last = U_subspace[(L - 1) * D : L * D, :]

    mu_L = float(np.linalg.norm(U_first, 2))
    nu_L = float(np.linalg.norm(U_last, 2))

    return mu_L, nu_L


def _compute_reduced_propagator(
    U_M: NDArray[np.complexfloating],
    S_M: NDArray[np.floating],
    V_M: NDArray[np.complexfloating],
    X_emb_1: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """
    Compute the reduced propagator A_M in the truncated SVD basis.

    Uses the formula: A_M = (U_M^H @ X_emb_1 @ V_M) * (1.0 / S_M)
    where the division is applied via broadcasting on columns.

    Args:
        U_M: First M left singular vectors, shape (DL, M).
        S_M: First M singular values, shape (M,).
        V_M: First M right singular vectors, shape (ncols, M).
        X_emb_1: Second embedded snapshot matrix X[:, 1:], shape (DL, ncols).

    Returns:
        Reduced propagator matrix A_M, shape (M, M).
    """
    # Compute G = U_M^H @ X_emb_1 @ V_M
    G = U_M.conj().T @ X_emb_1 @ V_M

    # Scale columns by 1/S_M using broadcasting
    A_M = G * (1.0 / S_M)[None, :]

    return A_M


def _compute_coupling_metrics(
    A_M: NDArray[np.complexfloating],
    I_m: NDArray[np.int_],
    I_tail: NDArray[np.int_],
) -> tuple[float, float, float]:
    """
    Compute coupling metrics from the reduced propagator A_M.

    Computes three metrics:
    1. max_coupling_norm: Maximum spectral norm of off-diagonal blocks
    2. distance_to_spectrum: Minimum distance between diagonal block spectra (0 if spectra touch)
    3. resolvent_norm: Maximum resolvent norm over tail eigenvalues (inf if resolvent explodes)

    Args:
        A_M: Reduced propagator matrix, shape (M, M).
        I_m: Indices of signal subspace in A_M, shape (m,).
        I_tail: Indices of spurious subspace in A_M, shape (M-m,).

    Returns:
        Tuple of (max_coupling_norm, distance_to_spectrum, resolvent_norm).
        Returns NaN for metrics that cannot be computed (empty blocks or eigenvalue failure).
        Returns inf for resolvent_norm if resolvent explodes.
    """
    m = len(I_m)
    M_minus_m = len(I_tail)

    # Handle edge cases: empty blocks
    if m == 0 or M_minus_m == 0:
        return np.nan, np.nan, np.nan

    # Extract blocks using advanced indexing
    A_mm = A_M[np.ix_(I_m, I_m)]
    A_tail = A_M[np.ix_(I_tail, I_tail)]
    Gamma12 = A_M[np.ix_(I_m, I_tail)]
    Gamma21 = A_M[np.ix_(I_tail, I_m)]

    # =========================================================================
    # 1. Compute max_coupling_norm
    # =========================================================================
    if Gamma12.size > 0:
        svals_12 = np.linalg.svdvals(Gamma12)
        g12 = float(svals_12[0]) if len(svals_12) > 0 else 0.0
    else:
        g12 = 0.0

    if Gamma21.size > 0:
        svals_21 = np.linalg.svdvals(Gamma21)
        g21 = float(svals_21[0]) if len(svals_21) > 0 else 0.0
    else:
        g21 = 0.0

    max_coupling_norm = max(g12, g21)

    # =========================================================================
    # 2. Compute distance_to_spectrum
    # =========================================================================
    try:
        eig_mm = np.linalg.eigvals(A_mm)
        eig_tail = np.linalg.eigvals(A_tail)

        # Check for NaN eigenvalues - this indicates failure
        if np.any(np.isnan(eig_mm)) or np.any(np.isnan(eig_tail)):
            return np.nan, np.nan, np.nan

        # Vectorized pairwise distance computation
        # dist_mat[i, j] = |eig_tail[i] - eig_mm[j]|
        dist_mat = np.abs(eig_tail[:, None] - eig_mm[None, :])
        distance_to_spectrum = float(dist_mat.min())  # Will be 0 if spectra touch

    except np.linalg.LinAlgError:
        # Eigenvalue computation failed
        return np.nan, np.nan, np.nan

    # =========================================================================
    # 3. Compute resolvent_norm
    # =========================================================================
    try:
        resolvent_norms = []
        I_mat = np.eye(m, dtype=A_mm.dtype)

        for lam in eig_tail:
            # Compute (lambda * I - A_mm)
            M_lam = lam * I_mat - A_mm

            # Compute smallest singular value
            svals = np.linalg.svdvals(M_lam)
            smin = float(svals[-1])

            # Resolvent norm is 1 / sigma_min
            # If sigma_min is very small, resolvent explodes -> report inf
            if smin < 1e-14:  # Numerical threshold for near-singularity
                resolvent_norms.append(np.inf)
            else:
                resolvent_norms.append(1.0 / smin)

        resolvent_norm = max(resolvent_norms) if resolvent_norms else np.nan

    except np.linalg.LinAlgError:
        # SVD computation failed
        resolvent_norm = np.nan

    return max_coupling_norm, distance_to_spectrum, resolvent_norm


def build_eigenvalue_result_dict(
    L: int,
    trial_id: int,
    setting: str,
    eigenvalue: complex,
    mode_index: int,
    mu_L: float,
    nu_L: float,
    svd_rank_used: int,
    tail_rank_used: int,
    system_params: dict[str, Any],
    signal_params: dict[str, Any],
    random_seed: int,
    max_coupling_norm: float = np.nan,
    distance_to_spectrum: float = np.nan,
    resolvent_norm: float = np.nan,
) -> dict[str, Any]:
    """
    Build a dictionary for a single eigenvalue result row.

    Args:
        L: Embedding length.
        trial_id: Trial iteration number.
        setting: Either "mixture_spurious" or "noise_only".
        eigenvalue: Complex eigenvalue.
        mode_index: Index of this eigenvalue.
        mu_L: Subspace boundary norm (first block).
        nu_L: Subspace boundary norm (last block).
        svd_rank_used: Rank used in SVD.
        tail_rank_used: Number of tail components.
        system_params: Dict with D, N_used, N_cols, m, M.
        signal_params: Dict with signal generation parameters.
        random_seed: Random seed used.
        max_coupling_norm: Maximum spectral norm of off-diagonal coupling blocks (default: NaN).
        distance_to_spectrum: Minimum distance between diagonal block spectra (default: NaN).
        resolvent_norm: Maximum resolvent norm over tail eigenvalues (default: NaN).

    Returns:
        Dictionary containing all columns for CSV export.
    """
    mode_type = "spurious" if setting == "mixture_spurious" else "noise"

    return {
        # Experiment parameters
        "L": L,
        "trial_id": trial_id,
        "mode_index": mode_index,
        "setting": setting,
        # Eigenvalue information
        "eigenvalue_magnitude": float(np.abs(eigenvalue)),
        "eigenvalue_real": float(np.real(eigenvalue)),
        "eigenvalue_imag": float(np.imag(eigenvalue)),
        "mode_type": mode_type,
        # Subspace metrics
        "mu_L": mu_L,
        "nu_L": nu_L,
        "svd_rank_used": svd_rank_used,
        "tail_rank_used": tail_rank_used,
        # Coupling metrics
        "max_coupling_norm": max_coupling_norm,
        "distance_to_spectrum": distance_to_spectrum,
        "resolvent_norm": resolvent_norm,
        # System parameters
        "spatial_dim": system_params["D"],
        "num_timesteps": system_params["N_used"],
        "N_cols": system_params["N_cols"],
        "num_modes": system_params["m"],
        "max_rank": system_params["M"],
        # Signal parameters
        "signal_eigenvalue_magnitude": signal_params["eigenvalue_magnitude"],
        "frequency_separation": signal_params["frequency_separation"],
        "snr_db": signal_params["snr_db"],
        "top_amplitude": signal_params["top_amplitude"],
        "noise_mode": signal_params["noise_mode"],
        # Random seed
        "random_seed": random_seed,
    }


# =============================================================================
# Main Experiment Class
# =============================================================================


class SpuriousEigenvalueExperiment:
    """
    Collects spurious eigenvalue magnitudes for different embedding lengths L.

    This class encapsulates the entire experiment workflow, from data generation
    to result saving, avoiding repetitive config extraction and deep parameter passing.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize experiment with configuration.

        Args:
            config: Configuration dictionary with keys: system, signal, experiment.
        """
        self.config = config

        # Extract config sections once (avoid repeated dict access)
        self.sys_cfg = config["system"]
        self.sig_cfg = config["signal"]
        self.exp_cfg = config["experiment"]

        # Extract commonly used values
        self.m = self.sys_cfg["num_modes"]  # true number of modes
        self.M = self.sys_cfg["max_rank"]  # DMD truncation rank
        self.D = self.sys_cfg["spatial_dim"]  # spatial dimension

        # Validate that M > m (need at least one spurious mode)
        if self.M <= self.m:
            raise ValueError(
                f"max_rank (M={self.M}) must be greater than num_modes (m={self.m}). "
                f"Need at least M = m+1 to have spurious modes."
            )

        self.L_values = self.exp_cfg["L_values"]  # embedding lengths to test
        self.n_mc = self.exp_cfg["n_mc_iterations"]  # MC iterations per L
        self.base_seed = self.exp_cfg.get("base_random_seed", 42)

        # Sample size control mode
        self.sample_size_mode = self.sys_cfg.get("sample_size_mode", "fixed_N")
        self.N_base = self.sys_cfg["num_timesteps"]  # Baseline N (for L_min)

        # Compute baseline values from lowest L
        self.L_min = min(self.L_values)
        self.N_cols_base = self.N_base - self.L_min  # Effective samples at L_min
        self.oversampling_base = self.N_cols_base / (self.D * self.L_min)

    def _create_kv_embedded_modes(
        self,
        true_modes: NDArray[np.complexfloating],
        true_eigenvalues: NDArray[np.complexfloating],
        L: int,
    ) -> NDArray[np.complexfloating]:
        """
        Create delay-embedded true modes with Kronecker-Vandermonde structure.

        Args:
            true_modes: True mode shapes, shape (D, m).
            true_eigenvalues: True eigenvalues, shape (m,).
            L: Embedding length.

        Returns:
            Delay-embedded modes with KV structure, shape (DL, m).
        """
        true_modes_embedded = np.zeros((L * self.D, self.m), dtype=complex)

        for mode_idx in range(self.m):
            true_mode_spatial = true_modes[:, mode_idx]
            true_eig = true_eigenvalues[mode_idx]

            # Create KV structure: [phi, lambda*phi, lambda^2*phi, ..., lambda^(L-1)*phi]
            for delay_idx in range(L):
                start_idx = delay_idx * self.D
                end_idx = start_idx + self.D
                true_modes_embedded[start_idx:end_idx, mode_idx] = (
                    true_eig**delay_idx
                ) * true_mode_spatial

        return true_modes_embedded

    def _compute_num_timesteps(self, L: int) -> int:
        """
        Compute number of timesteps N for given L based on sample size mode.

        Preserves the value at L_min for all other L values:
        - 'fixed_N': N stays constant (current behavior)
        - 'fixed_N_cols': N_cols = N - L stays constant
        - 'fixed_N_cols_over_DL': N_cols/(D·L) stays constant

        Args:
            L: Embedding length.

        Returns:
            Number of timesteps N to use for this L.
        """
        if self.sample_size_mode == "fixed_N":
            return self.N_base

        elif self.sample_size_mode == "fixed_N_cols":
            # N_cols = N - L constant → N = N_cols_base + L
            return self.N_cols_base + L

        elif self.sample_size_mode == "fixed_N_cols_over_DL":
            # N_cols/(D·L) constant → N_cols = oversampling_base * D * L
            # N = N_cols + L = oversampling_base * D * L + L
            return int(self.oversampling_base * self.D * L + L)

        else:
            raise ValueError(
                f"Unknown sample_size_mode: '{self.sample_size_mode}'. "
                f"Valid options: 'fixed_N', 'fixed_N_cols', 'fixed_N_cols_over_DL'"
            )

    def _generate_mixture_data(
        self,
        L: int,
        seed: int,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """
        Generate mixture data (signal + noise) for one trial.

        Args:
            L: Embedding length (number of delays).
            seed: Random seed for data generation.

        Returns:
            Tuple of:
            - X_noisy: Noisy snapshot matrix, shape (D, N).
            - X_clean: Clean signal snapshot matrix, shape (D, N).
            - true_eigenvalues: Ground truth eigenvalues, shape (m,).
            - true_modes: True mode shapes, shape (D, m).
            - true_modes_embedded: Delay-embedded true modes, shape (DL, m).
        """
        # Compute N based on sample size mode
        N = self._compute_num_timesteps(L)

        # Generate data
        generator = DMDDataGenerator(
            eigenvalue_magnitude=self.sig_cfg["eigenvalue_magnitude"],
            frequency_separation=self.sig_cfg["frequency_separation"],
            snr_db=self.sig_cfg["snr_db"],
            top_amplitude=self.sig_cfg.get("top_amplitude", 1.0),
            noise_mode=self.sig_cfg.get("noise_mode", "gaussian"),
            random_seed=seed,
            force_mode_orthogonality=self.sig_cfg["force_mode_orthogonality"],
        )

        X_noisy, X_clean, true_eigenvalues, true_modes, true_amplitudes = (
            generator.generate(
                n_spatial=self.D,
                n_timesteps=N,
                n_modes=self.m,
            )
        )

        # Embed true modes for SSL classification
        true_modes_embedded = self._create_kv_embedded_modes(
            true_modes, true_eigenvalues, L
        )

        return X_noisy, X_clean, true_eigenvalues, true_modes, true_modes_embedded

    def _run_dmd(
        self,
        X_noisy: NDArray,
        L: int,
        svd_rank: int,
    ) -> tuple[NDArray, NDArray]:
        """
        Run DMD on snapshot data.

        Args:
            X_noisy: Noisy snapshot matrix, shape (D, N).
            L: Embedding length (number of delays).
            svd_rank: SVD truncation rank.

        Returns:
            Tuple of:
            - recovered_eigenvalues: All recovered eigenvalues, shape (svd_rank,).
            - recovered_modes: All recovered DMD modes, shape (DL, svd_rank).
        """
        dmd = fit_dmd(
            X_noisy,
            svd_rank=svd_rank,
            mode="exact",
            num_delays=L,
        )

        return dmd.eigs, dmd.modes

    def _run_single_iteration(
        self,
        L: int,
        trial_id: int,
    ) -> list[dict[str, Any]]:
        """
        Run a single Monte Carlo iteration with both mixture and noise-only runs.

        Args:
            L: Embedding length.
            trial_id: Trial iteration number (formerly mc_iter).

        Returns:
            List of dictionaries containing eigenvalues and metrics from both
            mixture_spurious and noise_only settings.
        """
        # Deterministic random seed
        seed = self.base_seed + L * 1000 + trial_id

        # Compute N for this L
        N_used = self._compute_num_timesteps(L)
        N_cols = N_used - L

        # Common parameter dicts for result building
        system_params = {
            "D": self.D,
            "N_used": N_used,
            "N_cols": N_cols,
            "m": self.m,
            "M": self.M,
        }
        signal_params = {
            "eigenvalue_magnitude": self.sig_cfg["eigenvalue_magnitude"],
            "frequency_separation": self.sig_cfg["frequency_separation"],
            "snr_db": self.sig_cfg["snr_db"],
            "top_amplitude": self.sig_cfg.get("top_amplitude", 1.0),
            "noise_mode": self.sig_cfg.get("noise_mode", "gaussian"),
        }

        # =====================================================================
        # MIXTURE RUN (signal + noise)
        # =====================================================================

        # Generate mixture data
        (
            X_noisy,
            X_clean,
            true_eigenvalues,
            true_modes,
            true_modes_embedded,
        ) = self._generate_mixture_data(L, seed)

        # Run DMD on the mixture data
        recovered_eigenvalues, recovered_modes = self._run_dmd(
            X_noisy, L, svd_rank=self.M
        )

        # Compute U_M from the SAME X_noisy data
        # Keep full SVD to compute reduced propagator A_M
        embedding = DelayEmbedding(L)
        X_emb = embedding.transform(X_noisy)
        X_emb_0 = X_emb[:, :-1]
        X_emb_1 = X_emb[:, 1:]

        U, S, Vh = np.linalg.svd(X_emb_0, full_matrices=False)
        U_M = U[:, : self.M]
        S_M = S[: self.M]
        V_M = Vh.conj().T[:, : self.M]  # Shape: (ncols, M)

        # Select U_tail using pivoted QR
        I_m, I_tail, U_tail = select_subspaces_via_pivoted_qr(
            U_M, true_modes_embedded, self.m
        )

        # High-SNR validation diagnostic
        if self.sig_cfg["snr_db"] >= 50:
            overlap_count = len(set(I_m) & set(range(self.m)))
            if overlap_count < self.m - 1:
                print(
                    f"WARNING: High SNR ({self.sig_cfg['snr_db']} dB) but I_m overlap is only {overlap_count}/{self.m}"
                )
                print(f"  I_m = {I_m}, expected first m indices")

        # Compute mu_L and nu_L from U_tail
        mu_L_mixture, nu_L_mixture = compute_subspace_boundary_norms(U_tail, self.D, L)

        # Compute reduced propagator A_M
        A_M = _compute_reduced_propagator(U_M, S_M, V_M, X_emb_1)

        # Compute coupling metrics
        max_coupling_norm, distance_to_spectrum, resolvent_norm = (
            _compute_coupling_metrics(A_M, I_m, I_tail)
        )

        # If eigenvalue computation failed (returned NaN), skip this trial
        if np.isnan(distance_to_spectrum) and not (len(I_m) == 0 or len(I_tail) == 0):
            # Skip trial only if NaN is due to eigenvalue failure, not empty blocks
            return []

        # Classify modes and extract spurious eigenvalues
        mode_labels = classify_modes_by_ssr(
            recovered_modes, true_modes_embedded, self.m
        )
        spurious_mask = mode_labels == "spurious"
        spurious_eigenvalues = recovered_eigenvalues[spurious_mask]
        spurious_indices = np.where(spurious_mask)[0]

        # Build results for mixture_spurious setting
        results = []
        for idx, eig in zip(spurious_indices, spurious_eigenvalues):
            result = build_eigenvalue_result_dict(
                L=L,
                trial_id=trial_id,
                setting="mixture_spurious",
                eigenvalue=eig,
                mode_index=int(idx),
                mu_L=mu_L_mixture,
                nu_L=nu_L_mixture,
                svd_rank_used=self.M,
                tail_rank_used=self.M - self.m,
                system_params=system_params,
                signal_params=signal_params,
                random_seed=seed,
                max_coupling_norm=max_coupling_norm,
                distance_to_spectrum=distance_to_spectrum,
                resolvent_norm=resolvent_norm,
            )
            results.append(result)

        # =====================================================================
        # NOISE-ONLY RUN (using paired residual)
        # =====================================================================

        # Use paired noise: X_noise = X_noisy - X_clean
        X_noise = X_noisy - X_clean

        # Delay embed and compute U_noise
        X_noise_emb = embedding.transform(X_noise)
        X_noise_emb_0 = X_noise_emb[:, :-1]
        U_noise_full, _, _ = np.linalg.svd(X_noise_emb_0, full_matrices=False)
        rank_noise = self.M - self.m
        U_noise = U_noise_full[:, :rank_noise]

        # Compute mu_L and nu_L from U_noise
        mu_L_noise, nu_L_noise = compute_subspace_boundary_norms(U_noise, self.D, L)

        # Run DMD on noise-only data to get eigenvalues
        noise_eigenvalues, _ = self._run_dmd(X_noise, L, svd_rank=rank_noise)

        # Sanity check: expect exactly rank_noise eigenvalues
        if len(noise_eigenvalues) != rank_noise:
            raise RuntimeError(
                f"Expected {rank_noise} noise eigenvalues but DMD returned {len(noise_eigenvalues)}. "
                f"This may indicate a change in fit_dmd behavior."
            )

        # Build results for noise_only setting
        for idx, eig in enumerate(noise_eigenvalues):
            result = build_eigenvalue_result_dict(
                L=L,
                trial_id=trial_id,
                setting="noise_only",
                eigenvalue=eig,
                mode_index=idx,
                mu_L=mu_L_noise,
                nu_L=nu_L_noise,
                svd_rank_used=rank_noise,
                tail_rank_used=rank_noise,
                system_params=system_params,
                signal_params=signal_params,
                random_seed=seed,  # Same base seed for pairing
            )
            results.append(result)

        return results

    def run(self) -> pd.DataFrame:
        """
        Run the full experiment across all L values and MC iterations.

        Returns:
            DataFrame with all results, containing spurious eigenvalue magnitudes
            and all experiment parameters.
        """
        self._print_header()

        all_results = []

        print("\nStarting simulation loops...")
        for L in self.L_values:
            # Progress bar for each L value
            for trial_id in tqdm(
                range(self.n_mc), desc=f"Simulating L={L}", leave=True
            ):
                results = self._run_single_iteration(L, trial_id)
                all_results.extend(results)

        df = pd.DataFrame(all_results)

        self._print_footer(df)

        return df

    def _print_header(self):
        """Print experiment header information."""
        print("=" * 70)
        print("SPURIOUS EIGENVALUE MAGNITUDE vs L EXPERIMENT")
        print("=" * 70)
        print(f"Sample size mode: {self.sample_size_mode}")
        print(
            f"Baseline: N={self.N_base} at L_min={self.L_min} (N_cols={self.N_cols_base})"
        )
        print(f"L values: {self.L_values}")
        print(f"MC iterations per L: {self.n_mc}")
        print(f"Base random seed: {self.base_seed}")
        print(f"System: D={self.D}, m={self.m}, M={self.M}")
        print(
            f"Signal: eig_mag={self.sig_cfg['eigenvalue_magnitude']}, "
            f"freq_sep={self.sig_cfg['frequency_separation']}, "
            f"SNR={self.sig_cfg['snr_db']} dB"
        )
        print("=" * 70)
        print()

    def _print_footer(self, df: pd.DataFrame):
        """Print experiment completion information."""
        print()
        print("=" * 70)
        print("EXPERIMENT COMPLETED")
        print("=" * 70)
        print(f"Total eigenvalues collected: {len(df)}")
        # Expected: (M-m) spurious + (M-m) noise per trial
        expected_per_trial = 2 * (self.M - self.m)
        expected = expected_per_trial * self.n_mc * len(self.L_values)
        print(f"Expected: {expected} ({expected_per_trial} per trial)")
        print("=" * 70)
        print()

    def save(self, df: pd.DataFrame, output_path: str):
        """
        Save results to CSV.

        Args:
            df: Results DataFrame.
            output_path: Path to save CSV file.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print()


# =============================================================================
# CLI
# =============================================================================


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration YAML file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main(
    config: str = "analysis/spurious_eigenvalues_and_L/spurious_eigs_L_config.yaml",
    output: str = "results/spurious_eigs_L_results.csv",
):
    """
    Run spurious eigenvalue magnitude vs embedding length L experiment.

    Args:
        config: Path to config YAML file
        output: Output CSV file path
    """
    # Load config and create experiment
    config_dict = load_config(config)
    experiment = SpuriousEigenvalueExperiment(config_dict)

    # Run experiment
    df = experiment.run()

    # Save results
    experiment.save(df, output)

    print("Done!")


if __name__ == "__main__":
    fire.Fire(main)
