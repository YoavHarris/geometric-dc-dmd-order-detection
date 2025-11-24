"""
Leakage Separation Experiment: SELN vs RELN validation.

This script validates the theoretical separation of leakage norms between
true and spurious DMD components using:
- SELN (Signal-subspace Empirical Leakage Norm): oracle metric using true signal subspace
- RELN (Rank-estimated Empirical Leakage Norm): practical metric using estimated subspace

The experiment runs at a fixed working point (N=200, D=45, m=5, L=64, M=15, etc.)
and outputs a CSV with per-component leakage measurements.
"""

from __future__ import annotations

import fire
from pathlib import Path
from typing import Any
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import yaml
from tqdm import tqdm

from utils.data_generation import DMDDataGenerator
from utils.dmd_utils import fit_dmd
from analysis.leakage_separation_tools import (
    compute_ssl,
    compute_esl,
    compute_exact_mode_norm,
)
from analysis.subspace_analysis import (
    build_block_vandermonde_modes,
    compute_estimated_basis,
)


# =============================================================================
# Main Experiment Class
# =============================================================================


class LeakageSeparationExperiment:
    """
    Validates leakage norm separation between true and spurious DMD components.

    This class encapsulates the entire experiment workflow at a fixed working point,
    measuring both SELN and RELN for each recovered component.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize experiment with configuration.

        Args:
            config: Configuration dictionary with keys: system, signal, experiment, output.
        """
        self.config = config

        # Extract config sections
        self.sys_cfg = config["system"]
        self.sig_cfg = config["signal"]
        self.exp_cfg = config["experiment"]
        self.out_cfg = config["output"]

        # Extract commonly used values
        self.D = self.sys_cfg["spatial_dim"]
        self.N = self.sys_cfg["num_timesteps"]
        self.m = self.sys_cfg["num_modes"]
        self.M = self.sys_cfg["max_rank"]
        self.L = self.sys_cfg["num_delays"]

        self.num_trials = self.exp_cfg["num_trials"]
        self.base_seed = self.exp_cfg.get("base_random_seed", 42)

    def _create_bv_embedded_modes(
        self,
        true_modes: NDArray[np.complexfloating],
        true_eigenvalues: NDArray[np.complexfloating],
    ) -> NDArray[np.complexfloating]:
        """
        Create delay-embedded true modes with Block-Vandermonde structure.

        Args:
            true_modes: True mode shapes, shape (D, m).
            true_eigenvalues: True eigenvalues, shape (m,).

        Returns:
            Delay-embedded modes with BV structure, shape (DL, m).
        """
        return build_block_vandermonde_modes(true_modes, true_eigenvalues, self.L)

    def _classify_modes_by_ssl(
        self,
        recovered_modes: NDArray[np.complexfloating],
        signal_basis: NDArray[np.complexfloating],
    ) -> NDArray[np.bool_]:
        """
        Classify modes as true or spurious using Signal Subspace Leakage (SSL).

        Modes with lowest SSL (most aligned with signal subspace) are classified as true.

        Args:
            recovered_modes: All recovered DMD modes, shape (DL, M).
            signal_basis: Basis for true signal subspace, shape (DL, m).

        Returns:
            Boolean array of shape (M,), True for true modes, False for spurious.
        """
        M = recovered_modes.shape[1]

        # Compute SSL for each mode
        ssl_values = np.zeros(M)
        for i in range(M):
            mode = recovered_modes[:, i]
            ssl_values[i] = compute_ssl(mode, signal_basis)

        # Sort modes by SSL (ascending)
        sorted_indices = np.argsort(ssl_values)

        # First m modes with lowest SSL are true, rest are spurious
        is_true = np.zeros(M, dtype=bool)
        is_true[sorted_indices[: self.m]] = True

        return is_true

    def _generate_data_and_run_dmd(
        self, seed: int
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
        """
        Generate data and run DMD with delay embedding.

        Args:
            seed: Random seed for data generation.

        Returns:
            Tuple of:
            - recovered_eigenvalues: All recovered eigenvalues, shape (M,).
            - recovered_modes: All recovered DMD modes, shape (DL, M).
            - true_eigenvalues: Ground truth eigenvalues, shape (m,).
            - true_modes: Ground truth spatial modes, shape (D, m).
            - X_clean: Clean trajectory, shape (D, N).
            - X_noisy: Noisy trajectory, shape (D, N).
        """
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
                n_timesteps=self.N,
                n_modes=self.m,
            )
        )

        # Run DMD with delay embedding
        dmd = fit_dmd(
            X_noisy,
            svd_rank=self.M,
            mode="exact",
            num_delays=self.L,
        )

        return (
            dmd.eigs,
            dmd.modes,
            true_eigenvalues,
            true_modes,
            X_clean,
            X_noisy,
        )

    def _run_single_trial(self, trial_id: int) -> list[dict[str, Any]]:
        """
        Run a single trial and compute leakage metrics for all components.

        Args:
            trial_id: Trial identifier.

        Returns:
            List of dictionaries, one per component, with all leakage measurements.
        """
        # Deterministic random seed
        seed = self.base_seed + trial_id

        # Generate data and run DMD
        (
            recovered_eigenvalues,
            recovered_modes,
            true_eigenvalues,
            true_modes,
            X_clean,
            X_noisy,
        ) = self._generate_data_and_run_dmd(seed)

        # Create true signal subspace (for SSL computation)
        # This is the delay-embedded true modes with Block-Vandermonde structure
        true_bv_modes = self._create_bv_embedded_modes(true_modes, true_eigenvalues)

        # Create estimated subspace (for ESL computation)
        # This comes from SVD of the noisy delay-embedded data
        from dmd.dmd_tools import DelayEmbedding

        embedder = DelayEmbedding(self.L)
        X_noisy_embedded = embedder.transform(X_noisy)
        estimated_basis = compute_estimated_basis(X_noisy_embedded, self.M)

        # Classify modes as true or spurious
        is_true = self._classify_modes_by_ssl(recovered_modes, true_bv_modes)

        # Compute leakage metrics for each component
        results = []
        for component_id in range(self.M):
            mode = recovered_modes[:, component_id]

            # Compute exact mode norm
            exact_mode_norm = compute_exact_mode_norm(mode)

            # Compute SSL and ESL
            ssl = compute_ssl(mode, true_bv_modes)
            esl = compute_esl(mode, estimated_basis)

            # Compute relative norms
            seln = ssl / exact_mode_norm if exact_mode_norm > 0 else 0.0
            reln = esl / exact_mode_norm if exact_mode_norm > 0 else 0.0

            result = {
                "trial_id": trial_id,
                "component_id": component_id,
                "is_true": int(is_true[component_id]),
                "exact_mode_norm": float(exact_mode_norm),
                "ssl": float(ssl),
                "esl": float(esl),
                "seln": float(seln),
                "reln": float(reln),
                "snr_db": self.sig_cfg["snr_db"],
                "noise_model": self.sig_cfg.get("noise_mode", "gaussian"),
            }
            results.append(result)

        return results

    def run(self) -> pd.DataFrame:
        """
        Run the full experiment across all trials.

        Returns:
            DataFrame with all results, one row per component.
        """
        self._print_header()

        all_results = []

        with tqdm(total=self.num_trials, desc="Running trials") as pbar:
            for trial_id in range(self.num_trials):
                results = self._run_single_trial(trial_id)
                all_results.extend(results)
                pbar.update(1)

        df = pd.DataFrame(all_results)

        self._print_footer(df)

        return df

    def _print_header(self):
        """Print experiment header information."""
        print("=" * 70)
        print("LEAKAGE SEPARATION EXPERIMENT: SELN vs RELN")
        print("=" * 70)
        print(f"Working point:")
        print(f"  N={self.N}, D={self.D}, m={self.m}, M={self.M}, L={self.L}")
        print(f"  r={self.sig_cfg['eigenvalue_magnitude']}")
        print(f"  Δθ={self.sig_cfg['frequency_separation']}")
        print(f"  SNR={self.sig_cfg['snr_db']} dB")
        print(f"  Noise: {self.sig_cfg.get('noise_mode', 'gaussian')}")
        print(f"Trials: {self.num_trials}")
        print(f"Base random seed: {self.base_seed}")
        print("=" * 70)
        print()

    def _print_footer(self, df: pd.DataFrame):
        """Print experiment completion information."""
        print()
        print("=" * 70)
        print("EXPERIMENT COMPLETED")
        print("=" * 70)
        print(f"Total components measured: {len(df)}")
        print(f"Expected: {self.num_trials * self.M}")
        true_count = df["is_true"].sum()
        spurious_count = len(df) - true_count
        print(f"True components: {true_count} (~{self.m} per trial)")
        print(f"Spurious components: {spurious_count} (~{self.M - self.m} per trial)")
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


def run_experiment(
    config: str = "analysis/leakage_separation_config.yaml",
    output: str | None = None,
):
    """
    Run the leakage separation experiment via Python Fire.

    Args:
        config: Path to configuration YAML file.
        output: Optional output CSV file path. If provided, overrides config["output"]["csv_path"].
    """
    # Load config
    cfg = load_config(config)

    # Override output path if provided
    if output is not None:
        cfg["output"]["csv_path"] = output

    # Create and run experiment
    experiment = LeakageSeparationExperiment(cfg)
    df = experiment.run()

    # Save results
    experiment.save(df, cfg["output"]["csv_path"])

    print("Done!")


if __name__ == "__main__":
    fire.Fire(run_experiment)
