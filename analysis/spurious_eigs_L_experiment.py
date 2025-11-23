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
from tqdm import tqdm

from utils.data_generation import DMDDataGenerator
from utils.dmd_utils import fit_dmd


# =============================================================================
# Utility Functions (Pure, Reusable)
# =============================================================================

def compute_leakage_projector(
    basis: NDArray[np.complexfloating]
) -> NDArray[np.complexfloating]:
    """
    Compute the leakage projector: I - P where P projects onto span(basis).
    
    Args:
        basis: Basis matrix (columns are basis vectors), shape (DL, rank).
        
    Returns:
        Orthogonal complement projector I - basis @ basis^H, shape (DL, DL).
    """
    DL = basis.shape[0]
    projector = basis @ basis.conj().T
    leakage_projector = np.eye(DL, dtype=complex) - projector
    return leakage_projector


def classify_modes_by_ssl(
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
    
    # Compute signal subspace projector
    # CRITICAL: Signal basis must be orthonormal for P = Q @ Q^H
    signal_basis_orthonormal, _ = np.linalg.qr(signal_basis)
    
    # Compute leakage projector: I - P_signal (projects onto signal complement)
    signal_leakage_projector = compute_leakage_projector(signal_basis_orthonormal)
    
    # Compute SSL for each recovered mode: ||(I - P_S) @ mode||^2
    ssl_values = np.zeros(M)
    for i in range(M):
        mode = recovered_modes[:, i]
        leakage = signal_leakage_projector @ mode
        ssl_values[i] = np.sum(np.abs(leakage) ** 2)
    
    # Sort modes by SSL (ascending)
    sorted_indices = np.argsort(ssl_values)
    
    # First num_true_modes with lowest SSL are true, rest are spurious
    mode_labels = np.array(["spurious"] * M, dtype=object)
    mode_labels[sorted_indices[:num_true_modes]] = "true"
    
    return mode_labels


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
        self.sys_cfg = config['system']
        self.sig_cfg = config['signal']
        self.exp_cfg = config['experiment']
        
        # Extract commonly used values
        self.m = self.sys_cfg['num_modes']          # true number of modes
        self.M = self.sys_cfg['max_rank']           # DMD truncation rank
        self.D = self.sys_cfg['spatial_dim']        # spatial dimension
        self.N = self.sys_cfg['num_timesteps']      # number of timesteps
        
        self.L_values = self.exp_cfg['L_values']    # embedding lengths to test
        self.n_mc = self.exp_cfg['n_mc_iterations'] # MC iterations per L
        self.base_seed = self.exp_cfg.get('base_random_seed', 42)
    
    def _create_bv_embedded_modes(
        self, 
        true_modes: NDArray[np.complexfloating],
        true_eigenvalues: NDArray[np.complexfloating],
        L: int,
    ) -> NDArray[np.complexfloating]:
        """
        Create delay-embedded true modes with Block-Vandermonde structure.
        
        Args:
            true_modes: True mode shapes, shape (D, m).
            true_eigenvalues: True eigenvalues, shape (m,).
            L: Embedding length.
            
        Returns:
            Delay-embedded modes with BV structure, shape (DL, m).
        """
        true_modes_embedded = np.zeros((L * self.D, self.m), dtype=complex)
        
        for mode_idx in range(self.m):
            true_mode_spatial = true_modes[:, mode_idx]
            true_eig = true_eigenvalues[mode_idx]
            
            # Create BV structure: [phi, lambda*phi, lambda^2*phi, ..., lambda^(L-1)*phi]
            for delay_idx in range(L):
                start_idx = delay_idx * self.D
                end_idx = start_idx + self.D
                true_modes_embedded[start_idx:end_idx, mode_idx] = (
                    (true_eig ** delay_idx) * true_mode_spatial
                )
        
        return true_modes_embedded
    
    def _generate_data_and_run_dmd(
        self, 
        L: int, 
        rng: np.random.Generator,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Generate data and run DMD with delay embedding.
        
        Args:
            L: Embedding length (number of delays).
            rng: Random number generator.
            
        Returns:
            Tuple of:
            - recovered_eigenvalues: All recovered eigenvalues, shape (M,).
            - recovered_modes: All recovered DMD modes, shape (DL, M).
            - true_eigenvalues: Ground truth eigenvalues, shape (m,).
            - true_modes_embedded: Delay-embedded true modes, shape (DL, m).
        """
        # Generate data
        generator = DMDDataGenerator(
            eigenvalue_magnitude=self.sig_cfg['eigenvalue_magnitude'],
            frequency_separation=self.sig_cfg['frequency_separation'],
            snr_db=self.sig_cfg['snr_db'],
            top_amplitude=self.sig_cfg.get('top_amplitude', 1.0),
            noise_mode=self.sig_cfg.get('noise_mode', 'gaussian'),
            random_seed=None,  # Use external rng
        )
        generator.rng = rng  # Override internal RNG
        
        X_noisy, X_clean, true_eigenvalues, true_modes, true_amplitudes = generator.generate(
            n_spatial=self.D,
            n_timesteps=self.N,
            n_modes=self.m,
        )
        
        # Embed true modes for SSL classification
        true_modes_embedded = self._create_bv_embedded_modes(
            true_modes, true_eigenvalues, L
        )
        
        # Run DMD
        dmd = fit_dmd(
            X_noisy,
            svd_rank=self.M,
            mode="exact",
            num_delays=L,
        )
        
        return dmd.eigs, dmd.modes, true_eigenvalues, true_modes_embedded
    
    def _run_single_iteration(
        self, 
        L: int, 
        mc_iter: int,
    ) -> list[dict[str, Any]]:
        """
        Run a single Monte Carlo iteration.
        
        Args:
            L: Embedding length.
            mc_iter: Monte Carlo iteration number.
            
        Returns:
            List of dictionaries, one per spurious eigenvalue, containing all
            parameters and eigenvalue information.
        """
        # Deterministic random seed
        seed = self.base_seed + L * 1000 + mc_iter
        rng = np.random.default_rng(seed)
        
        # Generate data and run DMD
        recovered_eigenvalues, recovered_modes, true_eigenvalues, true_modes_embedded = \
            self._generate_data_and_run_dmd(L, rng)
        
        # Classify modes as true or spurious
        mode_labels = classify_modes_by_ssl(
            recovered_modes,
            true_modes_embedded,
            self.m,
        )
        
        # Extract spurious eigenvalues
        spurious_mask = mode_labels == "spurious"
        spurious_eigenvalues = recovered_eigenvalues[spurious_mask]
        spurious_indices = np.where(spurious_mask)[0]
        
        # Build results
        results = []
        for idx, eig in zip(spurious_indices, spurious_eigenvalues):
            result = {
                # Experiment parameters
                'L': L,
                'mc_iter': mc_iter,
                'mode_index': int(idx),
                
                # Eigenvalue information
                'eigenvalue_magnitude': float(np.abs(eig)),
                'eigenvalue_real': float(np.real(eig)),
                'eigenvalue_imag': float(np.imag(eig)),
                'mode_type': 'spurious',
                
                # System parameters (for identifiability)
                'spatial_dim': self.D,
                'num_timesteps': self.N,
                'num_modes': self.m,
                'max_rank': self.M,
                
                # Signal parameters
                'signal_eigenvalue_magnitude': self.sig_cfg['eigenvalue_magnitude'],
                'frequency_separation': self.sig_cfg['frequency_separation'],
                'snr_db': self.sig_cfg['snr_db'],
                'top_amplitude': self.sig_cfg.get('top_amplitude', 1.0),
                'noise_mode': self.sig_cfg.get('noise_mode', 'gaussian'),
                
                # Random seed (for reproducibility)
                'random_seed': seed,
            }
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
        total_runs = len(self.L_values) * self.n_mc
        
        with tqdm(total=total_runs, desc="Running experiment") as pbar:
            for L in self.L_values:
                for mc_iter in range(self.n_mc):
                    results = self._run_single_iteration(L, mc_iter)
                    all_results.extend(results)
                    pbar.update(1)
        
        df = pd.DataFrame(all_results)
        
        self._print_footer(df)
        
        return df
    
    def _print_header(self):
        """Print experiment header information."""
        print("=" * 70)
        print("SPURIOUS EIGENVALUE MAGNITUDE vs L EXPERIMENT")
        print("=" * 70)
        print(f"L values: {self.L_values}")
        print(f"MC iterations per L: {self.n_mc}")
        print(f"Base random seed: {self.base_seed}")
        print(f"System: D={self.D}, N={self.N}, m={self.m}, M={self.M}")
        print(f"Signal: eig_mag={self.sig_cfg['eigenvalue_magnitude']}, "
              f"freq_sep={self.sig_cfg['frequency_separation']}, "
              f"SNR={self.sig_cfg['snr_db']} dB")
        print("=" * 70)
        print()
    
    def _print_footer(self, df: pd.DataFrame):
        """Print experiment completion information."""
        print()
        print("=" * 70)
        print("EXPERIMENT COMPLETED")
        print("=" * 70)
        print(f"Total spurious eigenvalues collected: {len(df)}")
        expected = sum((self.M - self.m) * self.n_mc for _ in self.L_values)
        print(f"Expected: {expected}")
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
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point for CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Collect spurious eigenvalue magnitude data vs embedding length L"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='analysis/spurious_eigs_L_data_config.yaml',
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='analysis/spurious_eigs_L_results.csv',
        help='Output CSV file path'
    )
    
    args = parser.parse_args()
    
    # Load config and create experiment
    config = load_config(args.config)
    experiment = SpuriousEigenvalueExperiment(config)
    
    # Run experiment
    df = experiment.run()
    
    # Save results
    experiment.save(df, args.output)
    
    print("Done!")


if __name__ == "__main__":
    main()
