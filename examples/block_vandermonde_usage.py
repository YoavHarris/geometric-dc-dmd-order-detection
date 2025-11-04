"""
Example: Using Block-Vandermonde Fit Algorithms

This example demonstrates the refactored BV-fit classes.
"""

import numpy as np
from algorithms.block_vandermonde_fit_vec import NestedDMD, FixedEigenvalueBVFit
from algorithms.estimated_subspace_leakage import EstimatedSubspaceLeakage
from algorithms.clustering import ModeClustering


def example_bv_fit():
    """Example showing both BV-fit strategies."""
    
    # Simulate some DMD results
    num_delays = 10
    spatial_dim = 5
    num_modes = 20
    
    # Random modes and eigenvalues (replace with real DMD output)
    rng = np.random.default_rng(42)
    modes = rng.standard_normal((spatial_dim * num_delays, num_modes)) + \
            1j * rng.standard_normal((spatial_dim * num_delays, num_modes))
    eigenvalues = rng.uniform(0.8, 1.0, num_modes) * \
                  np.exp(1j * rng.uniform(0, 2*np.pi, num_modes))
    
    print("=" * 60)
    print("Block-Vandermonde Fit Example")
    print("=" * 60)
    print(f"Number of modes: {num_modes}")
    print(f"Spatial dimension: {spatial_dim}")
    print(f"Number of delays: {num_delays}")
    print()
    
    # Strategy 1: Nested DMD (more accurate, slower)
    print("Strategy 1: Nested DMD")
    print("-" * 60)
    nested = NestedDMD(
        num_delays=num_delays,
        spatial_dim=spatial_dim,
        epsilon=1e-12
    )
    
    nested_features = nested.compute_features(
        modes=modes,
        eigenvalues=eigenvalues,
        plot=False
    )
    
    print(f"Features computed: {list(nested_features.keys())}")
    print(f"Reconstruction scores (first 5): {nested_features['Reconstruction'][:5]}")
    print(f"Eigenvalue consistency (first 5): {nested_features['Eigenvalue-Consistency'][:5]}")
    print()
    
    # Strategy 2: Fixed-Eigenvalue BV Fit (faster, closed-form)
    print("Strategy 2: Fixed-Eigenvalue BV Fit (FEBVF)")
    print("-" * 60)
    febvf = FixedEigenvalueBVFit(
        num_delays=num_delays,
        spatial_dim=spatial_dim,
        epsilon=1e-12
    )
    
    febvf_features = febvf.compute_features(
        modes=modes,
        eigenvalues=eigenvalues,
        plot=False
    )
    
    print(f"Features computed: {list(febvf_features.keys())}")
    print(f"BV-Fit scores (first 5): {febvf_features['BV-Fit'][:5]}")
    print()
    
    # Combine with ESL for better separation
    print("Combined Approach: ESL + Nested DMD")
    print("-" * 60)
    
    esl = EstimatedSubspaceLeakage(epsilon=1e-12)
    esl_features = esl.compute_features(
        exact_modes=modes,  # In practice, use exact modes
        eigenvalues=eigenvalues,
        plot=False
    )
    
    # Combine features
    combined_features = {
        **nested_features,
        **esl_features
    }
    
    # Use only score features (not _raw)
    score_features = {
        k: v for k, v in combined_features.items()
        if not k.endswith("_raw")
    }
    
    print(f"Combined features: {list(score_features.keys())}")
    
    # Cluster modes
    clusterer = ModeClustering(
        normalization="min_max",
        strategy="vote",  # Use majority vote
        algorithm="kmeans",
        random_state=42
    )
    
    labels = clusterer.fit(score_features).labels_
    num_true = labels.sum()
    
    print(f"\nClustering results:")
    print(f"  Estimated true modes: {num_true} / {num_modes}")
    print(f"  Mode labels (first 10): {labels[:10]}")
    print()
    
    return {
        "nested": nested_features,
        "febvf": febvf_features,
        "esl": esl_features,
        "labels": labels,
    }


def compare_strategies():
    """Compare computational cost of both strategies."""
    import time
    
    print("=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    
    # Test different sizes
    sizes = [
        (5, 10, 50),   # (spatial_dim, num_delays, num_modes)
        (10, 20, 100),
        (20, 30, 200),
    ]
    
    for spatial_dim, num_delays, num_modes in sizes:
        print(f"\nSize: D={spatial_dim}, L={num_delays}, M={num_modes}")
        print("-" * 60)
        
        # Generate data
        rng = np.random.default_rng(42)
        modes = rng.standard_normal((spatial_dim * num_delays, num_modes)) + \
                1j * rng.standard_normal((spatial_dim * num_delays, num_modes))
        eigenvalues = rng.uniform(0.8, 1.0, num_modes) * \
                      np.exp(1j * rng.uniform(0, 2*np.pi, num_modes))
        
        # Time Nested DMD
        nested = NestedDMD(num_delays, spatial_dim)
        start = time.time()
        nested_features = nested.compute_features(modes, eigenvalues)
        nested_time = time.time() - start
        
        # Time FEBVF
        febvf = FixedEigenvalueBVFit(num_delays, spatial_dim)
        start = time.time()
        febvf_features = febvf.compute_features(modes, eigenvalues)
        febvf_time = time.time() - start
        
        print(f"  Nested DMD:  {nested_time*1000:.2f} ms")
        print(f"  FEBVF:       {febvf_time*1000:.2f} ms")
        print(f"  Speedup:     {nested_time/febvf_time:.1f}x")


if __name__ == "__main__":
    # Run examples
    results = example_bv_fit()
    print("\n")
    compare_strategies()
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)

