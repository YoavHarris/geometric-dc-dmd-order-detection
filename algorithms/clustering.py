"""
clustering.py
=============

Two-cluster assignment for DMD mode scores.

Supports K-means and Gaussian Mixture Models (GMM) with multiple
strategies for deciding which cluster represents "true" modes.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def normalize_features(
    features: dict[str, np.ndarray],
    mode: str | None,
) -> dict[str, np.ndarray]:
    """
    Normalize features per-column.

    Parameters
    ----------
    features : dict of arrays
        Feature name -> values (M,)
    mode : {"min_max", "z_score", None}
        Normalization mode.

    Returns
    -------
    Normalized feature dict.
    """
    if mode is None:
        return features

    normalized = {}
    for name, values in features.items():
        values = values.astype(float)

        if mode == "min_max":
            min_val, max_val = np.nanmin(values), np.nanmax(values)
            normalized[name] = (values - min_val) / (max_val - min_val + 1e-12)

        elif mode == "z_score":
            mean_val, std_val = np.nanmean(values), np.nanstd(values)
            normalized[name] = (values - mean_val) / (std_val + 1e-12)

        else:
            raise ValueError(f"Unknown normalization mode: {mode}")

    return normalized


def choose_true_cluster(
    features: dict[str, np.ndarray],
    feature_matrix: np.ndarray,
    base_labels: np.ndarray,
    strategy: str,
    pilot_feature: str,
    weights: dict[str, float],
) -> np.ndarray:
    """
    Decide which cluster (0 or 1) represents "true" modes.

    Strategies:
    - "single": Use one pilot feature's mean
    - "vote": Majority vote across all features
    - "mean": Cluster with larger weighted mean
    - "distance": Cluster farther from origin (in weighted norm)

    Returns
    -------
    labels : array of 0/1 where 1 = true cluster
    """
    cluster_0 = base_labels == 0
    cluster_1 = base_labels == 1
    feature_names = list(features.keys())

    if strategy == "single":
        values = features[pilot_feature]
        true_is_cluster_0 = values[cluster_0].mean() > values[cluster_1].mean()

    elif strategy == "vote":
        votes_for_0 = sum(
            features[name][cluster_0].mean() > features[name][cluster_1].mean()
            for name in feature_names
        )
        true_is_cluster_0 = votes_for_0 >= len(feature_names) / 2

    elif strategy == "mean":
        weight_array = np.array([weights.get(k, 1.0) for k in feature_names])
        mean_0 = (feature_matrix[cluster_0] * weight_array).mean()
        mean_1 = (feature_matrix[cluster_1] * weight_array).mean()
        true_is_cluster_0 = mean_0 > mean_1

    elif strategy == "distance":
        weight_array = np.array([weights.get(k, 1.0) for k in feature_names])
        center_0 = feature_matrix[cluster_0].mean(axis=0)
        center_1 = feature_matrix[cluster_1].mean(axis=0)
        distance_0 = np.sqrt((weight_array * center_0**2).sum())
        distance_1 = np.sqrt((weight_array * center_1**2).sum())
        true_is_cluster_0 = distance_0 > distance_1

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Return labels where 1 = true cluster
    return 1 - base_labels if true_is_cluster_0 else base_labels


class ModeClustering:
    """
    Two-cluster labeling for DMD modes.

    Takes feature dict (feature_name -> scores), clusters into 2 groups,
    and identifies which cluster represents "true" modes.

    Parameters
    ----------
    normalization : {"min_max", "z_score", None}, default "min_max"
        How to normalize features before clustering.
    strategy : {"single", "vote", "mean", "distance"}, default "vote"
        Strategy for choosing the "true" cluster.
    pilot_feature : str
        Required if strategy="single". Which feature to use.
    weights : dict, optional
        Per-feature weights for "mean" and "distance" strategies.
    algorithm : {"kmeans", "gmm"}, default "kmeans"
        Clustering algorithm.
    gmm_covariance : str, default "full"
        Covariance type for GMM (passed to sklearn).
    random_state : int or None
        Random seed for reproducibility.

    Attributes
    ----------
    labels_ : array of 0/1 (set after fit)
        Binary labels where 1 = predicted true mode.
    model_ : sklearn estimator (set after fit)
        Fitted KMeans or GaussianMixture object.
    feature_names_ : list of str (set after fit)
        Order of features used in clustering.
    """

    def __init__(
        self,
        normalization: str | None = "min_max",
        strategy: str = "vote",
        pilot_feature: str = "",
        weights: dict[str, float] | None = None,
        algorithm: str = "kmeans",
        gmm_covariance: str = "full",
        random_state: int | None = None,
    ):
        self.normalization = normalization
        self.strategy = strategy
        self.pilot_feature = pilot_feature
        self.weights = weights or {}
        self.algorithm = algorithm
        self.gmm_covariance = gmm_covariance
        self.random_state = random_state

        # Set after fit
        self.labels_ = None
        self.model_ = None
        self.feature_names_ = None

    def fit(self, features: dict[str, np.ndarray]) -> "ModeClustering":
        """
        Fit clustering and assign labels.

        Parameters
        ----------
        features : dict
            feature_name -> scores (M,)

        Returns
        -------
        self
        """
        # Validate
        if self.strategy == "single" and not self.pilot_feature:
            raise ValueError("pilot_feature required when strategy='single'")

        # Normalize
        normalized_features = normalize_features(features, self.normalization)
        self.feature_names_ = list(normalized_features.keys())

        # Stack into matrix (M, num_features)
        feature_matrix = np.stack(
            [normalized_features[name] for name in self.feature_names_], axis=1
        )

        # Cluster
        if self.algorithm == "kmeans":
            self.model_ = KMeans(
                n_clusters=2, n_init="auto", random_state=self.random_state
            ).fit(feature_matrix)
            base_labels = self.model_.labels_

        elif self.algorithm == "gmm":
            self.model_ = GaussianMixture(
                n_components=2,
                covariance_type=self.gmm_covariance,
                n_init=5,
                random_state=self.random_state,
            ).fit(feature_matrix)
            base_labels = self.model_.predict(feature_matrix)

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Decide which cluster is "true"
        self.labels_ = choose_true_cluster(
            normalized_features,
            feature_matrix,
            base_labels,
            self.strategy,
            self.pilot_feature,
            self.weights,
        )

        return self

    def __call__(self, features: dict[str, np.ndarray]) -> np.ndarray:
        """Convenience: fit and return labels in one call."""
        return self.fit(features).labels_


if __name__ == "__main__":
    # Simple test
    rng = np.random.default_rng(42)
    num_modes = 100

    test_features = {
        "feature_1": np.concatenate(
            [
                rng.normal(2.0, 0.3, num_modes // 2),
                rng.normal(-1.0, 0.2, num_modes // 2),
            ]
        ),
        "feature_2": rng.normal(0, 1, num_modes),
    }

    clusterer = ModeClustering(algorithm="kmeans", strategy="vote", random_state=42)
    labels = clusterer.fit(test_features).labels_

    print(f"Predicted {labels.sum()} true modes out of {num_modes}")
    print(f"First 10 labels: {labels[:10]}")
