"""
mode_clustering.py
==================

Two-cluster assignment for DMD-mode (or other low-dim) scores
with a choice of back-ends:

    • K-means (equal-variance assumption, fastest)
    • two-component Gaussian Mixture (heteroscedastic, soft EM)

The `ModeClustering` class mirrors scikit-learn: call
    clf = ModeClustering(...).fit(score_dict)
and inspect

    clf.labels_        -> 1 = “true” cluster picked by your decider
    clf.model_         -> fitted KMeans or GaussianMixture estimator
    clf.feat_names_    -> order of columns used in X

You can also use the instance as a one-shot callable:
    labels = ModeClustering(...)(score_dict)
"""

from __future__ import annotations
from typing import Dict, Literal, Union

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# ----------------------------------------------------------------------
# type aliases
# ----------------------------------------------------------------------
Decider = Literal["single", "vote", "mean", "distance"]
Algorithm = Literal["kmeans", "gmm"]


# ----------------------------------------------------------------------
# helpers (module-private)
# ----------------------------------------------------------------------
def _normalize(
    score_dict: Dict[str, np.ndarray],
    mode: str | None,
) -> Dict[str, np.ndarray]:
    """Return a new dict with per-feature scaling."""
    if mode is None:
        return score_dict

    out: Dict[str, np.ndarray] = {}
    for k, v in score_dict.items():
        x = v.astype(float)
        if mode == "min_max":
            lo, hi = np.nanmin(x), np.nanmax(x)
            out[k] = (x - lo) / (hi - lo + 1e-12)
        elif mode == "z_score":
            mu, sd = np.nanmean(x), np.nanstd(x) + 1e-12
            out[k] = (x - mu) / sd
        else:
            raise ValueError(f"Unknown normalization mode '{mode}'")
    return out


def _choose_true_cluster(
    Xdict: Dict[str, np.ndarray],
    X: np.ndarray,
    base_labels: np.ndarray,
    decider: Decider,
    pilot_feature: str,
    weights: Dict[str, float],
) -> np.ndarray:
    """Map base_labels (0/1) → final labels where 1 == 'true'."""
    c0, c1 = base_labels == 0, base_labels == 1
    keys = list(Xdict)

    if decider == "single":
        vals = Xdict[pilot_feature]
        true_is_0 = vals[c0].mean() > vals[c1].mean()

    elif decider == "vote":
        votes0 = sum((Xdict[k][c0].mean() > Xdict[k][c1].mean()) for k in keys)
        true_is_0 = votes0 >= len(keys) / 2

    else:  # mean or distance
        w = np.array([weights.get(k, 1.0) for k in keys]) if weights else 1.0

        if decider == "mean":
            true_is_0 = (X[c0] * w).mean() > (X[c1] * w).mean()

        elif decider == "distance":
            centres = np.vstack([X[c0].mean(axis=0), X[c1].mean(axis=0)])
            d0 = np.sqrt((w * centres[0] ** 2).sum())
            d1 = np.sqrt((w * centres[1] ** 2).sum())
            true_is_0 = d0 > d1

        else:
            raise ValueError(f"Unknown decider '{decider}'")

    return 1 - base_labels if true_is_0 else base_labels


# ----------------------------------------------------------------------
# main class
# ----------------------------------------------------------------------
class ModeClustering:
    """
    Two-cluster labeler.

    Parameters
    ----------
    normalization_mode : "min_max", "z_score", or None
    decider            : "single" | "vote" | "mean" | "distance"
    pilot_feature      : required if decider == "single"
    weights            : per-feature weights for "mean"/"distance"
    algorithm          : "kmeans" | "gmm"
    gmm_covariance_type: passed to sklearn GaussianMixture
    random_state       : int or None

    After `.fit()` you have
        labels_   – np.ndarray of 0/1 (1 == true cluster)
        model_    – fitted estimator (KMeans or GaussianMixture)
    """

    def __init__(
        self,
        normalization_mode: str | None = "min_max",
        decider: Decider = "vote",
        pilot_feature: str = "",
        weights: Dict[str, float] | None = None,
        algorithm: Algorithm = "kmeans",
        gmm_covariance_type: str = "full",
        random_state: int | None = None,
    ):
        self.norm_mode = normalization_mode
        self.decider = decider
        self.pilot_feat = pilot_feature
        self.weights = weights or {}
        self.algorithm = algorithm
        self.cov_type = gmm_covariance_type
        self.random = random_state

    # --------------------------------------------------------------
    def fit(self, score_dict: Dict[str, np.ndarray]) -> "ModeClustering":
        if self.decider == "single" and not self.pilot_feat:
            raise ValueError("pilot_feature must be set when decider='single'")

        # 1) normalise
        Xdict = _normalize(score_dict, self.norm_mode)
        self.feat_names_ = list(Xdict)
        X = np.stack([Xdict[k] for k in self.feat_names_], axis=1)

        # 2) cluster
        if self.algorithm == "kmeans":
            self.model_ = KMeans(
                n_clusters=2, n_init="auto", random_state=self.random
            ).fit(X)
            base_labels = self.model_.labels_

        elif self.algorithm == "gmm":
            self.model_ = GaussianMixture(
                n_components=2,
                covariance_type=self.cov_type,
                n_init=5,
                random_state=self.random,
            ).fit(X)
            base_labels = self.model_.predict(X)

        else:
            raise ValueError(f"Unknown algorithm '{self.algorithm}'")

        # 3) decide which cluster is 'true'
        self.labels_ = _choose_true_cluster(
            Xdict, X, base_labels, self.decider, self.pilot_feat, self.weights
        )
        return self

    # --------------------------------------------------------------
    def __call__(self, score_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """One-liner: returns labels after fitting."""
        return self.fit(score_dict).labels_


# ----------------------------------------------------------------------
__all__ = ["ModeClustering"]


if __name__ == "__main__":
    # toy data
    rng = np.random.default_rng(0)
    m = 400
    score_dict = {
        "main": np.r_[rng.normal(1, 0.2, m // 2), rng.normal(-1, 0.1, m // 2)],
        "aux": rng.normal(0, 1, m),
    }

    clf = ModeClustering(algorithm="gmm", decider="distance", random_state=0).fit(
        score_dict
    )

    print(clf.labels_[:10])  # first ten binary labels
    print(clf.model_.means_)  # GMM means for inspection
