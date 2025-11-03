from typing import Dict, Union, Literal

import numpy as np
from numpy.linalg import svd, qr
from numpy.typing import NDArray
from sklearn.cluster import KMeans


def normalize_scores(
    score_dict: Dict[str, np.ndarray], mode: str = "min_max"
) -> Dict[str, np.ndarray]:
    assert mode in ["z_score", "min_max"]

    if mode == "z_score":
        normalized = {
            k: (
                (v - np.mean(v, keepdims=True)) / np.std(v, keepdims=True)
                if np.std(v) > 0
                else v
            )
            for k, v in score_dict.items()
        }
    elif mode == "min_max":
        normalized = {}
        for k, v in score_dict.items():
            if np.ptp(v) == 0:  # constant array
                normalized[k] = np.zeros_like(v)
            else:
                normalized[k] = (v - v.min()) / np.ptp(v)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")
    return normalized


Decider = Literal["single", "vote", "mean", "distance"]


def cluster_modes(
    score_dict: Dict[str, np.ndarray],
    normalization_mode: Union[str, None] = "min_max",
    decider: Decider = "vote",
    pilot_feature: str = "",
    weights: Union[Dict[str, float], None] = None,
) -> NDArray[np.floating]:
    """
    Cluster DMD modes (K=2) and decide which cluster is 'true'.

    Parameters
    ----------
    score_dict           : dict[str, np.ndarray]
        Raw or already‑scaled feature arrays, all length M.
    normalization_mode   : "min_max" | "z_score" | None
        Passed to `normalize_scores`.  Leave None if `score_dict`
        is pre‑scaled by `align_and_scale_scores`.
    decider              : "single" | "vote" | "mean" | "distance"
        * single    – legacy behaviour using one `pilot_feature`
        * vote      – majority vote over features
        * mean      – cluster with larger weighted mean score
        * distance  – cluster with larger weighted ℓ² distance from 0
    pilot_feature        : str
        Used only when `decider="single"`.
    weights              : dict[str, float]  (optional)
        Per‑feature weights for "mean" or "distance".  Defaults to
        uniform 1 for all keys in `score_dict`.

    Returns
    -------
    np.ndarray
        Binary labels (1 == predicted true mode).
    """
    if decider == "single":
        assert pilot_feature, "Must specify pilot_feature if using single decider"
    # 1) normalise
    Xdict = (
        normalize_scores(score_dict, mode=normalization_mode)
        if normalization_mode
        else score_dict
    )
    feat_names = list(Xdict.keys())
    X = np.stack([Xdict[k] for k in feat_names], axis=1)  # (M, p)

    # 2) K‑means
    km = KMeans(n_clusters=2, n_init="auto").fit(X)
    labels = km.labels_
    c0, c1 = (labels == 0), (labels == 1)

    # 3) choose cluster
    if decider == "single":
        values = Xdict[pilot_feature]
        m0, m1 = values[c0].mean(), values[c1].mean()
        true_is_0 = m0 > m1

    elif decider == "vote":
        votes0 = sum((Xdict[k][c0].mean() > Xdict[k][c1].mean()) for k in feat_names)
        true_is_0 = votes0 >= (len(feat_names) / 2)  # majority

    elif decider == "mean":
        w = np.array([weights.get(k, 1.0) for k in feat_names]) if weights else 1.0
        m0 = (X[c0] * w).mean()
        m1 = (X[c1] * w).mean()
        true_is_0 = m0 > m1

    elif decider == "distance":
        w = np.array([weights.get(k, 1.0) for k in feat_names]) if weights else 1.0
        d0 = np.sqrt((w * km.cluster_centers_[0] ** 2).sum())
        d1 = np.sqrt((w * km.cluster_centers_[1] ** 2).sum())
        true_is_0 = d0 > d1  # farther from origin → higher scores

    else:
        raise ValueError(f"Unknown decider '{decider}'")

    return 1 - labels if true_is_0 else labels


# ────────── Sub-space similarity helper ──────────
def subspace_stats(Phi, U, k, *, eps=1e-12):
    """
    Phi : (n × k') basis of the 'true-signal' subspace      (not necessarily orthonormal)
    U   : (n × r ) orthonormal columns from the SVD (use first k!)
    k   : dimension of the true subspace (= num_modes)

    Returns
    -------
    dict with:
        theta         – length-k array of principal angles  (rad)
        overlap_rho   – (1/k) Σ cos² θ_i          → 1 if identical
        chordal_dist  – || sin θ ||₂              → 0 if identical
        sin_theta_max – max_i sin θ_i             → worst-case mis-alignment
    """
    # 1) take only the k columns that span the subspace we care about
    Phi_k = Phi[:, :k]  # Φ  may have >k columns (spurious modes) – discard them
    U_k = U[:, :k]  # U  already orthonormal from SVD

    # 2) orthonormalise Φ once (economy QR)
    Q_phi, R = np.linalg.qr(Phi_k)  # n × k,  k × k

    # guard against numerical rank deficiency
    if np.abs(np.diag(R)).min() < eps:
        raise ValueError(
            "Φ is numerically rank-deficient – cannot define k-D subspace."
        )

    # 3) principal angles via k×k SVD
    _, svals, _ = np.linalg.svd(Q_phi.conj().T @ U_k)  # singular values = cos θ_i
    cos_theta = np.clip(svals, 0.0, 1.0)
    theta = np.arccos(cos_theta)
    sin_theta = np.sqrt(1.0 - cos_theta**2)

    stats = dict(
        theta=theta,  # full vector if you want it
        overlap_rho=np.mean(cos_theta**2),  # ρ  ∈ [0,1]   higher = better
        chordal_dist=np.linalg.norm(sin_theta),  # d_ch ∈ [0,√k] lower = better
        sin_theta_max=sin_theta.max(),  # worst-case direction
    )
    return stats
