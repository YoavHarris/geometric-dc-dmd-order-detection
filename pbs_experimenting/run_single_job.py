#!/usr/bin/env python3
"""
Run a single DMD order-detection experiment job.
Reads configuration from a YAML file.
"""

import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
import yaml

import numpy as np
import pandas as pd
import scipy
from scipy import optimize
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
import fire

from algorithms.estimated_subspace_residual import EstimatedSubspaceResidual
from algorithms.kronecker_vandermonde_fit import FixedEigenvalueKVFit, NestedDMD
from algorithms.res_dmd import ResDMDResidual
from algorithms.clustering import ModeClustering
from algorithms.stc import STC
from algorithms.bic import (
    compute_bic_rank,
    gap_ranks,
)

from utils.delay_embedding import DelayEmbedding
from utils.data_generation import DMDDataGenerator
from dmd.dmd_utils import fit_dmd, align_modes_and_amplitudes_phases
from utils.visualizations import (
    imshow_complex,
    plot_mode_table,
    scatter_scores_1d,
)
from analysis.subspace_analysis import compute_subspace_principal_angles


# Methods that require delay embedding (num_delays > 1)
DELAY_EMBEDDING_METHODS = {
    "STC",
    "NestedDMD",
    "FixedEigenvalueKVFit",
    "NestedDMD+ESR",
    "FixedEigenvalueKVFit+ESR",
}

# Information criteria methods
INFO_CRITERIA_METHODS = {
    "BIC",
}

# Methods whose per-mode scores are reported as ROC-AUC / PR-AUC.
# Excludes BIC and GAP (no per-mode scalar score) and the combination
# methods (no single scalar without an extra fusion hyperparameter).
# ROC/PR-AUC are rank-based, so the score's scale is irrelevant — only
# its orientation ("higher = more likely true mode").
ROC_AUC_METHODS = {
    "ESR-Energy",
    "NestedDMD",
    "FixedEigenvalueKVFit",
    "STC",
    "ResDMDResidual",
}


def _skip_run_with_empty_csv(
    output_path: str | Path,
    reason: str,
) -> None:
    """Abort the run and write a header-only CSV."""
    summary_cols = [
        # parameters
        "snr_db",
        "freq_sep",
        "eig_mag",
        "eig_mag_spread",
        "rho_mode",
        "delays_over_timesteps",
        "num_delays",
        "temporal_dim",
        "spatial_dim",
        "num_modes",
        "n_independent_modes",
        "top_amplitude",
        "max_rank",
        "artificial_damping",
        "n_iter",
        "dt",
        "noise_mode",
        "base_random_seed",
        "random_seed",
        "method",
        # metrics
        "abs_diff_mean",
        "abs_diff_std",
        "bias_frac_over",
        "bias_frac_under",
        "bias_skewness",
        "order_hit_prob",
        "mean_time_sec",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "tp",
        "fp",
        "fn",
        "tn",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "pr_auc_std",
        # subspace proximity
        "max_pa_clean_mean",
        "max_pa_clean_std",
        "sine_max_pa_clean_mean",
        "sine_max_pa_clean_std",
        "mean_pa_clean_mean",
        "mean_pa_clean_std",
        "max_pa_practical_mean",
        "max_pa_practical_std",
        "sine_max_pa_practical_mean",
        "sine_max_pa_practical_std",
        "mean_pa_practical_mean",
        "mean_pa_practical_std",
    ]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=summary_cols).to_csv(out, index=False)
    print(f"[SKIP] {reason}.  Empty results written → {out}")


def get_gt_eigs_indices(gt_eigs: np.ndarray, pred_eigs: np.ndarray) -> np.ndarray:
    """Find indices of predicted eigenvalues that best match ground truth."""
    eig_diff = np.abs(gt_eigs[:, None] - pred_eigs[None, :])
    gt_idx, pred_idx = optimize.linear_sum_assignment(eig_diff)
    return pred_idx[np.argsort(gt_idx)]


def cluster_scores(
    scores: dict[str, np.ndarray],
    clustering_config: dict[str, Any],
) -> tuple[np.ndarray, int | float]:
    """Cluster scores and return labels and order estimate."""
    """
    normalization: str | None = "min_mafx",
        strategy: str = "vote",
        pilot_feature: str = "",
        weights: dict[str, float] | None = None,
        algorithm: str = "kmeans",
        gmm_covariance: str = "full",
        random_state: int | None = None,
    """
    try:
        clustering = ModeClustering(
            algorithm=clustering_config.get("algorithm", "gmm"),
            strategy=clustering_config.get("strategy", "mean"),
            normalization=clustering_config.get("normalization", "min_max"),
        )
        labels = clustering.fit(scores).labels_
        return labels.astype(int), int(labels.sum())
    except ValueError:
        # Clustering failed (e.g. empty cluster)
        n_modes = len(next(iter(scores.values())))
        return np.zeros(n_modes, dtype=int), np.nan


def compute_subspace_proximity_stats(
    clean_angles: np.ndarray,
    practical_angles: np.ndarray,
) -> dict[str, float]:
    """
    Compute summary statistics from principal angle arrays.

    Returns dictionary with 6 keys:
        - max_pa_clean, sine_max_pa_clean, mean_pa_clean
        - max_pa_practical, sine_max_pa_practical, mean_pa_practical
    """
    return {
        "max_pa_clean": float(clean_angles.max()),
        "sine_max_pa_clean": float(np.sin(clean_angles.max())),
        "mean_pa_clean": float(clean_angles.mean()),
        "max_pa_practical": float(practical_angles.max()),
        "sine_max_pa_practical": float(np.sin(practical_angles.max())),
        "mean_pa_practical": float(practical_angles.mean()),
    }


def compute_subspace_proximity_analysis(
    noisy_signal: np.ndarray,
    clean_signal: np.ndarray,
    true_modes: np.ndarray,
    true_eigenvalues: np.ndarray,
    num_modes: int,
    num_delays: int,
) -> dict[str, float]:
    """
    Compute subspace proximity summary statistics.

    Returns dictionary with 6 keys: max/sine_max/mean for clean and practical angles.
    """
    # Delay embed both signals
    delay_emb_noisy = DelayEmbedding(num_delays=num_delays).transform(noisy_signal)
    delay_emb_clean = DelayEmbedding(num_delays=num_delays).transform(clean_signal)

    # Compute principal angles
    angle_results = compute_subspace_principal_angles(
        embedded_data=delay_emb_noisy[:, :-1],
        true_modes=true_modes,
        true_eigenvalues=true_eigenvalues,
        num_modes=num_modes,
        embedded_signal=delay_emb_clean[:, :-1],
    )

    # Extract summary statistics
    return compute_subspace_proximity_stats(
        clean_angles=angle_results["clean_subspace_angles"],
        practical_angles=angle_results["practical_subspace_angles"],
    )


class MethodEvaluator:
    """Evaluates all enabled methods and produces order estimates."""

    def __init__(
        self,
        enabled_methods: list[str],
        clustering_config: dict,
        options_config: dict,
        num_delays: int,
        spatial_dim: int,
        dt: float = 1.0,
    ):
        self.enabled_methods = enabled_methods
        self.clustering_config = clustering_config
        self.options_config = options_config
        self.num_delays = num_delays
        self.spatial_dim = spatial_dim
        self.dt = dt

    def evaluate(
        self,
        signal: np.ndarray,
        proj_dmd,
        exact_dmd,
        max_rank: int,
        plot: bool = False,
    ) -> tuple[dict[str, int], dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Evaluate all methods and return order estimates, masks, and scores.

        Returns:
            order_estimates: {method_name: order}
            pred_masks: {method_name: binary_labels}
            pred_scores: {method_name: per_mode_scalar_score} for methods in
                ROC_AUC_METHODS only. Score is oriented "higher = more likely
                true mode"; ROC/PR-AUC are rank-based so scaling is irrelevant.
                Score order matches proj_dmd.eigs / true_mask ordering (see
                `compute_dmd_and_ground_truth` invariant).
        """
        order_estimates = {}
        pred_masks = {}
        pred_scores: dict[str, np.ndarray] = {}
        scores_cache = {}  # Cache for feature scores

        # Align modes
        aligned_proj_modes, _ = align_modes_and_amplitudes_phases(
            proj_dmd.modes, proj_dmd.amplitudes
        )
        aligned_ex_modes, _ = align_modes_and_amplitudes_phases(
            exact_dmd.modes, exact_dmd.amplitudes
        )

        # BIC method
        if "BIC" in self.enabled_methods:
            order_estimates["BIC"] = compute_bic_rank(
                signal, max_rank=max_rank, num_delays=self.num_delays, plot=plot
            )

        # GAP statistic
        if "GAP" in self.enabled_methods:
            order_estimates["GAP"] = gap_ranks(signal, num_delays=self.num_delays)

        # ExactModeNorm
        if "ExactModeNorm" in self.enabled_methods:
            exact_mode_norms = np.linalg.norm(aligned_ex_modes, axis=0)
            scores = {"ExactModeNorm": exact_mode_norms}
            labels, order = cluster_scores(scores, self.clustering_config)
            order_estimates["ExactModeNorm"] = order
            pred_masks["ExactModeNorm"] = labels

            if plot:
                scatter_scores_1d(
                    exact_mode_norms, "Exact-Mode-Norms", "ExactModeNorm", show_id=True
                )

        if "EigenvalueMagnitude" in self.enabled_methods:
            magnitudes = np.abs(exact_dmd.eigs)
            labels, order = cluster_scores(
                {"EigenvalueMagnitude": magnitudes}, self.clustering_config
            )
            order_estimates["EigenvalueMagnitude"] = order
            pred_masks["EigenvalueMagnitude"] = labels

            if plot:
                scatter_scores_1d(
                    magnitudes, "Magnitude", "EigenvalueMagnitude", show_id=True
                )
        if "ESR-Energy" in self.enabled_methods:
            esr = EstimatedSubspaceResidual()
            scores = esr.compute_features(
                exact_modes=aligned_ex_modes,  # shape (D*L, M)
                eigenvalues=exact_dmd.eigs,  # match exact modes
                plot=plot,
            )
            scores_cache["ESR"] = scores  # Save to cache
            labels, order = cluster_scores(scores, self.clustering_config)
            order_estimates["ESR-Energy"] = order
            pred_masks["ESR-Energy"] = labels
            pred_scores["ESR-Energy"] = scores["Estimated-Subspace-Residual"]

        if "ResDMDResidual" in self.enabled_methods:
            resdmd = ResDMDResidual(num_delays=self.num_delays)
            scores = resdmd.compute_features(
                signal=signal,
                exact_dmd=exact_dmd,
                plot=plot,
            )
            labels, order = cluster_scores(scores, self.clustering_config)
            order_estimates["ResDMDResidual"] = order
            pred_masks["ResDMDResidual"] = labels
            pred_scores["ResDMDResidual"] = scores["ResDMDResidual"]

        # Delay embedding methods
        if self.num_delays > 1:
            delay_methods = [
                m for m in self.enabled_methods if m in DELAY_EMBEDDING_METHODS
            ]
            if delay_methods:
                self._evaluate_delay_methods(
                    delay_methods,
                    proj_dmd.eigs,
                    aligned_proj_modes,
                    aligned_ex_modes,
                    scores_cache,
                    order_estimates,
                    pred_masks,
                    pred_scores,
                    plot,
                )
        else:
            # Mark delay embedding methods as N/A
            for method in DELAY_EMBEDDING_METHODS:
                if method in self.enabled_methods:
                    order_estimates[method] = np.nan
                    pred_masks[method] = np.full(max_rank, np.nan)

        return order_estimates, pred_masks, pred_scores

    def _evaluate_delay_methods(
        self,
        methods: list[str],
        eigenvalues: np.ndarray,
        aligned_proj_modes: np.ndarray,
        aligned_ex_modes: np.ndarray,
        scores_cache: dict,
        order_estimates: dict,
        pred_masks: dict,
        pred_scores: dict,
        plot: bool,
    ):
        """Evaluate delay-embedding-based methods."""

        # --- Determine what features need to be computed ---
        features_to_compute = set()
        for method in methods:
            if "+" in method:
                # Combination method: add all components
                features_to_compute.update(method.split("+"))
            else:
                # Individual method
                features_to_compute.add(method)

        # --- Compute all needed features and save to cache ---

        if "STC" in features_to_compute:
            stc = STC(
                num_delays=self.num_delays,
                dt=self.dt,
            )
            scores_cache["STC"] = stc.compute_features(
                eigenvalues=eigenvalues,
                modes=aligned_proj_modes,
                plot=plot and "STC" in methods,
            )

        if "NestedDMD" in features_to_compute:
            nested_dmd = NestedDMD(
                num_delays=self.num_delays,
                spatial_dim=self.spatial_dim,
            )
            scores_cache["NestedDMD"] = nested_dmd.compute_features(
                modes=aligned_proj_modes,
                eigenvalues=eigenvalues,
                plot=plot and "NestedDMD" in methods,
            )

        if "FixedEigenvalueKVFit" in features_to_compute:
            febvf = FixedEigenvalueKVFit(
                num_delays=self.num_delays,
                spatial_dim=self.spatial_dim,
            )
            scores_cache["FixedEigenvalueKVFit"] = febvf.compute_features(
                modes=aligned_proj_modes,
                eigenvalues=eigenvalues,
                plot=plot and "FixedEigenvalueKVFit" in methods,
            )

        if "ESR" in features_to_compute and "ESR" not in scores_cache:
            esr = EstimatedSubspaceResidual()
            scores_cache["ESR"] = esr.compute_features(
                exact_modes=aligned_ex_modes,
                eigenvalues=eigenvalues,
                plot=False,
            )

        # --- Cluster individual methods ---

        if "STC" in methods:
            labels, order = cluster_scores(scores_cache["STC"], self.clustering_config)
            order_estimates["STC"] = order
            pred_masks["STC"] = labels
            pred_scores["STC"] = scores_cache["STC"]["STC"]

        if "NestedDMD" in methods:
            labels, order = cluster_scores(
                scores_cache["NestedDMD"], self.clustering_config
            )
            order_estimates["NestedDMD"] = order
            pred_masks["NestedDMD"] = labels
            pred_scores["NestedDMD"] = scores_cache["NestedDMD"]["Reconstruction"]

        if "FixedEigenvalueKVFit" in methods:
            labels, order = cluster_scores(
                scores_cache["FixedEigenvalueKVFit"], self.clustering_config
            )
            order_estimates["FixedEigenvalueKVFit"] = order
            pred_masks["FixedEigenvalueKVFit"] = labels
            pred_scores["FixedEigenvalueKVFit"] = scores_cache["FixedEigenvalueKVFit"]["KV-Fit"]

        # --- Cluster combinations (features guaranteed in cache) ---

        if "NestedDMD+ESR" in methods:
            combined_scores = {**scores_cache["NestedDMD"], **scores_cache["ESR"]}
            labels, order = cluster_scores(combined_scores, self.clustering_config)
            order_estimates["NestedDMD+ESR"] = order
            pred_masks["NestedDMD+ESR"] = labels

        if "FixedEigenvalueKVFit+ESR" in methods:
            combined_scores = {
                **scores_cache["FixedEigenvalueKVFit"],
                **scores_cache["ESR"],
            }
            labels, order = cluster_scores(combined_scores, self.clustering_config)
            order_estimates["FixedEigenvalueKVFit+ESR"] = order
            pred_masks["FixedEigenvalueKVFit+ESR"] = labels


class MetricsTracker:
    """Tracks metrics across iterations."""

    def __init__(self, methods: list[str]):
        self.metrics = defaultdict(
            lambda: dict(
                tp=0,
                fp=0,
                fn=0,
                tn=0,
                order_diffs=[],
                order_hits=[],
                roc_aucs=[],  # per-iteration roc_auc_score; only filled for ROC_AUC_METHODS
                pr_aucs=[],  # per-iteration average_precision_score
            )
        )
        # Subspace proximity stats (collected per iteration)
        self.subspace_stats = {
            "max_pa_clean": [],
            "sine_max_pa_clean": [],
            "mean_pa_clean": [],
            "max_pa_practical": [],
            "sine_max_pa_practical": [],
            "mean_pa_practical": [],
        }

    def update(
        self,
        method: str,
        pred_order: int,
        pred_mask: np.ndarray | None,
        true_mask: np.ndarray,
        true_order: int,
        pred_score: np.ndarray | None = None,
    ):
        """Update metrics for a method.

        Per-iteration ROC-AUC and PR-AUC are computed and accumulated only
        when ``method`` is in ``ROC_AUC_METHODS`` and ``pred_score`` is given;
        the average over iterations is reported in ``to_dataframe``. ROC/PR-AUC
        are rank-based, so the score's scale is irrelevant — only its
        orientation ("higher = more likely true mode"), which the score-key
        map (`_ROC_AUC_SCORE_KEY`) is responsible for guaranteeing.
        """
        rec = self.metrics[method]
        rec["order_diffs"].append(pred_order - true_order)
        rec["order_hits"].append(int(pred_order == true_order))

        if pred_mask is not None:
            L = min(len(pred_mask), len(true_mask))
            p = pred_mask[:L].astype(bool)
            t = true_mask[:L].astype(bool)
            rec["tp"] += int((p & t).sum())
            rec["fp"] += int((p & ~t).sum())
            rec["fn"] += int((~p & t).sum())
            rec["tn"] += int((~p & ~t).sum())

        if method in ROC_AUC_METHODS and pred_score is not None:
            L = min(len(pred_score), len(true_mask))
            t = true_mask[:L].astype(int)
            s = pred_score[:L]
            # roc_auc / pr_auc are undefined if only one class is present
            # in the truth or if scores contain NaN; skip silently in that case.
            if 0 < t.sum() < t.size and np.all(np.isfinite(s)):
                rec["roc_aucs"].append(float(roc_auc_score(t, s)))
                rec["pr_aucs"].append(float(average_precision_score(t, s)))

    def add_subspace_proximity_stats(self, stats: dict[str, float]):
        """Add subspace proximity statistics from current iteration."""
        for key, value in stats.items():
            self.subspace_stats[key].append(value)

    def to_dataframe(
        self,
        methods: list[str],
        config: dict,
        mean_time_sec: float,
    ) -> pd.DataFrame:
        """Convert metrics to DataFrame."""
        summary_rows = []

        for method in methods:
            rec = self.metrics[method]
            diffs = np.array(rec["order_diffs"])
            abs_diffs = np.abs(diffs)

            summary = config.copy()
            summary.update(
                method=method,
                abs_diff_mean=abs_diffs.mean() if abs_diffs.size else np.nan,
                abs_diff_std=abs_diffs.std(ddof=0) if abs_diffs.size else np.nan,
                bias_frac_over=np.mean(diffs > 0) if diffs.size else np.nan,
                bias_frac_under=np.mean(diffs < 0) if diffs.size else np.nan,
                bias_skewness=(
                    scipy.stats.skew(diffs)
                    if diffs.size and not np.all(diffs == diffs[0])
                    else np.nan
                ),
                order_hit_prob=np.mean(rec["order_hits"]),
                mean_time_sec=mean_time_sec,
                precision=np.nan,
                recall=np.nan,
                f1=np.nan,
                accuracy=np.nan,
                tp=np.nan,
                fp=np.nan,
                fn=np.nan,
                tn=np.nan,
                roc_auc_mean=np.nan,
                roc_auc_std=np.nan,
                pr_auc_mean=np.nan,
                pr_auc_std=np.nan,
            )

            # ROC-AUC / PR-AUC averaged across Monte Carlo iterations
            # (per-iteration AUC, then mean/std across iterations). Only
            # populated for methods in ROC_AUC_METHODS.
            roc_aucs = np.asarray(rec["roc_aucs"], dtype=float)
            pr_aucs = np.asarray(rec["pr_aucs"], dtype=float)
            if roc_aucs.size > 0:
                summary.update(
                    roc_auc_mean=float(roc_aucs.mean()),
                    roc_auc_std=(
                        float(roc_aucs.std(ddof=1)) if roc_aucs.size > 1 else 0.0
                    ),
                    pr_auc_mean=float(pr_aucs.mean()),
                    pr_auc_std=(
                        float(pr_aucs.std(ddof=1)) if pr_aucs.size > 1 else 0.0
                    ),
                )

            # Add binary classification metrics if available
            if rec["tp"] + rec["fp"] + rec["fn"] + rec["tn"] > 0:
                tp, fp, fn, tn = rec["tp"], rec["fp"], rec["fn"], rec["tn"]
                summary.update(
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    tn=tn,
                    accuracy=(tp + tn) / (tp + fp + fn + tn),
                    precision=tp / (tp + fp) if (tp + fp) else np.nan,
                    recall=tp / (tp + fn) if (tp + fn) else np.nan,
                    f1=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
                )

            summary_rows.append(summary)

        # Add subspace proximity statistics (mean and std across iterations)
        subspace_summary = {}
        for key, values in self.subspace_stats.items():
            if values:
                arr = np.array(values)
                subspace_summary[f"{key}_mean"] = arr.mean()
                subspace_summary[f"{key}_std"] = arr.std(ddof=1)
            else:
                subspace_summary[f"{key}_mean"] = np.nan
                subspace_summary[f"{key}_std"] = np.nan

        # Add subspace stats to all rows
        for row in summary_rows:
            row.update(subspace_summary)

        return pd.DataFrame(summary_rows)


def compute_dmd_and_ground_truth(
    signal: np.ndarray,
    max_rank: int,
    num_delays: int,
    gt_eigs: np.ndarray,
) -> tuple:
    """Compute DMD results and ground truth matching."""
    proj_dmd = fit_dmd(
        signal, svd_rank=max_rank, mode="projected", num_delays=num_delays
    )
    exact_dmd = fit_dmd(signal, svd_rank=max_rank, mode="exact", num_delays=num_delays)

    # Per-eigenpair classification metrics (TP/FP/FN/TN/accuracy/precision/
    # recall/F1 in MetricsTracker.update) compare pred_mask[i] against
    # true_mask[i] elementwise. true_mask is built in proj_dmd.eigs order
    # below; for that comparison to be meaningful, every method's pred_mask
    # must be in the same ordering. Methods scored off exact_dmd (e.g.
    # ESR-Energy, ResDMDResidual after align_by_eigs) rely on this assertion.
    if not np.allclose(proj_dmd.eigs, exact_dmd.eigs, rtol=1e-10, atol=1e-12):
        raise AssertionError(
            "proj_dmd.eigs and exact_dmd.eigs differ in ordering or value; "
            "per-eigenpair classification metrics for methods scored off "
            "exact_dmd would be misaligned with true_mask."
        )

    true_idx = get_gt_eigs_indices(gt_eigs, proj_dmd.eigs)
    true_mask = np.zeros(max_rank, dtype=int)
    true_mask[true_idx] = 1

    return proj_dmd, exact_dmd, true_mask


def run_experiment(job_config: dict, plot: bool = False) -> None:
    """
    Run the experiment for a single job configuration.

    Args:
        job_config: Job configuration dictionary
        plot: Whether to generate plots
    """
    start_time = time.time()

    # Extract configuration
    params = job_config["parameters"]
    static = job_config["static_args"]
    methods = job_config["methods"]
    clustering = job_config["clustering"]
    options = job_config.get("options", {})
    output_path = job_config["output_path"]
    random_seed = job_config["random_seed"]
    base_random_seed = job_config["base_random_seed"]
    job_id = job_config["job_id"]

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    print(f"Job {job_id} started - output directory: {output_path_obj.parent}")

    # Extract parameters
    snr_db = params["snr_db"]
    freq_sep = params["freq_sep"]
    eig_mag = params["eig_mag"]
    eig_mag_spread = params.get("eig_mag_spread", 0.0)
    delays_over_timesteps = params["delays_over_timesteps"]
    temporal_dim = params["temporal_dim"]
    spatial_dim = params["spatial_dim"]
    num_modes = params["num_modes"]
    n_independent_modes = params.get("n_independent_modes")
    top_amplitude = params["top_amplitude"]
    max_rank = params["max_rank"]
    artificial_damping = params.get("artificial_damping", 1.0)
    noise_mode = params["noise_mode"]

    n_iter = static["n_iter"]
    dt = static["dt"]
    rho_mode = static.get("rho_mode", "random")

    # Compute derived parameters
    num_delays = round(delays_over_timesteps * temporal_dim)

    # Check dimension limits
    MAX_LD = 15_000
    if num_delays * spatial_dim > MAX_LD:
        _skip_run_with_empty_csv(
            output_path,
            reason=f"num_delays*spatial_dim = {num_delays * spatial_dim} > {MAX_LD}",
        )
        return

    # Initialize
    metrics_tracker = MetricsTracker(methods)
    method_evaluator = MethodEvaluator(
        enabled_methods=methods,
        clustering_config=clustering,
        options_config=options,
        num_delays=num_delays,
        spatial_dim=spatial_dim,
        dt=dt,
    )

    # Run iterations
    for i in tqdm(range(n_iter), desc=f"Job {job_id}"):
        plot_iter = plot and i == 0
        iter_seed = random_seed + i

        # Generate data
        gen = DMDDataGenerator(
            eigenvalue_magnitude=eig_mag,
            eigenvalue_magnitude_spread=eig_mag_spread,
            rho_mode=rho_mode,
            frequency_separation=freq_sep,
            snr_db=snr_db,
            top_amplitude=top_amplitude,
            dt=dt,
            noise_mode=noise_mode,
            random_seed=iter_seed,
        )
        sig, sig_clean, gt_eigs, modes, amps = gen.generate(
            spatial_dim,
            temporal_dim,
            num_modes,
            n_independent_modes=n_independent_modes,
        )

        # Apply artificial damping if needed
        if artificial_damping < 1.0:
            damping_dynamic = artificial_damping ** np.arange(temporal_dim)
            sig = sig * damping_dynamic[None, :]

        if plot_iter:
            imshow_complex(sig, title="Noisy Data")
            plot_mode_table(gt_eigs, amps)

        # Compute DMD
        proj_dmd, exact_dmd, true_mask = compute_dmd_and_ground_truth(
            sig, max_rank, num_delays, gt_eigs
        )

        # Subspace proximity results
        proximity_stats = compute_subspace_proximity_analysis(
            sig, sig_clean, modes, gt_eigs, num_modes, num_delays
        )
        metrics_tracker.add_subspace_proximity_stats(proximity_stats)

        # Evaluate all methods
        order_estimates, pred_masks, pred_scores = method_evaluator.evaluate(
            signal=sig,
            proj_dmd=proj_dmd,
            exact_dmd=exact_dmd,
            max_rank=max_rank,
            plot=plot_iter,
        )

        # Update metrics
        for method in methods:
            if method in order_estimates:
                metrics_tracker.update(
                    method=method,
                    pred_order=order_estimates[method],
                    pred_mask=pred_masks.get(method),
                    true_mask=true_mask,
                    true_order=num_modes,
                    pred_score=pred_scores.get(method),
                )

    # Compute aggregated results
    mean_time = (time.time() - start_time) / n_iter

    cfg_dict = dict(
        snr_db=snr_db,
        freq_sep=freq_sep,
        eig_mag=eig_mag,
        eig_mag_spread=eig_mag_spread,
        rho_mode=rho_mode,
        delays_over_timesteps=delays_over_timesteps,
        num_delays=num_delays,
        temporal_dim=temporal_dim,
        spatial_dim=spatial_dim,
        num_modes=num_modes,
        n_independent_modes=n_independent_modes,
        top_amplitude=top_amplitude,
        max_rank=max_rank,
        artificial_damping=artificial_damping,
        n_iter=n_iter,
        dt=dt,
        noise_mode=noise_mode,
        base_random_seed=base_random_seed,
        random_seed=random_seed,
    )

    summary_df = metrics_tracker.to_dataframe(methods, cfg_dict, mean_time)

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    # Print summary
    print(f"\nJob {job_id} completed in {time.time() - start_time:.1f}s")
    print(f"  Results saved to: {output_path}")
    print("\nMetrics Summary:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    print(
        summary_df[
            [
                "method",
                "abs_diff_mean",
                "order_hit_prob",
                "bias_frac_over",
                "bias_frac_under",
                "accuracy",
                "precision",
                "recall",
                "roc_auc_mean",
                "pr_auc_mean",
            ]
        ]
    )


def run(config_path: str, plot: bool = False) -> None:
    """
    Run a single experiment job from YAML config.

    Args:
        config_path: Path to job configuration YAML file
        plot: Whether to generate plots (default: False)
    """
    # Load job configuration
    with open(config_path, "r", encoding="utf-8") as f:
        job_config = yaml.safe_load(f)

    # Run experiment
    try:
        run_experiment(job_config, plot=plot)
    except Exception as e:
        print(f"✗ Job failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    fire.Fire({"run": run})
