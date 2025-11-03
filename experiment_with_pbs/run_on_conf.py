#!/usr/bin/env python3
"""
Run DMD order-detection experiment for a parameter configuration.
Uses Fire for CLI. Prints runtime summary.
"""

import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy
from scipy import optimize
from tqdm import tqdm
import fire

from algorithms.mode_energy import ModeEnergyEvaluator
from algorithms.deprecated.block_vandermonde_matching import StructureModeEvaluator
from algorithms.clustering import ModeClustering
from algorithms.deprecated.dc_dmd import DcDMDModeSelector
from algorithms.info_criteria import (
    InformationCriteriaOrderEstimator,
    gap_ranks,
)
from algorithms.subspace_projection_energy import SubsetProjectionEvaluator
from algorithms.algo_utils import subspace_stats

from dmd.dmd_tools import DelayEmbedding
from utils.data_generation import DMDDataGenerator
from utils.dmd_utils import fit_dmd, align_modes_and_amplitudes_phases
from utils.visualizations import imshow_complex, plot_mode_table, scatter_scores_1d

STRICTLY_DELAY_EMB_METHODS = [
    "ModeEnergy",
    "ModeMatDMD",
    "BlockVandermondeMatching",
    "dcDMD",
]


def _skip_run_with_empty_csv(
    output_path: Union[str, Path],
    reason: str,
) -> None:
    """
    Abort the run, write a header‑only CSV, and print the skip reason.
    """
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
        "top_amplitude",
        "max_rank",
        "artificial_damping",
        "n_iter",
        "dt",
        "noise_mode",
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
    ]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=summary_cols).to_csv(out, index=False)
    print(f"[SKIP] {reason}.  Empty results written → {out}")
    return


def get_gt_eigs_indices(gt_eigs_: np.ndarray, pred_eigs_: np.ndarray) -> np.ndarray:
    eig_diff = np.abs(gt_eigs_[:, None] - pred_eigs_[None, :])  # (m, M)
    gt_idx, pred_idx = optimize.linear_sum_assignment(eig_diff)
    return pred_idx[np.argsort(gt_idx)]


def _cluster_scores(
    scores: Dict[str, np.ndarray],
    *,
    algo: str = "gmm",
    decider: str = "distance",
    norm: Optional[str] = "min_max",
) -> Tuple[np.ndarray, int]:
    """
    Thin wrapper around ModeClustering so every call looks identical.
    """
    labels = (
        ModeClustering(
            algorithm=algo,
            decider=decider,
            normalization_mode=norm,
        )
        .fit(scores)
        .labels_
    )
    return labels.astype(int), int(labels.sum())


def label_all_methods(
    pred_eigs: np.ndarray,
    aligned_proj_modes: np.ndarray,
    aligned_ex_modes: np.ndarray,
    num_delays: int,
    spatial_dim: int,
    cluster_algorithm: str = "kmeans",
    plot: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """
    Compute masks & order estimates for every method.

    Returns
    -------
    pred_mask_lookup : {method name → 0/1 labels}
    order_estimates  : {method name → integer order}
    """

    # ---- build score dictionaries exactly once -------------------
    score_tables: Dict[str, Dict[str, np.ndarray]] = {}
    pred_mask_lookup: Dict[str, np.ndarray] = {}
    order_estimates: Dict[str, int] = {}

    exact_mode_norms = np.linalg.norm(aligned_ex_modes, axis=0)
    scaled_residuals = exact_mode_norms / np.abs(pred_eigs) - 1
    if plot:
        scatter_scores_1d(np.log10(scaled_residuals), "log_scaled_res")
        scatter_scores_1d(
            exact_mode_norms,
            "ExactModeNorm",
            "Exact Mode Norm",
            show_id=True,
        )
    score_tables["ExactModeNorm"] = {"ExactModeNorm": exact_mode_norms}

    if num_delays > 1:
        # ---- dcDMD (non-score path) ----------------------------------
        dc_labels = (
            DcDMDModeSelector(num_delays=num_delays)
            .label_modes(pred_eigs, aligned_proj_modes)
            .astype(int)
        )
        pred_mask_lookup["dcDMD"] = dc_labels
        order_estimates["dcDMD"] = int(dc_labels.sum())

        # TODO: should we consider ignoring later blocks of the BV modes, following the Nyquist argument from dcDMD?
        # Mode Energy
        score_tables["ModeEnergy"] = ModeEnergyEvaluator(
            num_delays, spatial_dim
        ).compute_scores(aligned_ex_modes, plot=plot)

        from algorithms.mode_energy_and_phase import ModeEnergyPhaseEvaluator

        energy_phase_scores = ModeEnergyPhaseEvaluator(
            num_delays, spatial_dim
        ).compute_scores(aligned_ex_modes, pred_eigs, False, plot=plot)

        # Structure-based
        struct_scores = StructureModeEvaluator(num_delays, spatial_dim).compute_scores(
            aligned_proj_modes,
            eigenvalues=pred_eigs,
            singular_values_for_scaling=None,
            plot=plot,
        )
        score_tables["ModeMatDMD"] = struct_scores
        score_tables["BlockVandermondeMatching"] = {
            "BlockVandermondeRMS": struct_scores["Block-Vandermonde-RMS"],
            "BlockVandermondeCosine": struct_scores["Block-Vandermonde-Cosine"],
        }

    else:
        for k in STRICTLY_DELAY_EMB_METHODS:
            order_estimates[k] = np.nan
            pred_mask_lookup[k] = np.full_like(pred_eigs, np.nan)

    # ---- per-method clustering parameters ------------------------
    params = {
        "ModeEnergy": dict(algo=cluster_algorithm, decider="mean", norm="min_max"),
        "ModeMatDMD": dict(algo=cluster_algorithm, decider="mean", norm="min_max"),
        "BlockVandermondeMatching": dict(
            algo=cluster_algorithm, decider="mean", norm="min_max"
        ),
        "ExactModeNorm": dict(algo=cluster_algorithm, decider="mean"),
    }

    # ---- run clustering in one tidy loop -------------------------
    for name, scores in score_tables.items():
        labels, order = _cluster_scores(scores, **params[name])
        pred_mask_lookup[name] = labels
        order_estimates[name] = order

    return pred_mask_lookup, order_estimates


def compute_dmd_results(sig, max_rank, num_delays, gt_eigs):
    proj_dmd = fit_dmd(sig, svd_rank=max_rank, mode="projected", num_delays=num_delays)
    ex_dmd = fit_dmd(sig, svd_rank=max_rank, mode="exact", num_delays=num_delays)

    true_idx = get_gt_eigs_indices(gt_eigs, proj_dmd.eigs)
    true_mask = np.zeros(max_rank, dtype=int)
    true_mask[true_idx] = 1
    return proj_dmd, ex_dmd, true_mask


def update_metrics(
    method: str,
    pred_order: int,
    pred_mask: Optional[np.ndarray],
    true_mask: np.ndarray,
    true_order: int,
    bucket: Dict[str, Dict[str, any]],
) -> None:
    rec = bucket[method]
    rec["order_diffs"].append(pred_order - true_order)
    rec["order_hits"].append(int(pred_order == true_order))
    if pred_mask is None:
        return
    L = min(len(pred_mask), len(true_mask))
    p, t = pred_mask[:L].astype(bool), true_mask[:L].astype(bool)
    rec["tp"] += int((p & t).sum())
    rec["fp"] += int((p & ~t).sum())
    rec["fn"] += int((~p & t).sum())
    rec["tn"] += int((~p & ~t).sum())


def aggregate_and_save(
    metrics, run_cfg, mean_time_sec, estimator_names
) -> pd.DataFrame:
    summary_rows = []
    for method in estimator_names:
        rec = metrics[method]
        diffs = np.array(rec["order_diffs"])
        abs_diffs = np.abs(diffs)
        summary = run_cfg.copy()
        summary.update(
            dict(
                method=method,
                abs_diff_mean=abs_diffs.mean() if abs_diffs.size else np.nan,
                abs_diff_std=abs_diffs.std(ddof=0) if abs_diffs.size else np.nan,
                bias_frac_over=np.mean(diffs > 0) if diffs.size else np.nan,
                bias_frac_under=np.mean(diffs < 0) if diffs.size else np.nan,
                bias_skewness=(
                    (scipy.stats.skew(diffs) if diffs.size else np.nan)
                    if not np.all(diffs == diffs[0])
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
            )
        )
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
    if "SubspaceStats" in metrics:
        for key, values in metrics["SubspaceStats"].items():
            values_arr = np.array(values)
            for row in summary_rows:
                row[f"{key}_mean"] = values_arr.mean()
                row[f"{key}_std"] = values_arr.std(ddof=0)
    return pd.DataFrame(summary_rows)


def run(
    snr_db: float,
    freq_sep: float,
    delays_over_timesteps: float,
    temporal_dim: int,
    spatial_dim: int,
    num_modes: int,
    top_amplitude: float,
    max_rank: int,
    n_iter: int,
    dt: float,
    noise_mode: str,
    eig_mag: Union[float, Sequence[float]],
    eig_mag_spread: Optional[float] = None,
    rho_mode: str = "linspace",
    cluster_algorithm: str = "kmeans",
    artificial_damping: Optional[float] = None,
    random_seed: int = None,
    output_path: str = "results.csv",
    plot: bool = False,
) -> None:
    start_time = time.time()
    num_delays = round(delays_over_timesteps * temporal_dim)
    MAX_LD = 15_000
    if num_delays * spatial_dim > MAX_LD:
        _skip_run_with_empty_csv(
            output_path,
            reason=(
                f"num_delays*spatial_dim = {num_delays * spatial_dim} "
                f"> {MAX_LD} size limit"
            ),
        )
        return

    estimator_names = [
        "AIC",
        "AICc",
        "BIC",
        "GAP",
        "ExactModeNorm",
        "dcDMD",
        "ModeMatDMD",
        "ModeEnergy",
        "BlockVandermondeMatching",
        "MSMS",
        "CDFAUC",
    ]
    metrics = defaultdict(
        lambda: dict(tp=0, fp=0, fn=0, tn=0, order_diffs=[], order_hits=[])
    )
    subspace_stat_keys = ["overlap_rho", "chordal_dist", "sin_theta_max"]
    metrics["SubspaceStats"] = {key: [] for key in subspace_stat_keys}

    for i in tqdm(range(n_iter)):
        if i > 0:
            plot = False
        seed = (random_seed + i) if random_seed is not None else None
        gen = DMDDataGenerator(
            eigenvalue_magnitude=eig_mag,
            eigenvalue_magnitude_spread=eig_mag_spread,
            rho_mode=rho_mode,
            frequency_separation=freq_sep,
            snr_db=snr_db,
            top_amplitude=top_amplitude,
            dt=dt,
            noise_mode=noise_mode,
            random_seed=seed,
        )
        sig, sig_clean, gt_eigs, modes, amps = gen.generate(
            spatial_dim, temporal_dim, num_modes
        )

        if artificial_damping is not None and artificial_damping < 1.0:
            damping_dynamic = artificial_damping ** np.arange(temporal_dim)
            sig = sig * damping_dynamic[None, :]

        if plot:
            imshow_complex(sig, title="Noisy Data")
            plot_mode_table(gt_eigs, amps)

        info_est = InformationCriteriaOrderEstimator(num_delays=num_delays)
        info_orders = info_est.fit(sig, max_rank=max_rank, plot=plot)
        info_orders["GAP"] = gap_ranks(sig).item()

        proj_dmd, exact_dmd, true_mask = compute_dmd_results(
            sig, max_rank, num_delays, gt_eigs
        )
        pred_eigs = proj_dmd.eigs
        gt_inds = get_gt_eigs_indices(gt_eigs, pred_eigs)
        spur_inds = [i for i in range(max_rank) if i not in gt_inds]
        spur_eigs = pred_eigs[spur_inds]

        delay_emb_sig = DelayEmbedding(L=num_delays).transform(sig)
        U, s, Vh = np.linalg.svd(delay_emb_sig[:, :-1])
        U_M, V_M, s_M = U[:, :max_rank], Vh.conj().T[:, :max_rank], s[:max_rank]

        aligned_proj_modes, _ = align_modes_and_amplitudes_phases(
            proj_dmd.modes, proj_dmd.amplitudes
        )

        aligned_ex_modes, _ = align_modes_and_amplitudes_phases(
            exact_dmd.modes, exact_dmd.amplitudes
        )

        subspaces_stats = subspace_stats(
            aligned_ex_modes[:, gt_inds], U_M[:, :num_modes], num_modes
        )
        for key in subspace_stat_keys:
            metrics["SubspaceStats"][key].append(subspaces_stats[key])

        subset_projections = SubsetProjectionEvaluator(
            aligned_ex_modes, U_M, base_sat_tol=0.15
        )

        sq_norms = np.linalg.norm(aligned_ex_modes, axis=0) ** 2
        sq_rho = np.abs(pred_eigs) ** 2
        res = sq_norms - sq_rho

        subspace_projection_outputs = subset_projections.estimate_model_order(
            max_tol=0.4, plot=plot
        )

        pred_masks, order_estimates = label_all_methods(
            pred_eigs=proj_dmd.eigs,
            aligned_proj_modes=aligned_proj_modes,
            aligned_ex_modes=aligned_ex_modes,
            num_delays=num_delays,
            spatial_dim=spatial_dim,
            cluster_algorithm=cluster_algorithm,
            plot=plot,
        )
        order_estimates.update(info_orders)
        order_estimates["MSMS"], pred_masks["MSMS"] = subspace_projection_outputs[
            "saturated_subset"
        ]
        order_estimates["CDFAUC"], pred_masks["CDFAUC"] = subspace_projection_outputs[
            "mode_cdf_auc"
        ]

        for method, pred_order in order_estimates.items():
            update_metrics(
                method=method,
                pred_order=pred_order,
                pred_mask=pred_masks.get(method),
                true_mask=true_mask,
                true_order=num_modes,
                bucket=metrics,
            )

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
        top_amplitude=top_amplitude,
        max_rank=max_rank,
        artificial_damping=artificial_damping,
        n_iter=n_iter,
        dt=dt,
        noise_mode=noise_mode,
        random_seed=random_seed,
    )
    summary_df = aggregate_and_save(metrics, cfg_dict, mean_time, estimator_names)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)  # or use a large number like 200
    pd.set_option("display.max_colwidth", None)
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
            ]
        ]
    )


if __name__ == "__main__":
    fire.Fire({"run": run})
