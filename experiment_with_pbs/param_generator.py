"""
Parameter‑set generator
----------------------
• Each *parameter* now declares a **role**
    role: "wp"         → varies one‑axis‑at‑a‑time around the working‑point
    role: "cartesian"  → part of the full Cartesian product
  (missing role defaults to "cartesian" → backward‑compatible)
• Two exploration modes remain:
    mode="cartesian"  → ignore roles, take full Cartesian product of *all* params
    mode="around"     → build the product  ( single‑axis‑WP‑scans × Cartesian‑params )

"""

from __future__ import annotations

import itertools
import stat
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Union


# ────────────────────────────── helpers ────────────────────────────────


def _gen_range(spec: Dict[str, Any]) -> List[Union[int, float]]:
    start, end, steps = spec["start"], spec["end"], spec["num_steps"]
    scale = spec.get("scale", "lin")
    vals = (
        np.logspace(np.log10(start), np.log10(end), steps)
        if scale == "log"
        else np.linspace(start, end, steps)
    )
    if spec.get("as_int", False):
        return [int(round(v)) for v in vals]
    return vals.tolist()


def _gen_list(spec: Dict[str, Any]) -> List[Any]:
    return spec["values"]


def _gen_const(spec: Dict[str, Any]) -> List[Any]:
    return [spec["value"]]


_GENERATORS = {"range": _gen_range, "list": _gen_list, "const": _gen_const}


def _format_value(val: Any, sigfigs: int = 3) -> str:
    if isinstance(val, float):
        return f"{val:.{sigfigs}g}"
    if isinstance(val, str):
        return f'"{val}"'
    return str(val)


def _deduplicate_jobs(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[tuple] = set()
    unique: list[Dict[str, Any]] = []
    for d in jobs:
        key = tuple(sorted(d.items()))  # stable, hashable identifier
        if key not in seen:
            unique.append(d)
            seen.add(key)
    return unique


# ───────────────────────────── core builders ───────────────────────────


def _cartesian_product(
    names: List[str],
    param_values: Dict[str, List[Any]],
    static_args: Dict[str, Any],
    sigfigs: int,
) -> List[str]:
    job_args: list[str] = []
    for combo in itertools.product(*(param_values[n] for n in names)):
        parts = [f"--{n}={_format_value(v, sigfigs)}" for n, v in zip(names, combo)]
        parts += [
            f"--{k}={_format_value(v, sigfigs)}"
            for k, v in static_args.items()
            if v is not None
        ]
        job_args.append(" ".join(parts))
    return job_args


def _generate_job_list(
    wp_params: list[str],
    cart_params: list[str],
    param_values: Dict[str, List[Any]],
    working_point: Dict[str, Any],
    static_args: Dict[str, Any],
    sigfigs: int,
) -> List[str]:
    # --- 1. build wp_param_set (union of 1‑D scans) ------------------
    wp_set: list[dict[str, Any]] = []
    base = working_point.copy() if working_point else {}
    wp_set.append(base)  # include exact WP
    for p in wp_params:
        wp_val_list = param_values[p]
        for v in wp_val_list:
            if v == working_point[p]:
                continue  # will already be in base
            d = working_point.copy()
            d[p] = v
            wp_set.append(d)

    # deduplicate (in case range includes WP)
    uniq_wp_set = _deduplicate_jobs(wp_set)

    # --- 2. build cartesian_param_set --------------------------------
    cart_axes = [param_values[p] for p in cart_params] or [()]
    cart_set = [
        dict(zip(cart_params, combo)) for combo in itertools.product(*cart_axes)
    ]

    # --- 3. final product -------------------------------------------
    job_args: list[str] = []
    for w in uniq_wp_set:
        for c in cart_set:
            merged = {**w, **c}
            parts = [f"--{k}={_format_value(v, sigfigs)}" for k, v in merged.items()]
            parts += [
                f"--{k}={_format_value(v, sigfigs)}"
                for k, v in static_args.items()
                if v is not None
            ]
            job_args.append(" ".join(parts))
    return job_args


# ───────────────────────────── public API ──────────────────────────────


def generate_job_args(
    parameters: Dict[str, Dict[str, Any]],
    static_args: Dict[str, Any],
    *,
    sigfigs: int = 3,
    working_point: Dict[str, Any] | None = None,
) -> List[str]:
    """Return a list of CLI‑ready argument strings."""

    # 1. preprocess parameter specs -----------------------------------
    names: list[str] = []
    param_values: dict[str, list[Any]] = {}
    role_map: dict[str, str] = {}

    for name, spec in parameters.items():
        ptype = spec["type"]
        if ptype not in _GENERATORS:
            raise ValueError(f"Unknown parameter type '{ptype}' for '{name}'")
        vals = _GENERATORS[ptype](spec)
        names.append(name)
        param_values[name] = vals
        role_map[name] = spec.get("role", "cartesian")

    wp_params = [n for n, r in role_map.items() if r == "wp"]
    cart_params = [n for n, r in role_map.items() if r == "cartesian"]

    if wp_params and working_point is None:
        raise ValueError("working_point required because some parameters are 'wp'")

    validate_rank_dim_compat(param_values)

    return _generate_job_list(
        wp_params,
        cart_params,
        param_values,
        working_point,
        static_args,
        sigfigs,
    )


def validate_rank_dim_compat(param_values):
    M_max = np.max(param_values["max_rank"])
    N_min = np.min(param_values["temporal_dim"])
    N_max = np.max(param_values["temporal_dim"])
    tau_max = np.max(param_values["delays_over_timesteps"])
    L_max = tau_max * N_max
    if L_max > 1:
        assert (
            N_min * (1 - tau_max) >= M_max
        ), f"Rank and dimensions incompatible. Parameters must satisfy N-L > M for all configurations"


# ───────────────────────────── I/O helper ──────────────────────────────


def write_job_list(
    job_args: List[str], output_dir: Path, filename: str = "job_list.txt"
) -> str:
    """Write *job_list.txt* in *output_dir* and make it read‑only."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / filename
    with path.open("w") as f:
        f.write("\n".join(job_args) + "\n")
    # chmod 444
    path.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    return str(path)
