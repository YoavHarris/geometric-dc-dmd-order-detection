#!/usr/bin/env python3
"""
Parameter sweep generator.
Generates parameter combinations for experiment jobs.
"""

from __future__ import annotations

import itertools
from typing import Any
import numpy as np


# Maximum decimal places to prevent over-rounding
MAX_DECIMAL_PLACES = 15


def _compute_decimal_places(step_size: float, margin: int = 2) -> int:
    """
    Compute decimal places needed to distinguish adjacent grid points.
    Uses step size to adaptively determine precision: smaller steps need more decimals.
    Adds margin for safety and caps at MAX_DECIMAL_PLACES.
    """
    if step_size == 0 or not np.isfinite(step_size):
        return 0

    decimals = max(0, int(np.ceil(-np.log10(abs(step_size)))) + margin)
    return min(decimals, MAX_DECIMAL_PLACES)


def _gen_range(spec: dict[str, Any]) -> list[int | float]:
    """Generate values for range-type parameter."""
    start, end, steps = spec["start"], spec["end"], spec["num_steps"]
    scale = spec.get("scale", "lin")

    if steps == 1:
        val = start
        if spec.get("as_int", False):
            return [int(round(val))]
        return [float(val)]

    if scale == "log":
        # Generate in log space and round there for proper scaling
        log_start, log_end = np.log10(start), np.log10(end)
        log_step = (log_end - log_start) / (steps - 1)
        decimals = _compute_decimal_places(log_step)

        log_vals = np.linspace(log_start, log_end, steps)
        log_vals = np.round(log_vals, decimals=decimals)
        vals = np.power(10.0, log_vals)

        # Snap endpoints to exact values
        vals[0], vals[-1] = start, end
    else:
        # Linear spacing with adaptive precision
        step = (end - start) / (steps - 1)
        decimals = _compute_decimal_places(step)

        vals = np.linspace(start, end, steps)
        vals = np.round(vals, decimals=decimals)

        # Snap endpoints to exact values
        vals[0], vals[-1] = start, end

    if spec.get("as_int", False):
        return [int(round(v)) for v in vals]

    return vals.tolist()


def _gen_list(spec: dict[str, Any]) -> list[Any]:
    """Generate values for list-type parameter."""
    return spec["values"]


def _gen_const(spec: dict[str, Any]) -> list[Any]:
    """Generate values for const-type parameter."""
    return [spec["value"]]


_GENERATORS = {
    "range": _gen_range,
    "list": _gen_list,
    "const": _gen_const,
}


def _deduplicate_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate parameter combinations."""
    seen: set[tuple] = set()
    unique: list[dict[str, Any]] = []

    for d in jobs:
        key = tuple(sorted(d.items()))
        if key not in seen:
            unique.append(d)
            seen.add(key)

    return unique


def generate_parameter_combinations(
    parameters: dict[str, dict[str, Any]],
    working_point: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Generate all parameter combinations based on roles.

    Args:
        parameters: Parameter specifications from config
        working_point: Working point values (required if any param has role='wp')

    Returns:
        List of parameter dictionaries, one per job
    """
    # Separate parameters by role
    wp_params: list[str] = []
    cart_params: list[str] = []
    param_values: dict[str, list[Any]] = {}

    for name, spec in parameters.items():
        ptype = spec["type"]
        vals = _GENERATORS[ptype](spec)
        param_values[name] = vals

        role = spec.get("role", "cartesian")
        if role == "wp":
            wp_params.append(name)
        else:
            cart_params.append(name)

    # Generate working-point scan combinations
    if wp_params:
        if working_point is None:
            raise ValueError("working_point required when parameters have role='wp'")

        wp_set: list[dict[str, Any]] = []

        # Start with exact working point
        base = working_point.copy()
        wp_set.append(base)

        # Add one-axis-at-a-time variations
        for p in wp_params:
            wp_val_list = param_values[p]
            for v in wp_val_list:
                if v == working_point[p]:
                    continue  # Already in base
                d = working_point.copy()
                d[p] = v
                wp_set.append(d)

        # Deduplicate
        wp_set = _deduplicate_jobs(wp_set)
    else:
        # No working point params, just use empty dict
        wp_set = [{}]

    # Generate Cartesian product combinations
    if cart_params:
        cart_axes = [param_values[p] for p in cart_params]
        cart_set = [
            dict(zip(cart_params, combo)) for combo in itertools.product(*cart_axes)
        ]
    else:
        cart_set = [{}]

    # Final product: wp_set × cart_set
    all_combinations: list[dict[str, Any]] = []
    for w in wp_set:
        for c in cart_set:
            merged = {**w, **c}
            all_combinations.append(merged)

    # Ensure strict uniqueness over the FULL parameter dict before returning.
    deduped = _deduplicate_jobs(all_combinations)

    # (Optional) sanity check: fail fast if duplicates were generated.
    if len(deduped) != len(all_combinations):
        dup_count = len(all_combinations) - len(deduped)
        raise RuntimeError(
            f"Parameter generator produced {dup_count} duplicate point(s). "
            "This indicates overlap between working-point variants and the cartesian grid."
        )

    return deduped


def generate_job_parameters(config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Generate all job parameter combinations from config.

    Args:
        config: Full experiment configuration

    Returns:
        List of parameter dictionaries, one per job
    """
    parameters = config["parameters"]
    working_point = config.get("generator", {}).get("working_point")

    return generate_parameter_combinations(parameters, working_point)


if __name__ == "__main__":
    import yaml
    import sys

    if len(sys.argv) != 2:
        print("Usage: python param_generator.py <config.yaml>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    combos = generate_job_parameters(config)
    print(f"Generated {len(combos)} parameter combinations")

    if len(combos) <= 10:
        for i, combo in enumerate(combos):
            print(f"\nJob {i}:")
            for k, v in combo.items():
                print(f"  {k}: {v}")
