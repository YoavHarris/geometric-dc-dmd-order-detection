#!/usr/bin/env python3
"""
Parameter sweep generator.
Generates parameter combinations for experiment jobs.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Union
import numpy as np


def _gen_range(spec: Dict[str, Any]) -> List[Union[int, float]]:
    """Generate values for range-type parameter."""
    start, end, steps = spec["start"], spec["end"], spec["num_steps"]
    scale = spec.get("scale", "lin")
    
    if scale == "log":
        vals = np.logspace(np.log10(start), np.log10(end), steps)
    else:
        vals = np.linspace(start, end, steps)
    
    if spec.get("as_int", False):
        return [int(round(v)) for v in vals]
    
    return vals.tolist()


def _gen_list(spec: Dict[str, Any]) -> List[Any]:
    """Generate values for list-type parameter."""
    return spec["values"]


def _gen_const(spec: Dict[str, Any]) -> List[Any]:
    """Generate values for const-type parameter."""
    return [spec["value"]]


_GENERATORS = {
    "range": _gen_range,
    "list": _gen_list,
    "const": _gen_const,
}


def _deduplicate_jobs(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate parameter combinations."""
    seen: set[tuple] = set()
    unique: list[Dict[str, Any]] = []
    
    for d in jobs:
        key = tuple(sorted(d.items()))
        if key not in seen:
            unique.append(d)
            seen.add(key)
    
    return unique


def generate_parameter_combinations(
    parameters: Dict[str, Dict[str, Any]],
    working_point: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Generate all parameter combinations based on roles.
    
    Args:
        parameters: Parameter specifications from config
        working_point: Working point values (required if any param has role='wp')
        
    Returns:
        List of parameter dictionaries, one per job
    """
    # Separate parameters by role
    wp_params: List[str] = []
    cart_params: List[str] = []
    param_values: Dict[str, List[Any]] = {}
    
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
        
        wp_set: List[Dict[str, Any]] = []
        
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
            dict(zip(cart_params, combo))
            for combo in itertools.product(*cart_axes)
        ]
    else:
        cart_set = [{}]
    
    # Final product: wp_set × cart_set
    all_combinations: List[Dict[str, Any]] = []
    for w in wp_set:
        for c in cart_set:
            merged = {**w, **c}
            all_combinations.append(merged)
    
    return all_combinations


def generate_job_parameters(
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
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
    
    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)
    
    combos = generate_job_parameters(config)
    print(f"Generated {len(combos)} parameter combinations")
    
    if len(combos) <= 10:
        for i, combo in enumerate(combos):
            print(f"\nJob {i}:")
            for k, v in combo.items():
                print(f"  {k}: {v}")

