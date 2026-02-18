#!/usr/bin/env python3
"""
Configuration validator for PBS experiments.
Validates all configuration parameters before job generation.
"""

import os
import sys
from typing import Any

from .param_generator import generate_job_parameters

# Constants
VALID_METHODS = {
    "BIC",
    "GAP",
    "ExactModeNorm",
    "EigenvalueMagnitude",
    "ESR-Energy",
    "STC",
    "NestedDMD",
    "FixedEigenvalueKVFit",
    "NestedDMD+ESR",
    "FixedEigenvalueKVFit+ESR",
}

VALID_CLUSTERING_ALGORITHMS = {"gmm", "kmeans"}
VALID_CLUSTERING_STRATEGIES = {"mean", "distance"}
VALID_CLUSTERING_NORMALIZATIONS = {"min_max", "standard", "null", None}

VALID_PARAMETER_TYPES = {"range", "list", "const"}
VALID_ROLES = {"wp", "cartesian"}
VALID_SCALES = {"lin", "log"}
VALID_NOISE_MODES = {"gaussian", "bi_gaussian", "student_t", "hetero"}
VALID_RHO_MODES = {"random", "fixed"}


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    pass


def validate_config(config: dict[str, Any]) -> None:
    """
    Validate entire configuration.

    Raises:
        ConfigValidationError: If validation fails
    """
    _validate_experiment_section(config)
    _validate_methods_section(config)
    _validate_clustering_section(config)
    _validate_generator_section(config)
    _validate_parameters_section(config)
    _validate_static_args_section(config)
    _validate_pbs_section(config)
    _validate_paths_section(config)
    _validate_output_section(config)
    _validate_parameter_compatibility(config)


def _validate_experiment_section(config: dict[str, Any]) -> None:
    """Validate experiment section."""
    if "experiment" not in config:
        raise ConfigValidationError("Missing 'experiment' section")

    exp = config["experiment"]

    if "base_random_seed" not in exp:
        raise ConfigValidationError("Missing 'experiment.base_random_seed'")

    seed = exp["base_random_seed"]
    if not isinstance(seed, int) or seed < 0:
        raise ConfigValidationError(
            f"'experiment.base_random_seed' must be a non-negative integer, got {seed}"
        )


def _validate_methods_section(config: dict[str, Any]) -> None:
    """Validate methods section."""
    if "methods" not in config:
        raise ConfigValidationError("Missing 'methods' section")

    methods = config["methods"]

    if not isinstance(methods, list):
        raise ConfigValidationError("'methods' must be a list")

    if len(methods) == 0:
        raise ConfigValidationError("'methods' list cannot be empty")

    invalid_methods = set(methods) - VALID_METHODS
    if invalid_methods:
        raise ConfigValidationError(
            f"Invalid methods: {invalid_methods}. Valid methods: {VALID_METHODS}"
        )


def _validate_clustering_section(config: dict[str, Any]) -> None:
    """Validate clustering configuration."""
    if "clustering" not in config:
        raise ConfigValidationError("Missing 'clustering' section")

    clustering = config["clustering"]

    # Validate algorithm
    algo = clustering.get("algorithm")
    if algo not in VALID_CLUSTERING_ALGORITHMS:
        raise ConfigValidationError(
            f"Invalid clustering.algorithm: {algo}. Valid: {VALID_CLUSTERING_ALGORITHMS}"
        )

    # Validate strategy
    strategy = clustering.get("strategy")
    if strategy not in VALID_CLUSTERING_STRATEGIES:
        raise ConfigValidationError(
            f"Invalid clustering.strategy: {strategy}. Valid: {VALID_CLUSTERING_STRATEGIES}"
        )

    # Validate normalization
    norm = clustering.get("normalization")
    if norm not in VALID_CLUSTERING_NORMALIZATIONS:
        raise ConfigValidationError(
            f"Invalid clustering.normalization: {norm}. Valid: {VALID_CLUSTERING_NORMALIZATIONS}"
        )


def _validate_generator_section(config: dict[str, Any]) -> None:
    """Validate generator section."""
    if "generator" not in config:
        raise ConfigValidationError("Missing 'generator' section")

    gen = config["generator"]

    # Check for working_point if any parameters have role='wp'
    params = config.get("parameters", {})
    has_wp_params = any(p.get("role") == "wp" for p in params.values())

    if has_wp_params and "working_point" not in gen:
        raise ConfigValidationError(
            "Generator must have 'working_point' when parameters have role='wp'"
        )


def _validate_parameters_section(config: dict[str, Any]) -> None:
    """Validate parameters section."""
    if "parameters" not in config:
        raise ConfigValidationError("Missing 'parameters' section")

    params = config["parameters"]

    if not isinstance(params, dict):
        raise ConfigValidationError("'parameters' must be a dictionary")

    for name, spec in params.items():
        _validate_parameter_spec(name, spec)


def _validate_parameter_spec(name: str, spec: dict[str, Any]) -> None:
    """Validate a single parameter specification."""
    if "type" not in spec:
        raise ConfigValidationError(f"Parameter '{name}' missing 'type'")

    ptype = spec["type"]
    if ptype not in VALID_PARAMETER_TYPES:
        raise ConfigValidationError(
            f"Parameter '{name}' has invalid type '{ptype}'. Valid: {VALID_PARAMETER_TYPES}"
        )

    # Validate role if present
    if "role" in spec and spec["role"] not in VALID_ROLES:
        raise ConfigValidationError(
            f"Parameter '{name}' has invalid role '{spec['role']}'. Valid: {VALID_ROLES}"
        )

    # Type-specific validation
    if ptype == "range":
        _validate_range_spec(name, spec)
    elif ptype == "list":
        _validate_list_spec(name, spec)
    elif ptype == "const":
        _validate_const_spec(name, spec)


def _validate_range_spec(name: str, spec: dict[str, Any]) -> None:
    """Validate range-type parameter."""
    required = ["start", "end", "num_steps"]
    for field in required:
        if field not in spec:
            raise ConfigValidationError(f"Parameter '{name}' missing '{field}'")

    start, end, steps = spec["start"], spec["end"], spec["num_steps"]

    if not isinstance(steps, int) or steps < 1:
        raise ConfigValidationError(
            f"Parameter '{name}': num_steps must be positive integer, got {steps}"
        )

    if start >= end:
        raise ConfigValidationError(
            f"Parameter '{name}': start ({start}) must be < end ({end})"
        )

    # Validate scale if present
    if "scale" in spec and spec["scale"] not in VALID_SCALES:
        raise ConfigValidationError(
            f"Parameter '{name}': invalid scale '{spec['scale']}'. Valid: {VALID_SCALES}"
        )

    # If log scale, values must be positive
    if spec.get("scale") == "log" and (start <= 0 or end <= 0):
        raise ConfigValidationError(
            f"Parameter '{name}': log scale requires positive start and end"
        )


def _validate_list_spec(name: str, spec: dict[str, Any]) -> None:
    """Validate list-type parameter."""
    if "values" not in spec:
        raise ConfigValidationError(f"Parameter '{name}' missing 'values'")

    values = spec["values"]
    if not isinstance(values, list) or len(values) == 0:
        raise ConfigValidationError(
            f"Parameter '{name}': 'values' must be a non-empty list"
        )

    # Validate noise_mode values if applicable
    if name == "noise_mode":
        invalid = set(values) - VALID_NOISE_MODES
        if invalid:
            raise ConfigValidationError(
                f"Parameter 'noise_mode' has invalid values: {invalid}. Valid: {VALID_NOISE_MODES}"
            )


def _validate_const_spec(name: str, spec: dict[str, Any]) -> None:
    """Validate const-type parameter."""
    if "value" not in spec:
        raise ConfigValidationError(f"Parameter '{name}' missing 'value'")


def _validate_static_args_section(config: dict[str, Any]) -> None:
    """Validate static_args section."""
    if "static_args" not in config:
        raise ConfigValidationError("Missing 'static_args' section")

    static = config["static_args"]

    # Validate required static args
    if "n_iter" not in static:
        raise ConfigValidationError("Missing 'static_args.n_iter'")

    n_iter = static["n_iter"]
    if not isinstance(n_iter, int) or n_iter < 1:
        raise ConfigValidationError(
            f"'static_args.n_iter' must be positive integer, got {n_iter}"
        )

    if "dt" not in static:
        raise ConfigValidationError("Missing 'static_args.dt'")

    dt = static["dt"]
    if not isinstance(dt, (int, float)) or dt <= 0:
        raise ConfigValidationError(
            f"'static_args.dt' must be positive number, got {dt}"
        )

    # Validate rho_mode if present
    if "rho_mode" in static:
        rho_mode = static["rho_mode"]
        if rho_mode not in VALID_RHO_MODES:
            raise ConfigValidationError(
                f"Invalid 'static_args.rho_mode': {rho_mode}. Valid: {VALID_RHO_MODES}"
            )


def _validate_pbs_section(config: dict[str, Any]) -> None:
    """Validate PBS configuration."""
    if "pbs" not in config:
        raise ConfigValidationError("Missing 'pbs' section")

    pbs = config["pbs"]
    required_fields = ["queue", "mem", "ncpus", "mpiprocs", "walltime"]

    for field in required_fields:
        if field not in pbs:
            raise ConfigValidationError(f"Missing 'pbs.{field}'")

    # Validate numeric fields
    for field in ["ncpus", "mpiprocs"]:
        val = pbs[field]
        if not isinstance(val, int) or val < 1:
            raise ConfigValidationError(
                f"'pbs.{field}' must be positive integer, got {val}"
            )


def _validate_paths_section(config: dict[str, Any]) -> None:
    """Validate paths section."""
    if "paths" not in config:
        raise ConfigValidationError("Missing 'paths' section")

    paths = config["paths"]
    required = ["interpreter", "code_dir", "worker_script"]

    for field in required:
        if field not in paths:
            raise ConfigValidationError(f"Missing 'paths.{field}'")

    # Check interpreter exists (if local path)
    # Check interpreter exists (if local path)
    interp = paths["interpreter"]
    if not (interp.startswith("/") or (len(interp) > 1 and interp[1] == ":")):
        raise ConfigValidationError(
            f"'paths.interpreter' must be absolute path, got {interp}"
        )

    # Check code_dir exists (if local path)
    code_dir = paths["code_dir"]
    if not (code_dir.startswith("/") or (len(code_dir) > 1 and code_dir[1] == ":")):
        raise ConfigValidationError(
            f"'paths.code_dir' must be absolute path, got {code_dir}"
        )


def _validate_output_section(config: dict[str, Any]) -> None:
    """Validate output section."""
    if "output" not in config:
        raise ConfigValidationError("Missing 'output' section")

    output = config["output"]

    if "output_dir" not in output:
        raise ConfigValidationError("Missing 'output.output_dir'")


def _validate_parameter_compatibility(config: dict[str, Any]) -> None:
    """
    Validate that parameter combinations are compatible.
    Specifically: N * (1 - tau) >= M for all ACTUAL configurations.
    """
    # Generate all actual job parameters to check compatibility
    # This avoids false positives from checking worst-case combinations that never occur
    try:
        jobs = generate_job_parameters(config)
    except Exception as e:
        raise ConfigValidationError(
            f"Failed to generate parameters for validation: {e}"
        )

    for i, job in enumerate(jobs):
        # Extract parameters for this specific job
        # Fallback to working point or defaults if not in job dict (should be there though)
        N = job.get("temporal_dim")
        tau = job.get("delays_over_timesteps")
        M = job.get("max_rank")
        D = job.get("spatial_dim")

        # Skip if any required param is missing (should be caught by other validators)
        if N is None or tau is None or M is None or D is None:
            continue

        # Check dimension limit
        L = int(tau * N)
        MAX_LD = 15_000
        if L * D > MAX_LD:
            raise ConfigValidationError(
                f"Job {i}: num_delays * spatial_dim would exceed {MAX_LD}. "
                f"Parameters: temporal_dim={N}, delays={tau:.3f}, spatial_dim={D}. "
                f"Result: {L} * {D} = {L*D}"
            )

        # Check rank compatibility: N * (1 - tau) >= M
        min_available = N * (1 - tau)
        if min_available < M:
            raise ConfigValidationError(
                f"Job {i}: Parameter compatibility check failed. "
                f"temporal_dim * (1 - delays_over_timesteps) must be >= max_rank. "
                f"Job parameters: N={N}, delays={tau:.3f}, max_rank={M}. "
                f"Check: {N} * (1 - {tau:.3f}) = {min_available:.1f} < {M}"
            )


def validate_config_file(config_path: str) -> dict[str, Any]:
    """
    Load and validate a configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Validated configuration dictionary

    Raises:
        ConfigValidationError: If validation fails
    """
    import yaml

    if not os.path.exists(config_path):
        raise ConfigValidationError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    validate_config(config)

    return config


def main(config_path: str):
    """
    Validate a configuration file.

    Args:
        config_path: Path to YAML config file
    """
    try:
        validate_config_file(config_path)
        print("✓ Configuration is valid")
    except ConfigValidationError as e:
        print(f"✗ Configuration validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
