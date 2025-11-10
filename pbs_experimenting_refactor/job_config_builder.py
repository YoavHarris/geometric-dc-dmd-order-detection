#!/usr/bin/env python3
"""
Job configuration builder.
Creates individual YAML configuration files for each job.
"""

import os
import stat
from pathlib import Path
from typing import Dict, Any, List
import yaml


def build_job_config(
    job_id: int,
    params: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a complete job configuration.
    
    Args:
        job_id: Job index
        params: Parameter values for this job
        config: Main experiment configuration
        
    Returns:
        Complete job configuration dictionary
    """
    base_seed = config["experiment"]["base_random_seed"]
    job_seed = base_seed + job_id * 1000
    
    output_dir = Path(config["output"]["output_dir"])
    output_path = output_dir / "job_outputs" / f"job{job_id:04d}" / "results.csv"
    
    # Build job config
    job_config = {
        # Job metadata
        "job_id": job_id,
        "random_seed": job_seed,
        "base_random_seed": base_seed,
        
        # Output
        "output_path": str(output_path),
        
        # Methods and clustering
        "methods": config["methods"],
        "clustering": config["clustering"],
        
        # Parameter values (from sweep)
        "parameters": params,
        
        # Static arguments
        "static_args": config["static_args"],
    }
    
    return job_config


def write_job_configs(
    param_combinations: List[Dict[str, Any]],
    config: Dict[str, Any],
    output_dir: Path,
) -> int:
    """
    Write all job configuration files.
    
    Args:
        param_combinations: List of parameter dictionaries
        config: Main experiment configuration
        output_dir: Output directory path
        
    Returns:
        Number of job configs written
    """
    configs_dir = output_dir / "job_configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    for job_id, params in enumerate(param_combinations):
        job_config = build_job_config(job_id, params, config)
        
        # Write YAML file
        config_path = configs_dir / f"job{job_id:04d}.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(job_config, f, default_flow_style=False, sort_keys=False)
        
        # Make read-only
        config_path.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    
    return len(param_combinations)


def load_job_config(config_path: str) -> Dict[str, Any]:
    """
    Load a job configuration file.
    
    Args:
        config_path: Path to job config YAML
        
    Returns:
        Job configuration dictionary
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    import sys
    from param_generator import generate_job_parameters
    
    if len(sys.argv) != 2:
        print("Usage: python job_config_builder.py <config.yaml>")
        sys.exit(1)
    
    # Load main config
    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)
    
    # Generate parameters
    param_combos = generate_job_parameters(config)
    
    # Write job configs
    output_dir = Path(config["output"]["output_dir"])
    n_jobs = write_job_configs(param_combos, config, output_dir)
    
    print(f"✓ Wrote {n_jobs} job configuration files to {output_dir / 'job_configs'}")

