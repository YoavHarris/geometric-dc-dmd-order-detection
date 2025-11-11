#!/usr/bin/env python3
"""
Main experiment runner CLI for PBS-based experiments.

Commands:
  - make_jobs: Generate job configs and freeze experiment
  - submit: Submit jobs to PBS queue
  - resubmit: Resubmit failed jobs
  - status: Check experiment status
  - combine_results: Merge job results into single CSV
"""

import os
import shutil
import stat
from pathlib import Path
from typing import List, Set
import yaml
import fire

from config_validator import validate_config_file, ConfigValidationError
from param_generator import generate_job_parameters
from job_config_builder import write_job_configs
from pbs_utils import (
    build_pbs_script,
    write_script,
    submit_job,
    fetch_running_job_ids,
)
from state_manager import (
    save_state,
    mark_succeeded,
    mark_failed,
    get_succeeded,
    get_failed,
    get_combined,
    mark_combined,
    find_finished_jobs,
    find_attempted_jobs,
    DEFAULT_STATE,
)
from result_combiner import combine_results


def _load_config(config_path: str) -> dict:
    """Load and validate configuration."""
    frozen_config = _get_frozen_config_path(config_path)

    if os.path.exists(frozen_config):
        print(f"Using frozen config: {frozen_config}")
        with open(frozen_config, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # Validate and load original config
    try:
        config = validate_config_file(config_path)
        return config
    except ConfigValidationError as e:
        print(f"✗ Configuration validation failed: {e}")
        raise


def _get_frozen_config_path(config_path: str) -> str:
    """Get path to frozen config from original config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    output_dir = config["output"]["output_dir"]
    return os.path.join(output_dir, "config.frozen.yaml")


def _normalize_ids(ids_arg) -> List[int]:
    """Parse job IDs from various input formats."""
    if ids_arg is None:
        return []

    if isinstance(ids_arg, int):
        return [ids_arg]

    if isinstance(ids_arg, str):
        # Handle comma-separated or space-separated
        parts = ids_arg.replace(",", " ").split()
        return [int(x) for x in parts]

    if isinstance(ids_arg, (list, tuple)):
        result = []
        for item in ids_arg:
            if isinstance(item, int):
                result.append(item)
            else:
                result.extend(_normalize_ids(str(item)))
        return result

    return []


class ExperimentRunner:
    """Main experiment runner with Fire CLI."""

    def make_jobs(self, config: str) -> None:
        """
        Generate and freeze all experiment configuration.

        Creates:
          - config.frozen.yaml (read-only snapshot)
          - job_configs/job####.yaml (per-job configs, read-only)
          - jobs_state.json (empty state tracker)

        Args:
            config: Path to experiment configuration YAML
        """
        print("=" * 70)
        print("MAKE JOBS")
        print("=" * 70)

        # Load and validate config
        try:
            cfg = validate_config_file(config)
        except ConfigValidationError as e:
            print(f"✗ Configuration validation failed: {e}")
            return

        output_dir = Path(cfg["output"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Freeze main config
        frozen_path = output_dir / "config.frozen.yaml"
        shutil.copy(config, frozen_path)
        frozen_path.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        print(f"✓ Froze config → {frozen_path}")

        # Generate parameter combinations
        param_combos = generate_job_parameters(cfg)
        seen = set()
        for i, d in enumerate(param_combos):
            key = tuple(sorted(d.items()))
            if key in seen:
                raise RuntimeError(
                    f"Duplicate parameter point detected at index {i}: {d}"
                )
            seen.add(key)
        print(f"✓ Generated {len(param_combos)} parameter combinations")

        # Write job configs
        n_jobs = write_job_configs(param_combos, cfg, output_dir)
        print(f"✓ Wrote {n_jobs} job configuration files")

        # Initialize state
        state_path = output_dir / "jobs_state.json"
        save_state(DEFAULT_STATE.copy(), str(state_path))
        print(f"✓ Initialized state tracker → {state_path}")

        print(f"\n{'─' * 70}")
        print(f"Ready to submit {n_jobs} jobs!")
        print(f"Run: python experiment_runner.py submit --config={config}")

    def submit(
        self,
        config: str,
        ids: str | int | None = None,
    ) -> None:
        """
        Submit jobs to PBS queue.

        Skips jobs that have already succeeded or are currently running.

        Args:
            config: Path to experiment configuration YAML
            ids: Optional job IDs to submit (comma/space-separated or list)
                 If None, submits all pending jobs
        """
        print("=" * 70)
        print("SUBMIT JOBS")
        print("=" * 70)

        cfg = _load_config(config)
        output_dir = Path(cfg["output"]["output_dir"])
        state_path = output_dir / "jobs_state.json"

        # Count total jobs
        job_configs_dir = output_dir / "job_configs"
        if not job_configs_dir.exists():
            print("✗ No job configs found. Run make_jobs first.")
            return

        total_jobs = len(list(job_configs_dir.glob("job*.yaml")))
        print(f"Total jobs: {total_jobs}")

        # Determine which jobs can be submitted
        succeeded = set(get_succeeded(str(state_path)))
        running = fetch_running_job_ids()

        print(f"Already succeeded: {len(succeeded)}")
        print(f"Currently running: {len(running)}")

        # Compute schedulable set
        all_jobs = set(range(total_jobs))
        schedulable = all_jobs - succeeded - running

        # Apply ID filter if provided
        requested_ids = set(_normalize_ids(ids)) if ids is not None else schedulable
        final_ids = sorted(schedulable & requested_ids)

        if not final_ids:
            print("\n✓ Nothing to submit (all jobs done or running)")
            return

        print(f"\nSubmitting {len(final_ids)} jobs...")

        # Submit jobs
        submitted_count = 0
        failed_count = 0

        for job_id in final_ids:
            job_config_path = job_configs_dir / f"job{job_id}.yaml"
            job_output_dir = output_dir / "job_outputs" / f"job{job_id}"
            job_output_dir.mkdir(parents=True, exist_ok=True)

            # Build PBS script
            script_content = build_pbs_script(
                pbs_cfg=cfg["pbs"],
                interpreter=cfg["paths"]["interpreter"],
                code_dir=cfg["paths"]["code_dir"],
                worker_script=cfg["paths"]["worker_script"],
                job_config_path=str(job_config_path),
                job_id=job_id,
                output_dir=str(output_dir),
            )

            script_path = output_dir / f"pbs_scripts" / f"job_{job_id}.sh"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            write_script(script_content, str(script_path))

            # Submit to PBS
            stdout, stderr = submit_job(str(script_path))

            if stdout:
                submitted_count += 1
                # Clean up script after successful submission
                script_path.unlink()
                if job_id % 10 == 0 or job_id == final_ids[-1]:
                    print(f"  · Submitted jobs 0-{job_id}")
            else:
                failed_count += 1
                print(f"  ✗ Job {job_id} submission failed: {stderr}")

        print(f"\n{'─' * 70}")
        print(f"✓ Submitted: {submitted_count}")
        if failed_count > 0:
            print(f"✗ Failed: {failed_count}")

    def status(self, config: str) -> None:
        """
        Check experiment status and update state.

        Updates:
          - succeeded: jobs with results.csv on disk
          - running: jobs currently in PBS queue
          - failed: attempted jobs that aren't running or succeeded

        Args:
            config: Path to experiment configuration YAML
        """
        print("=" * 70)
        print("EXPERIMENT STATUS")
        print("=" * 70)

        cfg = _load_config(config)
        output_dir = Path(cfg["output"]["output_dir"])
        state_path = output_dir / "jobs_state.json"

        # Count total jobs
        job_configs_dir = output_dir / "job_configs"
        if not job_configs_dir.exists():
            print("✗ No job configs found. Run make_jobs first.")
            return

        total_jobs = len(list(job_configs_dir.glob("job*.yaml")))

        # Find finished jobs (have results.csv)
        finished = find_finished_jobs(str(output_dir), total_jobs)
        mark_succeeded(finished, str(state_path))
        succeeded = set(get_succeeded(str(state_path)))

        # Query PBS for running jobs
        running = fetch_running_job_ids()

        # Find attempted jobs (output directory exists)
        attempted = find_attempted_jobs(str(output_dir))

        # Infer failures
        prev_failed = set(get_failed(str(state_path)))
        new_failed = (attempted - succeeded) - running
        cleaned_failed = (prev_failed | new_failed) - succeeded - running
        mark_failed(cleaned_failed, str(state_path))
        failed = set(get_failed(str(state_path)))

        # Combined count
        combined = set(get_combined(str(state_path)))

        # Print summary
        print(f"\nTotal jobs:       {total_jobs}")
        print(f"Succeeded:        {len(succeeded)}")
        print(f"Running:          {len(running)}")
        print(f"Failed:           {len(failed)}")
        print(f"Combined:         {len(combined)}")
        print(
            f"Pending:          {total_jobs - len(succeeded) - len(running) - len(failed)}"
        )

        # Progress percentage
        progress = 100.0 * len(succeeded) / total_jobs if total_jobs > 0 else 0
        print(f"\nProgress: {progress:.1f}% complete")

        if failed:
            print(
                f"\nFailed job IDs: {sorted(list(failed))[:20]}"
                + ("..." if len(failed) > 20 else "")
            )

    def resubmit(self, config: str) -> None:
        """
        Resubmit all failed jobs.

        Args:
            config: Path to experiment configuration YAML
        """
        print("=" * 70)
        print("RESUBMIT FAILED JOBS")
        print("=" * 70)

        cfg = _load_config(config)
        output_dir = Path(cfg["output"]["output_dir"])
        state_path = output_dir / "jobs_state.json"

        # Update status first
        self.status(config)

        # Get failed IDs
        failed_ids = get_failed(str(state_path))

        if not failed_ids:
            print("\n✓ No failed jobs to resubmit")
            return

        print(f"\nResubmitting {len(failed_ids)} failed jobs...")

        # Submit them
        ids_str = ",".join(map(str, failed_ids))
        self.submit(config=config, ids=ids_str)

    def combine_results(
        self,
        config: str,
        incremental: bool = False,
    ) -> None:
        """
        Combine job results into single CSV.

        Args:
            config: Path to experiment configuration YAML
            incremental: Only combine newly finished jobs
        """
        print("=" * 70)
        print("COMBINE RESULTS")
        print("=" * 70)

        cfg = _load_config(config)
        output_dir = Path(cfg["output"]["output_dir"])
        state_path = output_dir / "jobs_state.json"

        # Count total jobs
        job_configs_dir = output_dir / "job_configs"
        total_jobs = len(list(job_configs_dir.glob("job*.yaml")))

        # Find finished jobs
        finished = find_finished_jobs(str(output_dir), total_jobs)

        if incremental:
            already_combined = set(get_combined(str(state_path)))
            to_combine = sorted(finished - already_combined)

            if not to_combine:
                print("\n✓ No new jobs to combine")
                return

            print(f"Combining {len(to_combine)} new jobs (incremental mode)")
        else:
            to_combine = sorted(finished)
            print(f"Combining {len(to_combine)} jobs (full mode)")

        # Combine
        combine_results(
            output_dir=str(output_dir),
            job_ids=list(to_combine),
            incremental=incremental,
        )

        # Mark as combined
        mark_combined(to_combine, str(state_path))
        print(f"\n✓ Marked {len(to_combine)} jobs as combined")


def main():
    """Main entry point."""
    fire.Fire(ExperimentRunner)


if __name__ == "__main__":
    main()
