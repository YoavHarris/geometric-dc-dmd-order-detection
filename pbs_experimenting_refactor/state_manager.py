#!/usr/bin/env python3
"""
Job state management for PBS experiments.
Tracks succeeded, failed, and combined jobs.
"""

import json
import os
from typing import Iterable, List, Set, Dict, Any
from pathlib import Path


DEFAULT_STATE: Dict[str, Any] = {
    "succeeded": [],
    "failed": [],
    "combined": [],
}


def load_state(state_path: str) -> Dict[str, Any]:
    """Load state from JSON file."""
    if not os.path.exists(state_path):
        return DEFAULT_STATE.copy()

    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: Dict[str, Any], state_path: str) -> None:
    """Save state to JSON file."""
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ────────────────────────────── SUCCEEDED ──────────────────────────────────


def mark_succeeded(ids: Iterable[int], state_path: str) -> None:
    """Mark jobs as succeeded."""
    st = load_state(state_path)
    st["succeeded"] = sorted(set(st.get("succeeded", [])) | set(ids))
    save_state(st, state_path)


def get_succeeded(state_path: str) -> List[int]:
    """Get list of succeeded job IDs."""
    return load_state(state_path).get("succeeded", [])


# ─────────────────────────────── FAILED ────────────────────────────────────


def mark_failed(ids: Iterable[int], state_path: str) -> None:
    """Mark jobs as failed (replaces existing failed list)."""
    st = load_state(state_path)
    st["failed"] = sorted(set(ids))
    save_state(st, state_path)


def get_failed(state_path: str) -> List[int]:
    """Get list of failed job IDs."""
    return load_state(state_path).get("failed", [])


# ───────────────────────────── COMBINED ─────────────────────────────────────


def mark_combined(ids: Iterable[int], state_path: str) -> None:
    """Mark jobs as combined into results."""
    st = load_state(state_path)
    st["combined"] = sorted(set(st.get("combined", [])) | set(ids))
    save_state(st, state_path)


def get_combined(state_path: str) -> List[int]:
    """Get list of combined job IDs."""
    return load_state(state_path).get("combined", [])


# ────────────────────────────── UTILITIES ───────────────────────────────────


def find_finished_jobs(output_dir: str, total_jobs: int) -> Set[int]:
    """
    Find which jobs have finished by checking for results.csv files.

    Args:
        output_dir: Output directory path
        total_jobs: Total number of jobs

    Returns:
        Set of finished job IDs
    """
    finished = set()
    job_outputs_dir = Path(output_dir) / "job_outputs"

    if not job_outputs_dir.exists():
        return finished

    for job_id in range(total_jobs):
        results_file = job_outputs_dir / f"job{job_id}" / "results.csv"
        if results_file.exists():
            finished.add(job_id)

    return finished


def find_attempted_jobs(output_dir: str) -> Set[int]:
    """
    Find which jobs have been attempted (output directory exists).

    Args:
        output_dir: Output directory path

    Returns:
        Set of attempted job IDs
    """
    attempted = set()
    job_outputs_dir = Path(output_dir) / "job_outputs"

    if not job_outputs_dir.exists():
        return attempted

    for item in job_outputs_dir.iterdir():
        if item.is_dir() and item.name.startswith("job"):
            try:
                job_id = int(item.name[3:])  # Extract from "job0042"
                attempted.add(job_id)
            except ValueError:
                pass

    return attempted


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python state_manager.py <output_dir>")
        sys.exit(1)

    output_dir = sys.argv[1]
    state_path = os.path.join(output_dir, "jobs_state.json")

    # Display current state
    state = load_state(state_path)

    print("Current State:")
    print(f"  Succeeded: {len(state.get('succeeded', []))}")
    print(f"  Failed: {len(state.get('failed', []))}")
    print(f"  Combined: {len(state.get('combined', []))}")
