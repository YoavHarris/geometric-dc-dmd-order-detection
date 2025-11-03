#!/usr/bin/env python3
"""
Driver CLI for PBS‑based experiments:
  - make_jobs
  - submit
  - resubmit
  - combine_results
  - status

Usage (via Fire):
  python experiment_on_pbs.py make_jobs --config=config.yaml
  python experiment_on_pbs.py submit   --config=config.yaml [--ids="0,5,10"]
  python experiment_on_pbs.py resubmit --config=config.yaml
  python experiment_on_pbs.py combine_results --config=config.yaml [--incremental]
  python experiment_on_pbs.py status   --config=config.yaml
"""

import os
import shutil
import stat
import fire
from pathlib import Path
from typing import Union, Iterable, List

from experiment_with_pbs.param_generator import (
    generate_job_args,
    write_job_list,
)
from experiment_with_pbs.state_manager import (
    save_state,
    get_combined,
    mark_combined,
    DEFAULT_STATE,
    get_succeeded,
    mark_succeeded,
    mark_failed,
    get_failed,
)
from experiment_with_pbs.pbs_utils import (
    build_pbs_script,
    write_script,
    submit_job,
    load_yaml,
    fetch_running_job_ids,
)
from experiment_with_pbs.result_combiner import combine


def load_experiment_config(config_path: str) -> dict:
    """
    Load the user config.yaml. If a frozen snapshot exists under output_dir, load that instead.
    """
    cfg = load_yaml(config_path)
    outdir = cfg["output"]["output_dir"]
    frozen = os.path.join(outdir, "config.frozen.yaml")
    if os.path.exists(frozen):
        cfg = load_yaml(frozen)
    return cfg


def _load_ctx(config: str):
    """
    Common helper: load config, determine paths, and read the static job list.
    Returns: (cfg, outdir, state_path, job_list)
    """
    cfg = load_experiment_config(config)
    outdir = cfg["output"]["output_dir"]
    state_path = os.path.join(outdir, "jobs_state.json")

    jl_path = os.path.join(outdir, "job_list.txt")
    with open(jl_path, "r") as f:
        job_list = [line.strip() for line in f]

    return cfg, outdir, state_path, job_list


def _normalize_ids(ids: Union[str, int, Iterable[Union[int, str]]]) -> List[int]:
    """
    Turn the flexible --ids value (str "0,5", int, list/tuple) into a flat list[int].
    """
    if ids is None:
        return []
    if isinstance(ids, int):
        return [ids]

    raw: List[Union[int, str]]
    if isinstance(ids, str):
        raw = ids.replace(",", " ").split()
    else:  # list or tuple
        raw = []
        for item in ids:
            raw.extend(str(item).replace(",", " ").split())

    return [int(x) for x in raw]


def make_jobs(*, config: str) -> None:
    """
    Generate and freeze everything needed to drive your sweep:
      1. Snapshot config.yaml → config.frozen.yaml (read-only, UTF-8)
      2. Build job_list.txt according to generator.mode
      3. Initialize an empty jobs_state.json
    """
    # 1) Load original config
    orig = load_yaml(config)

    # 2) Ensure output directory exists
    outdir = Path(orig["output"]["output_dir"])
    os.makedirs(outdir, exist_ok=True)

    # 3) Snapshot the config (immutable)
    snap = os.path.join(outdir, "config.frozen.yaml")
    shutil.copy(config, snap)
    os.chmod(snap, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)

    # 4) Extract sweep definitions and static args
    parameters = orig["parameters"]
    static_args = orig.get("static_args", {})

    # 5) Parse generator settings (with defaults)
    gen_cfg = orig.get("generator", {})
    working_point = gen_cfg.get("working_point", None)

    # 6) Generate the list of CLI-argument strings
    job_args = generate_job_args(
        parameters,
        static_args,
        sigfigs=3,
        working_point=working_point,
    )

    # 7) Write the frozen job_list.txt (read-only, UTF-8)
    job_list_path = write_job_list(job_args, outdir)

    # 8) Initialize an empty state file for tracking
    state_path = os.path.join(outdir, "jobs_state.json")
    save_state(DEFAULT_STATE.copy(), state_path)

    # 9) Report
    print(f"- make_jobs:")
    print(f"   - Wrote {len(job_args)} jobs to {job_list_path}")


def submit(
    *,
    config: str,
    ids: Union[str, int, Iterable[Union[int, str]]] = None,
) -> None:
    """
    Queue jobs that have not yet *succeeded* and are not currently *running*.
    Optional --ids restricts to a subset.
    """
    # ─── context ────────────────────────────────────────────────────────────
    cfg, outdir, state_path, job_list = _load_ctx(config)
    total = len(job_list)

    succeeded = set(get_succeeded(state_path))
    running = fetch_running_job_ids()

    # ─── compute the schedulable set ────────────────────────────────────────
    base_target = set(range(total)) - succeeded - running
    requested = set(_normalize_ids(ids)) if ids is not None else base_target
    final_ids = sorted(base_target & requested)

    if not final_ids:
        print("- submit: nothing to schedule")
        return

    print(f"- submit: queuing {len(final_ids)}/{total} jobs")

    # ─── loop and submit ────────────────────────────────────────────────────
    for job_id in final_ids:
        cli = job_list[job_id]

        job_out = os.path.join(outdir, "job_outputs", f"job{job_id}")
        os.makedirs(job_out, exist_ok=True)
        output_csv = os.path.join(job_out, f"job{job_id}_results.csv")
        full_cli = f'{cli} --output_path="{output_csv}"'

        script_txt = build_pbs_script(
            cfg["pbs"],
            cfg["paths"]["interpreter"],
            cfg["paths"]["code_dir"],
            cfg["paths"]["exp_script"] + " run",
            full_cli,
            job_id,
            outdir,
        )
        script_path = os.path.join(outdir, f"job_{job_id}.sh")
        write_script(script_txt, script_path)

        out, err = submit_job(script_path)
        if out:
            os.remove(script_path)
            print(f"  · job {job_id} queued  → {out}")
        else:
            print(f"  · job {job_id} FAILED → {err}")


def resubmit(*, config: str) -> None:
    """
    Re-queue every job currently marked as *failed* in jobs_state.json.
    The up-to-date failed list is produced by `status()`.
    """
    _, outdir, state_path, job_list = _load_ctx(config)
    failed_ids = get_failed(state_path)

    if not failed_ids:
        print("→ resubmit: nothing to re-submit (no failed jobs)")
        return

    print(f"→ resubmit: re-queuing {len(failed_ids)} failed jobs: {failed_ids[:10]}…")

    submit(config=config, ids=tuple(failed_ids))


def combine_results(*, config: str, incremental: bool = False) -> None:
    """
    Combine finished results into one CSV.
    If incremental, only process newly finished jobs.
    """
    cfg, outdir, state_path, job_list = _load_ctx(config)
    total = len(job_list)

    finished = {
        i
        for i in range(total)
        if os.path.exists(
            os.path.join(outdir, "job_outputs", f"job{i}", f"job{i}_results.csv")
        )
    }

    if incremental:
        already = set(get_combined(state_path))
        to_combine = sorted(finished - already)
    else:
        to_combine = sorted(finished)

    if not to_combine:
        print("→ combine_results: nothing to do")
        return

    combine(outdir, incremental=incremental)
    mark_combined(to_combine, state_path)
    print(f"→ combine_results: integrated {len(to_combine)} jobs")


def status(*, config: str) -> None:
    """
    Show the current experiment state and update
        • succeeded   (new CSVs on disk)
        • running     (jobs in the PBS queue, transient)
        • failed      (submitted → dir exists, but neither running nor succeeded)
    State file now contains only {succeeded, failed, combined}.
    """
    _, outdir, state_path, job_list = _load_ctx(config)
    total = len(job_list)

    # ── 1) Scan disk for finished CSVs ───────────────────────────────────────
    succeeded_on_disk = {
        i
        for i in range(total)
        if os.path.exists(
            os.path.join(outdir, "job_outputs", f"job{i}", f"job{i}_results.csv")
        )
    }
    mark_succeeded(succeeded_on_disk, state_path)
    succeeded: set[int] = set(get_succeeded(state_path))

    # ── 2) Query PBS for jobs currently running ──────────────────────────────
    running = fetch_running_job_ids()

    # ── 3) Discover which jobs have been *attempted* (dir exists) ────────────
    attempted: set[int] = set()
    jobs_root = os.path.join(outdir, "job_outputs")
    if os.path.isdir(jobs_root):
        for name in os.listdir(jobs_root):
            if name.startswith("job") and name[3:].isdigit():
                attempted.add(int(name[3:]))

    # ── 4) Infer failures: attempted − succeeded − running ───────────────────
    prev_failed: set[int] = set(get_failed(state_path))
    new_failed = (attempted - succeeded) - running
    # Remove any IDs that were previously marked failed but are now running or succeeded
    cleaned_failed = (prev_failed | new_failed) - succeeded - running
    mark_failed(cleaned_failed, state_path)
    failed: set[int] = set(get_failed(state_path))

    combined = set(get_combined(state_path))

    # ── 5) Print summary ─────────────────────────────────────────────────────
    print(f"Total jobs:   {total}")
    print(f"Running:      {len(running)}")
    print(f"Succeeded:    {len(succeeded)}")
    print(f"Failed:       {len(failed)}")
    print(f"Combined:     {len(combined)}")


if __name__ == "__main__":
    fire.Fire(
        {
            "make_jobs": make_jobs,
            "submit": submit,
            "resubmit": resubmit,
            "combine_results": combine_results,
            "status": status,
        }
    )
