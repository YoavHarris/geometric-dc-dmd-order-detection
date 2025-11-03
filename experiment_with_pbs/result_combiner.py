#!/usr/bin/env python3
"""
In-process merger of individual job CSV results into a single combined table,
with optional incremental updating.
"""
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Fields that define a unique run (must match combine_results.py original)
UNIQUE_FIELDS = [
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
    "method",
    "noise_mode",
    "artificial_damping",
]


def _load_job_list(base_dir: Path) -> int:
    """Return total number of jobs from job_list.txt."""
    jl = base_dir / "job_list.txt"
    if not jl.exists():
        raise FileNotFoundError(f"Job list not found at {jl}")
    lines = jl.read_text().splitlines()
    return len(lines)


def _find_finished_ids(base_dir: Path) -> set:
    """Detect which job IDs have finished by existence of their CSV."""
    total = _load_job_list(base_dir)
    job_out = base_dir / "job_outputs"
    finished = set()
    for i in range(total):
        csv_path = job_out / f"job{i}" / f"job{i}_results.csv"
        if csv_path.exists():
            finished.add(i)
    return finished


def _load_job_df(base_dir: Path, job_id: int) -> pd.DataFrame:
    """Load a single job's result CSV and tag with its job ID."""
    csv_path = base_dir / "job_outputs" / f"job{job_id}" / f"job{job_id}_results.csv"
    df = pd.read_csv(csv_path)
    df["id"] = job_id
    return df


def _validate_and_deduplicate(df: pd.DataFrame):
    """Drop exact duplicates and detect conflicting groups."""
    valid = []
    conflicts = []
    valid_fields = [f for f in UNIQUE_FIELDS if not df[f].isna().all()]
    grouped = df.groupby(valid_fields)
    for key, group in grouped:
        if len(group) == 1:
            valid.append(group)
        else:
            dedup = group.drop_duplicates()
            if len(dedup) == 1:
                valid.append(dedup)
            else:
                diff_cols = [
                    c
                    for c in df.columns
                    if c not in UNIQUE_FIELDS + ["id"] and group[c].nunique() > 1
                ]
                conflicts.append(
                    {
                        "group": key,
                        "job_ids": group["id"].tolist(),
                        "diff_fields": diff_cols,
                    }
                )
    if valid:
        valid_df = pd.concat(valid, ignore_index=True)
    else:
        valid_df = pd.DataFrame()
    return valid_df, conflicts


def combine(
    base_dir: str,
    combined_filename: str = "combined_results.csv",
    conflict_filename: str = "merge_conflicts.csv",
    incremental: bool = False,
):
    """
    Combine job result CSVs under <base_dir>/job_outputs.

    If incremental and an existing combined file is present, only new jobs are appended.

    Outputs:
      - combined_results.csv
      - merge_conflicts.csv (if any conflicts)
    """
    base = Path(base_dir)
    # detect finished jobs
    finished = _find_finished_ids(base)

    combined_path = base / combined_filename
    if incremental and combined_path.exists():
        # load existing combined data
        existing = pd.read_csv(combined_path)
        existing_ids = (
            set(existing["id"].unique()) if "id" in existing.columns else set()
        )
        new_ids = sorted(finished - existing_ids)
        if not new_ids:
            print("No new jobs to combine.")
            return
        # load only new jobs
        new_dfs = [_load_job_df(base, jid) for jid in new_ids]
        full_df = pd.concat([existing] + new_dfs, ignore_index=True)
    else:
        # full combine from scratch
        new_ids = sorted(finished)
        if not new_ids:
            print("No finished jobs found.")
            return
        full_df = pd.concat(
            [_load_job_df(base, jid) for jid in tqdm(new_ids, desc="CSV Files")],
            ignore_index=True,
        )

    # validate and deduplicate
    valid_df, conflicts = _validate_and_deduplicate(full_df)

    # save combined results
    valid_df.to_csv(combined_path, index=False)
    print(f"Combined {len(valid_df)} rows into {combined_filename}.")

    # save conflicts if present
    if conflicts:
        rows = []
        for c in conflicts:
            raw = c["group"]
            if not isinstance(raw, tuple):
                # if somehow it were a single value, wrap it
                raw = (raw,)
            row = dict(zip(UNIQUE_FIELDS, raw))
            row["job_ids"] = ",".join(map(str, c["job_ids"]))
            row["conflict_fields"] = ",".join(c["diff_fields"])
            rows.append(row)
        pd.DataFrame(rows).to_csv(base / conflict_filename, index=False)
        print(
            f"Found {len(conflicts)} conflicting groups; details in {conflict_filename}."
        )


if __name__ == "__main__":
    import fire

    fire.Fire({"combine": combine})
