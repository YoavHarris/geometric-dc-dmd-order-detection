#!/usr/bin/env python3
"""
Result combiner for merging individual job CSVs into combined results.
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm


# Fields that uniquely identify a run configuration
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
    "n_independent_modes",
    "top_amplitude",
    "max_rank",
    "method",
    "noise_mode",
    "artificial_damping",
]


def load_job_result(output_dir: Path, job_id: int) -> pd.DataFrame:
    """
    Load results CSV for a single job.

    Args:
        output_dir: Output directory path
        job_id: Job ID

    Returns:
        DataFrame with job results
    """
    csv_path = output_dir / "job_outputs" / f"job{job_id}" / "results.csv"
    df = pd.read_csv(csv_path)
    df["job_id"] = job_id
    return df


def validate_and_deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Validate combined results and detect conflicts.

    Args:
        df: Combined DataFrame

    Returns:
        Tuple of (valid_df, conflicts)
            valid_df: Deduplicated DataFrame
            conflicts: List of conflict information dicts
    """
    valid = []
    conflicts = []

    # Only group by fields that exist and aren't all NaN
    valid_fields = [
        f for f in UNIQUE_FIELDS if f in df.columns and not df[f].isna().all()
    ]

    grouped = df.groupby(valid_fields, dropna=False)

    for key, group in grouped:
        if len(group) == 1:
            valid.append(group)
        else:
            # Multiple rows with same parameters
            dedup = group.drop_duplicates()
            if len(dedup) == 1:
                # Exact duplicates, keep one
                valid.append(dedup)
            else:
                # Real conflicts - different results for same parameters
                diff_cols = [
                    c
                    for c in df.columns
                    if c not in valid_fields + ["job_id"] and group[c].nunique() > 1
                ]
                conflicts.append(
                    {
                        "group": key,
                        "job_ids": group["job_id"].tolist(),
                        "diff_fields": diff_cols,
                    }
                )

    if valid:
        valid_df = pd.concat(valid, ignore_index=True)
    else:
        valid_df = pd.DataFrame()

    return valid_df, conflicts


def combine_results(
    output_dir: str,
    job_ids: list[int] | None = None,
    incremental: bool = False,
    combined_filename: str = "combined_results.csv",
    conflict_filename: str = "merge_conflicts.csv",
) -> None:
    """
    Combine job result CSVs into a single file.

    Args:
        output_dir: Output directory path
        job_ids: List of job IDs to combine (None = all finished jobs)
        incremental: If True, only combine new jobs (not already in combined)
        combined_filename: Name for combined results file
        conflict_filename: Name for conflicts file
    """
    base_dir = Path(output_dir)
    combined_path = base_dir / combined_filename

    # Determine which jobs to combine
    if job_ids is None:
        # Find all finished jobs
        job_outputs = base_dir / "job_outputs"
        if not job_outputs.exists():
            print("No job outputs found.")
            return

        job_ids = []
        for item in sorted(job_outputs.iterdir()):
            if item.is_dir() and item.name.startswith("job"):
                try:
                    job_id = int(item.name[3:])
                    results_file = item / "results.csv"
                    if results_file.exists():
                        job_ids.append(job_id)
                except ValueError:
                    pass

    if not job_ids:
        print("No finished jobs to combine.")
        return

    # Handle incremental mode
    if incremental and combined_path.exists():
        existing = pd.read_csv(combined_path)
        existing_ids = (
            set(existing["job_id"].unique()) if "job_id" in existing.columns else set()
        )
        new_ids = [jid for jid in job_ids if jid not in existing_ids]

        if not new_ids:
            print("No new jobs to combine.")
            return

        print(f"Incremental mode: combining {len(new_ids)} new jobs")
        new_dfs = [
            load_job_result(base_dir, jid)
            for jid in tqdm(new_ids, desc="Loading new results")
        ]
        full_df = pd.concat([existing] + new_dfs, ignore_index=True)
    else:
        # Full combine from scratch
        print(f"Combining {len(job_ids)} jobs")
        job_dfs = [
            load_job_result(base_dir, jid)
            for jid in tqdm(job_ids, desc="Loading results")
        ]
        full_df = pd.concat(job_dfs, ignore_index=True)

    # Validate and deduplicate
    valid_df, conflicts = validate_and_deduplicate(full_df)

    # Save combined results
    valid_df.to_csv(combined_path, index=False)
    print(f"\n✓ Combined {len(valid_df)} rows into {combined_filename}")
    print(f"  Output: {combined_path}")

    # Save conflicts if present
    if conflicts:
        rows = []
        for c in conflicts:
            raw = c["group"]
            if not isinstance(raw, tuple):
                raw = (raw,)

            # Build row dict
            row = dict(zip(UNIQUE_FIELDS, raw))
            row["job_ids"] = ",".join(map(str, c["job_ids"]))
            row["conflict_fields"] = ",".join(c["diff_fields"])
            rows.append(row)

        conflict_path = base_dir / conflict_filename
        pd.DataFrame(rows).to_csv(conflict_path, index=False)
        print(f"\n⚠ Found {len(conflicts)} conflicting groups")
        print(f"  Details in: {conflict_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python result_combiner.py <output_dir> [--incremental]")
        sys.exit(1)

    output_dir = sys.argv[1]
    incremental = "--incremental" in sys.argv

    combine_results(output_dir, incremental=incremental)
