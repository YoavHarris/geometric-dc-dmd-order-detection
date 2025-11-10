#!/usr/bin/env python3
"""
PBS job submission utilities.
"""

import getpass
import os
import subprocess
import logging
import time
from typing import Tuple, Optional, Dict, Any, Set


def build_pbs_script(
    pbs_cfg: Dict[str, Any],
    interpreter: str,
    code_dir: str,
    worker_script: str,
    job_config_path: str,
    job_id: int,
    output_dir: str,
) -> str:
    """
    Build the content of a PBS submission script for one job.
    
    Args:
        pbs_cfg: PBS configuration dictionary
        interpreter: Path to Python interpreter
        code_dir: Path to code directory
        worker_script: Path to worker script (relative to code_dir)
        job_config_path: Path to job YAML config
        job_id: Job ID number
        output_dir: Output directory for logs
        
    Returns:
        PBS script content as string
    """
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    lines = [
        "#!/bin/bash",
        f"#PBS -q {pbs_cfg['queue']}",
        f"#PBS -l mem={pbs_cfg['mem']}",
        f"#PBS -l ncpus={pbs_cfg['ncpus']}",
        f"#PBS -l mpiprocs={pbs_cfg['mpiprocs']}",
        f"#PBS -l walltime={pbs_cfg['walltime']}",
        f"#PBS -N job_{job_id:04d}",
        f"#PBS -o {os.path.join(logs_dir, f'job_{job_id:04d}.out')}",
        f"#PBS -e {os.path.join(logs_dir, f'job_{job_id:04d}.err')}",
        "",
        f"cd {code_dir}",
        f"{interpreter} {worker_script} run {job_config_path}",
    ]
    
    return "\n".join(lines)


def write_script(content: str, script_path: str) -> None:
    """
    Write the PBS script content to the given filesystem path.
    
    Args:
        content: Script content
        script_path: Path to write script to
    """
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    with open(script_path, "w") as f:
        f.write(content)


def submit_job(
    script_path: str, retries: int = 5, retry_delay: int = 10
) -> Tuple[Optional[str], Optional[str]]:
    """
    Submit the PBS script via qsub, retrying on failure.
    
    Args:
        script_path: Path to PBS script
        retries: Number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Tuple of (stdout, stderr). On success, stderr is None.
    """
    for attempt in range(1, retries + 1):
        proc = subprocess.run(["qsub", script_path], capture_output=True, text=True)
        
        if proc.returncode == 0:
            return proc.stdout.strip(), None
        
        logging.warning(
            f"[PBS submit] job script={script_path} failed "
            f"(attempt {attempt}/{retries}): {proc.stderr.strip()}"
        )
        time.sleep(retry_delay)
    
    # After retries exhausted
    err = proc.stderr.strip() if proc.stderr else "Unknown error"
    return None, err


def fetch_running_job_ids() -> Set[int]:
    """
    Return set of job IDs currently in the PBS queue.
    Looks for jobs named 'job_XXXX' where XXXX is the job ID.
    
    Returns:
        Set of running job IDs
    """
    running: set[int] = set()
    
    proc = subprocess.run(
        ["qstat", "-u", getpass.getuser()],
        capture_output=True,
        text=True,
    )
    
    if proc.returncode != 0:
        return running
    
    # Parse qstat output
    # Skip header lines (first 2 lines)
    for line in proc.stdout.splitlines()[2:]:
        parts = line.split()
        if len(parts) >= 4 and parts[3].startswith("job_"):
            try:
                # Extract job ID from job_XXXX format
                job_id_str = parts[3].split("_", 1)[1]
                running.add(int(job_id_str))
            except (ValueError, IndexError):
                pass
    
    return running


def check_pbs_available() -> bool:
    """
    Check if PBS commands are available.
    
    Returns:
        True if qsub/qstat are available
    """
    try:
        subprocess.run(
            ["qstat", "--version"],
            capture_output=True,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


if __name__ == "__main__":
    # Test PBS availability
    if check_pbs_available():
        print("✓ PBS is available")
        
        running = fetch_running_job_ids()
        if running:
            print(f"  Currently running jobs: {sorted(running)}")
        else:
            print("  No running jobs")
    else:
        print("✗ PBS is not available (qstat not found)")

