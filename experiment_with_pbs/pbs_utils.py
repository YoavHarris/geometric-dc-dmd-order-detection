import getpass
import os
import subprocess
import logging
import time
import yaml

from typing import Tuple, Optional, Dict, Any, Set


def build_pbs_script(
    pbs_cfg: Dict[str, Any],
    interpreter: str,
    code_dir: str,
    exp_script: str,
    job_args: str,
    job_id: int,
    output_dir: str,
) -> str:
    """
    Build the content of a PBS submission script for one job.
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
        f"#PBS -N job_{job_id}",
        f"#PBS -o {os.path.join(logs_dir, f'job_{job_id}.out')}",
        f"#PBS -e {os.path.join(logs_dir, f'job_{job_id}.err')}",
        "",
        f"cd {code_dir}",
        f"{interpreter} {exp_script} {job_args}",
    ]
    return "\n".join(lines)


def write_script(content: str, script_path: str) -> None:
    """
    Write the PBS script content to the given filesystem path.
    """
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    with open(script_path, "w") as f:
        f.write(content)


def submit_job(
    script_path: str, retries: int = 5, retry_delay: int = 10
) -> Tuple[Optional[str], Optional[str]]:
    """
    Submit the PBS script via qsub, retrying on failure.
    Returns a tuple (stdout, stderr). On success, stderr is None.
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

    # after retries exhausted
    err = proc.stderr.strip() if proc.stderr else "Unknown error"
    return None, err


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_running_job_ids() -> Set[int]:
    """
    Return {job_id, …} for all jobs currently in the PBS queue
    that follow the naming convention Jobname='job_<id>'.
    """
    running: set[int] = set()
    proc = subprocess.run(
        ["qstat", "-u", getpass.getuser()],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return running

    for line in proc.stdout.splitlines()[2:]:  # skip the 2-line header
        parts = line.split()
        if len(parts) >= 4 and parts[3].startswith("job_"):
            try:
                running.add(int(parts[3].split("_", 1)[1]))
            except ValueError:
                pass

    return running
