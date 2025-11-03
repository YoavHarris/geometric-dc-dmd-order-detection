# experiment_on_pbs

A lightweight, all‑in‑Python system for:
- Defining parameter sweeps
- Submitting individual PBS jobs (one‑by‑one)
- Tracking submission/combination state
- Merging per‑job results into a single CSV

Everything is driven from one config file: `config.yaml`

---

## 1. File Structure

```
<repo_root>/
|
├── README.md               # this file
├── config.yaml             # your experiment definition
├── requirements.txt        # Python dependencies
├── experiment_on_pbs.py    # main Fire-based CLI driver
├── run_on_conf.py          # worker script for a single config
├── param_generator.py      # builds & writes job_list.txt
├── result_combiner.py      # merges CSVs into combined_results.csv
├── state_manager.py        # reads/writes jobs_state.json
├── pbs_utils.py            # builds & submits PBS scripts
```

---

## 2. After `make_jobs` (inside `<output_dir>`)

```
<output_dir>/
|
├── config.frozen.yaml       # read-only snapshot of config.yaml
├── job_list.txt             # read-only list of per-job argument lines
├── jobs_state.json          # tracks "submitted" and "combined" IDs
├── job_outputs/             # subfolder per job, each contains job<ID>_results.csv
├── logs/                    # PBS stdout (.out) and stderr (.err) files
```

---

## 3. config.yaml Schema

```yaml
parameters:
  <name>:
    type:      range | list | const
    # for range:
    start:     <number>
    end:       <number>
    num_steps: <int>
    scale:     lin | log    # optional, default "lin"
    as_int:    true|false   # optional
    # for list:
    values:    [v1, v2, …]
    # for const:
    value:     <single value>

static_args:
  <arg_name>: <value>       # e.g. n_iter, num_modes, dt, min_rank, max_rank, random_seed

pbs:
  queue:     <queue_name>
  mem:       <e.g. 1000MB>
  ncpus:     <int>
  mpiprocs:  <int>
  walltime:  "HH:MM:SS"

paths:
  interpreter: <full path to python>
  code_dir:    <full path to your code repo>
  exp_script:  <relative path inside code_dir to run_on_conf.py>

output:
  output_dir: <full path where jobs_state.json, outputs, logs go>
```

---

## 4. Installation

```bash
pip install -r requirements.txt
```

Requirements include:
- fire
- PyYAML
- numpy
- pandas
- tqdm
- pydmd
- any other packages needed by `run_on_conf.py`

---

## 5. Usage – `experiment_on_pbs.py`

All commands are exposed via Python Fire. From the repo root:

### 1) Generate a frozen job list
```bash
python experiment_on_pbs.py make_jobs --config=config.yaml
```
- Copies config.yaml → output_dir/config.frozen.yaml (read-only)
- Generates output_dir/job_list.txt (read-only)
- Creates an empty output_dir/jobs_state.json

### 2) Submit jobs
```bash
python experiment_on_pbs.py submit --config=config.yaml
# or submit only a subset:
python experiment_on_pbs.py submit --config=config.yaml --ids="0,5,10"
```
- Reads job_list.txt
- Skips IDs already in "submitted"
- For each new ID:
  - Builds and writes a PBS script in `<output_dir>`
  - Calls `qsub` on it
  - Records the ID in `jobs_state.json`

### 3) Resubmit missing jobs
```bash
python experiment_on_pbs.py resubmit --config=config.yaml
```
- Checks which `job_outputs/job<ID>_results.csv` files do NOT exist
- Re-submits exactly those missing IDs

### 4) Combine results
```bash
python experiment_on_pbs.py combine_results --config=config.yaml
# or incremental:
python experiment_on_pbs.py combine_results --config=config.yaml --incremental
```
- Finds finished job IDs by existence check
- In "full" mode merges all finished CSVs
- In incremental mode merges only newly finished CSVs
- Deduplicates according to your `UNIQUE_FIELDS`
- Writes `combined_results.csv` (and `merge_conflicts.csv` if conflicts)

### 5) Check status
```bash
python experiment_on_pbs.py status --config=config.yaml
```
Outputs counts:
- Total jobs
- Submitted
- Finished
- Combined

---

## 6. Direct Debugging with `run_on_conf.py`

To run a single configuration manually:

```bash
python run_on_conf.py run \
  --snr_db=0 \
  --freq_sep=0.01 \
  --eig_mag=0.99 \
  --delays_over_timesteps=0.1 \
  --spatial_dim=20 \
  --temporal_dim=200 \
  --noise_mode="gaussian" \
  --num_modes=2 \
  --top_amplitude_exponent=0.0 \
  --n_iter=1 \
  --min_rank=1 \
  --max_rank=12 \
  --dt=1.0 \
  --output_path=./tmp/job0/job0_results.csv \
  --plot
```

Add `--plot` to generate plots and print the DataFrame head; otherwise it prints only total runtime in minutes.

---

## 7. Summary

- One `config.yaml` defines everything
- `make_jobs` freezes config + job list
- `submit` and `resubmit` handle PBS job submission one-by-one
- `combine_results` merges outputs efficiently
- `status` gives you progress at a glance

All experiment state lives under `<output_dir>`, making it fully reproducible and self-contained.

