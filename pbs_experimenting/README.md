# PBS Experiment Framework (Refactored)

A modular, YAML-based framework for running large-scale DMD order detection experiments on HPC clusters with PBS job scheduling.

## Key Features

✅ **YAML-based configuration** - Worker scripts read YAML configs, not command-line arguments  
✅ **Proper random seed handling** - Deterministic, reproducible experiments  
✅ **Frozen experiment state** - All configs saved up-front for complete provenance  
✅ **Modular design** - Clean separation of concerns  
✅ **Upfront validation** - Catches configuration errors before job submission  
✅ **Method-agnostic** - Easy to add/remove methods via config  

## Quick Start

```bash
# 1. Configure your experiment
cp example_config.yaml my_experiment.yaml
# Edit my_experiment.yaml to set parameters

# 2. Generate job configurations
python experiment_runner.py make_jobs --config=my_experiment.yaml

# 3. Submit jobs to PBS
python experiment_runner.py submit --config=my_experiment.yaml

# 4. Monitor progress
python experiment_runner.py status --config=my_experiment.yaml

# 5. Combine results
python experiment_runner.py combine_results --config=my_experiment.yaml
```

## Directory Structure

After `make_jobs`, your output directory will look like:

```
experiment_output/
├── config.frozen.yaml          # Read-only snapshot of your config
├── job_configs/                # Individual job configs (read-only)
│   ├── job0000.yaml
│   ├── job0001.yaml
│   └── ...
├── jobs_state.json             # Tracks succeeded/failed/combined jobs
├── job_outputs/                # Per-job results
│   ├── job0000/
│   │   └── results.csv
│   └── ...
├── logs/                       # PBS stdout/stderr logs
│   ├── job_0000.out
│   ├── job_0000.err
│   └── ...
├── combined_results.csv        # Merged results from all jobs
└── merge_conflicts.csv         # Conflicts (if any)
```

## Configuration File

See `example_config.yaml` for a complete template. Key sections:

### Experiment Settings

```yaml
experiment:
  base_random_seed: 42  # Base seed; job N gets seed = base + N*1000
```

### Methods

List methods to evaluate (comment out to disable):

```yaml
methods:
  - BIC
  - GAP
  - ExactModeNorm
  - STC                      # Spatiotemporal Coupling
  - ESL-Norm                 # Estimated Subspace Leakage
  - NestedDMD                # Nested rank-1 DMD strategy
  - FixedEigenvalueBVFit     # Fixed-eigenvalue BV fit
  # Combination methods (merge features before clustering):
  - NestedDMD+ESL            # Combines NestedDMD with ESL-Norm
  - FixedEigenvalueBVFit+ESL # Combines FixedEigenvalueBVFit with ESL-Norm
```

### Clustering

Shared configuration for all clustering-based methods:

```yaml
clustering:
  algorithm: gmm        # Options: gmm, kmeans
  decider: mean         # Options: mean, distance
  normalization: min_max  # Options: min_max, standard, null
```

### Parameter Sweeps

Define parameter ranges and working points:

```yaml
generator:
  working_point:
    snr_db: 10.0
    freq_sep: 0.03
    # ... other parameters

parameters:
  snr_db:
    type: range
    role: wp          # One-axis-at-a-time around working point
    start: -5
    end: 20
    num_steps: 26
    scale: lin
  
  noise_mode:
    type: list
    role: cartesian   # Full Cartesian product
    values: ["gaussian", "uniform", "student_t"]
```

**Parameter Roles:**
- `wp` (working point): Varied one-at-a-time around the working point
- `cartesian`: Full Cartesian product with all other cartesian params

**Parameter Types:**
- `range`: Linear or log-spaced range
- `list`: Explicit list of values
- `const`: Single constant value

## CLI Commands

### `make_jobs`

Generate all job configurations and freeze the experiment.

```bash
python experiment_runner.py make_jobs --config=my_config.yaml
```

Creates:
- `config.frozen.yaml` (immutable snapshot)
- `job_configs/job####.yaml` (one per parameter combination)
- `jobs_state.json` (empty tracker)

### `submit`

Submit jobs to PBS queue.

```bash
# Submit all pending jobs
python experiment_runner.py submit --config=my_config.yaml

# Submit specific jobs
python experiment_runner.py submit --config=my_config.yaml --ids="0,5,10"
python experiment_runner.py submit --config=my_config.yaml --ids="0-99"
```

Skips jobs that have already succeeded or are currently running.

### `status`

Check experiment progress.

```bash
python experiment_runner.py status --config=my_config.yaml
```

Shows:
- Total jobs, succeeded, running, failed, pending
- Progress percentage
- Updates state file based on disk/PBS status

### `resubmit`

Resubmit all failed jobs.

```bash
python experiment_runner.py resubmit --config=my_config.yaml
```

### `combine_results`

Merge individual job CSVs into single file.

```bash
# Full combine (all finished jobs)
python experiment_runner.py combine_results --config=my_config.yaml

# Incremental (only new jobs)
python experiment_runner.py combine_results --config=my_config.yaml --incremental
```

Creates:
- `combined_results.csv` - All results
- `merge_conflicts.csv` - Conflicting runs (if any)

## Random Seed Handling

The system ensures full reproducibility:

1. **Base seed** is set in config: `experiment.base_random_seed: 42`
2. **Job seed** = `base_seed + job_id * 1000`
3. **Iteration seed** = `job_seed + iteration_index`

Example:
- Job 0, iteration 0: seed = 42
- Job 0, iteration 1: seed = 43
- Job 1, iteration 0: seed = 1042
- Job 1, iteration 1: seed = 1043

Both `base_random_seed` and job `random_seed` are saved to the output CSV.

To run a different random trial, just change `base_random_seed` in the config.

## Manual Job Execution

For debugging, run a single job manually:

```bash
# Run a specific job
python run_single_job.py run experiment_output/job_configs/job0042.yaml

# With plotting
python run_single_job.py run experiment_output/job_configs/job0042.yaml --plot

# Fire also supports this syntax
python run_single_job.py run --config_path=experiment_output/job_configs/job0042.yaml --plot
```

## Validation

Config validation runs automatically during `make_jobs`, but you can validate manually:

```bash
python config_validator.py my_config.yaml
```

Checks:
- Valid method names
- Parameter compatibility (e.g., `max_rank < temporal_dim * (1 - delays_over_timesteps)`)
- Required fields present
- PBS configuration
- Path validity

## Module Overview

| Module | Purpose |
|--------|---------|
| `experiment_runner.py` | Main CLI driver (Fire-based) |
| `run_single_job.py` | Worker script for single job |
| `config_validator.py` | Upfront config validation |
| `param_generator.py` | Parameter combination generation |
| `job_config_builder.py` | Build individual job YAMLs |
| `pbs_utils.py` | PBS submission utilities |
| `state_manager.py` | Job state tracking |
| `result_combiner.py` | CSV merging |

## Comparison with Old System

| Feature | Old System | New System |
|---------|------------|------------|
| Worker input | Command-line args | YAML config |
| Random seeds | Not handled | Deterministic per job |
| Job configs | Generated on-the-fly | Pre-generated, frozen |
| Methods | Hardcoded | Configurable list |
| Validation | Runtime errors | Upfront validation |
| Modularity | Monolithic functions | Class-based design |
| Reproducibility | Limited | Full provenance |

## Advanced Usage

### Custom Method Configuration

To add a new method:

1. Add to `VALID_METHODS` in `config_validator.py`
2. Add to `methods` list in your config
3. Implement evaluation in `MethodEvaluator` class in `run_single_job.py`

### Partial Experiments

You can create small test experiments:

```yaml
parameters:
  snr_db:
    type: list
    role: cartesian
    values: [0, 10, 20]  # Just 3 values
  
  noise_mode:
    type: const
    value: "gaussian"    # Single value
```

This creates only 3 jobs for quick testing.

### Checkpointing

The system is checkpoint-friendly:
- Jobs can be submitted in batches
- Failed jobs can be resubmitted anytime
- Results can be combined incrementally
- All state is persistent in `jobs_state.json`

## Troubleshooting

### "No job configs found"

You need to run `make_jobs` first to generate configurations.

### "PBS is not available"

The system requires `qsub` and `qstat` commands. Run on an HPC login node.

### Jobs failing silently

Check PBS logs in `experiment_output/logs/`:
```bash
tail experiment_output/logs/job_0042.err
```

### Results conflicts

If `merge_conflicts.csv` is created, different jobs produced different results for the same parameters. This indicates:
- Non-deterministic behavior (check random seeds)
- Parameter hash collision (very rare)
- Jobs ran with different code versions

## Citation

If you use this framework, please cite:

```
[Your paper/repo citation here]
```

## License

[Your license here]

