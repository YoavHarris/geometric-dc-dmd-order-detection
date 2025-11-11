# PBS Experiment Framework - Refactoring Notes

## Summary

Successfully refactored the PBS experiment system with the following improvements:

### ✅ Completed

1. **YAML-based job configs** - Workers now read YAML instead of command-line arguments
2. **Proper random seed handling** - Deterministic seeds: `base_seed + job_id * 1000 + iteration`
3. **Updated method names**:
   - ❌ Removed: `CDFAUC`, `MSMS`, `ModeEnergy`
   - ✅ Replaced: `BlockVandermondeMatching` → `FixedEigenvalueBVFit`
   - ✅ Replaced: `ModeMatDMD` → `NestedDMD`
   - ✅ Replaced: `dcDMD` → `STC`
4. **Modular design** - Separated concerns into classes:
   - `MethodEvaluator` - Handles method evaluation
   - `MetricsTracker` - Tracks metrics across iterations
5. **Upfront validation** - Config errors caught before submission
6. **Pre-generated configs** - All job YAMLs created upfront for transparency

### 📁 New Directory Structure

```
pbs_experimenting_refactor/
├── experiment_runner.py          # Main CLI (Fire-based)
├── run_single_job.py             # Worker (Fire-based, reads YAML)
├── config_validator.py           # Validation logic
├── param_generator.py            # Parameter sweeps
├── job_config_builder.py         # Build job YAMLs
├── pbs_utils.py                  # PBS submission
├── state_manager.py              # State tracking
├── result_combiner.py            # CSV merging
├── example_config.yaml           # Template
├── README.md                     # Documentation
└── REFACTORING_NOTES.md          # This file
```

### 🔥 Python Fire Throughout

Both CLI scripts use Python Fire for consistency:
- `experiment_runner.py`: Fire-based with `ExperimentRunner` class
- `run_single_job.py`: Fire-based with `run()` function

Benefits:
- Consistent interface
- Auto-generated help
- Flexible argument syntax
- No manual argparse boilerplate

## No Changes Needed to Existing Methods

The refactored code works with the current method implementations:

✅ **STC** (`algorithms/stc.py`)
  - Interface: `compute_features(eigenvalues, modes, plot)`
  - Returns: `{"STC": scores}`
  - Status: Working as-is

✅ **NestedDMD** (`algorithms/block_vandermonde_fit.py`)
  - Interface: `compute_features(modes, eigenvalues, plot)`
  - Returns: `{"Reconstruction": scores, "Eigenvalue-Consistency": scores}`
  - Status: Working as-is

✅ **FixedEigenvalueBVFit** (`algorithms/block_vandermonde_fit.py`)
  - Interface: `compute_features(modes, eigenvalues, plot)`
  - Returns: `{"BV-Fit": scores}`
  - Status: Working as-is

All three methods follow the consistent `compute_features(...)` API and return score dictionaries compatible with `ModeClustering`.

## Key Design Decisions

### 1. Random Seeds

**Approach:** Deterministic per-job seeding
```python
job_seed = base_random_seed + job_id * 1000
iteration_seed = job_seed + iteration_index
```

**Rationale:**
- Full reproducibility
- Easy ablation (change base seed)
- No seed collisions between jobs

### 2. Job Configs

**Approach:** Pre-generate all YAML configs (read-only)

**Rationale:**
- Complete transparency
- Easy debugging (`python run_single_job.py job_configs/job0042.yaml`)
- Audit trail for scientific reproducibility
- Consistent with "frozen experiment" philosophy

### 3. Method Configuration

**Approach:** Simple list of method names

```yaml
methods:
  - AIC
  - STC
  - NestedDMD
```

**Rationale:**
- Methods don't have many tunable parameters
- Clustering config is shared (separate section)
- Easy to add/remove methods
- No unnecessary complexity

### 4. Modularity

**Classes:**
- `MethodEvaluator`: Encapsulates method evaluation logic
- `MetricsTracker`: Tracks metrics with clean aggregation
- `ExperimentRunner`: Fire-based CLI with clear command separation

**Benefits:**
- Easy to extend
- Testable
- Clear separation of concerns
- Each class has single responsibility

## Comparison with Old System

| Aspect | Old | New |
|--------|-----|-----|
| Worker input | CLI args string | YAML config file |
| Config format | `--snr_db=10 --freq_sep=0.03 ...` | Structured YAML |
| Random seeds | `None` or ad-hoc | Deterministic per-job |
| Job configs | Not saved | Pre-generated, frozen |
| Methods | Hardcoded in runner | Configurable list |
| Validation | Runtime failures | Upfront validation |
| Code structure | Long procedural functions | Modular classes |
| Method names | Old deprecated names | Current method names |

## Migration Guide

To switch from old to new system:

1. **Convert config:**
   ```bash
   # Old: config.yaml with CLI-style args
   # New: Use example_config.yaml template
   ```

2. **Update method names in config:**
   ```yaml
   # Old methods (remove):
   # - dcDMD
   # - ModeMatDMD
   # - BlockVandermondeMatching
   # - ModeEnergy
   # - MSMS
   # - CDFAUC
   
   # New methods (use):
   - STC
   - NestedDMD
   - FixedEigenvalueBVFit
   ```

3. **Run new system:**
   ```bash
   python experiment_runner.py make_jobs --config=new_config.yaml
   python experiment_runner.py submit --config=new_config.yaml
   ```

## Output Format Compatibility

✅ **CSV format is unchanged** - Results can be combined with old data

Column structure remains the same:
- All parameter columns
- `base_random_seed` (NEW)
- `random_seed` (NEW - job-specific seed)
- Method metrics (same as before)

## Testing Recommendations

1. **Small test run:**
   ```yaml
   parameters:
     snr_db:
       type: list
       role: cartesian
       values: [0, 10]
     noise_mode:
       type: const
       value: "gaussian"
   static_args:
     n_iter: 10  # Small number for testing
   ```

2. **Manual job test:**
   ```bash
   python run_single_job.py run job_configs/job0000.yaml --plot
   ```

3. **Validate config:**
   ```bash
   python config_validator.py my_config.yaml
   ```

## Known Limitations

1. **PBS-specific** - Requires `qsub`/`qstat` commands
2. **No SLURM support** - Would need separate implementation
3. **No job dependencies** - All jobs are independent
4. **No automatic retry** - Use `resubmit` command manually

## Future Enhancements (Optional)

- [ ] Add SLURM support alongside PBS
- [ ] Parallel local execution mode (no PBS)
- [ ] Real-time progress dashboard
- [ ] Automatic result visualization
- [ ] Per-method configuration support
- [ ] Job dependency management
- [ ] Auto-retry on transient failures

## Questions?

See `README.md` for full usage documentation.

