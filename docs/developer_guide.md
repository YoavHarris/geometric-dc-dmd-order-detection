# Developer Guide

## Adding a New Method

To add a new order detection method to the framework:

1.  **Create Feature Extractor**:
    Implement a class in `algorithms/your_method.py` with a `compute_features` method:
    ```python
    class YourMethod:
        def compute_features(self, modes, eigenvalues, plot=False):
            # Compute per-mode scores
            scores = ...  # shape (M,)
            return {"YourFeatureName": scores}
    ```

2.  **Register in Experiment Framework**:
    *   Add method name to `VALID_METHODS` in `pbs_experimenting/config_validator.py`.
    *   Implement evaluation logic in `MethodEvaluator` class in `pbs_experimenting/run_single_job.py`.

3.  **Update Plotting Config**:
    Add your method's preferred color and linestyle to `figures/scans/design_config.yaml`.

## Running Tests

Individual experiments can be run manually for debugging:

```bash
python pbs_experimenting/run_single_job.py run pbs_experimenting/job_configs/test_job.yaml --plot
```
