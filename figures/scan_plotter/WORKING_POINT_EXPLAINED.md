# Working Point - Detailed Explanation

## What Is It?

A **working point** is a complete specification of where you are in parameter space - it fixes all parameters except the one(s) you're scanning.

## Visual Example

### Your CSV Data (simplified)

```
snr_db | freq_sep | num_modes | noise_mode | method      | order_hit_prob
-------|----------|-----------|------------|-------------|---------------
0      | 0.01     | 2         | gaussian   | ESL-Norm    | 0.3
5      | 0.01     | 2         | gaussian   | ESL-Norm    | 0.7
10     | 0.01     | 2         | gaussian   | ESL-Norm    | 0.95
0      | 0.01     | 2         | student_t  | ESL-Norm    | 0.25
5      | 0.01     | 2         | student_t  | ESL-Norm    | 0.65
10     | 0.01     | 2         | student_t  | ESL-Norm    | 0.90
0      | 0.01     | 3         | gaussian   | ESL-Norm    | 0.2
5      | 0.01     | 3         | gaussian   | ESL-Norm    | 0.5
...
```

### Scenario 1: Single Scan

**Goal**: Plot order_hit_prob vs. snr_db for num_modes=2, noise_mode=gaussian

**Working Point**:
```python
working_point = {
    'freq_sep': 0.01,
    'num_modes': 2,
    'noise_mode': 'gaussian'
}
x_param = 'snr_db'
```

**What `filter_data()` does**:
1. Keep only rows where `freq_sep == 0.01` ✓
2. Keep only rows where `num_modes == 2` ✓
3. Keep only rows where `noise_mode == 'gaussian'` ✓
4. **Don't filter** `snr_db` (it's in `exclude_params`)

**Result**: You get these rows:
```
snr_db | freq_sep | num_modes | noise_mode | method      | order_hit_prob
-------|----------|-----------|------------|-------------|---------------
0      | 0.01     | 2         | gaussian   | ESL-Norm    | 0.3    ← plot this
5      | 0.01     | 2         | gaussian   | ESL-Norm    | 0.7    ← plot this
10     | 0.01     | 2         | gaussian   | ESL-Norm    | 0.95   ← plot this
```

Now you can plot snr_db vs order_hit_prob!

### Scenario 2: Multi-Panel Scan

**Goal**: Same as above, but with 4 panels for different noise modes

**Working Point**:
```python
working_point = {
    'freq_sep': 0.01,
    'num_modes': 2,
    # NOTE: noise_mode NOT specified - it's the panel_param!
}
x_param = 'snr_db'
panel_param = 'noise_mode'
panel_values = ['gaussian', 'student_t', 'hetero', 'bi_gaussian']
```

**What happens**:
1. Filter by working_point (excludes both `snr_db` and `noise_mode`)
2. For each panel value:
   - Filter further: `noise_mode == 'gaussian'` → Panel 1
   - Filter further: `noise_mode == 'student_t'` → Panel 2
   - etc.

**Result**: 4 side-by-side plots, each showing snr_db vs order_hit_prob for a different noise mode

## Why Is This Manual?

**Because you need to tell the system what slice of your data to use.**

Your CSV has hundreds of parameter combinations. The system can't guess which one you want to visualize.

### Common Pattern

In practice, you typically:
1. Define a **base working point** with parameters that are constant across all your plots
2. Override specific parameters for each plot

Example in YAML:
```yaml
base:
  working_point:
    freq_sep: 0.01        # Always this value
    eig_mag: 0.98         # Always this value
    spatial_dim: 45       # Always this value
    # ... other fixed params

plots:
  # Override just what varies
  - working_point: {num_modes: 2}
    x_param: snr_db
    
  - working_point: {num_modes: 3}
    x_param: snr_db
    
  - working_point: {num_modes: 5}
    x_param: snr_db
```

The system **merges** these, so the effective working point for plot 1 is:
```yaml
{freq_sep: 0.01, eig_mag: 0.98, spatial_dim: 45, num_modes: 2}
```

## The Code

Here's exactly what happens in `filter_data()`:

```python
def filter_data(df, working_point, exclude_params=None):
    exclude_params = exclude_params or []
    filtered = df.copy()
    
    for param, value in working_point.items():
        # Skip if this param is being scanned (x_param or panel_param)
        if param in exclude_params:
            continue
        
        # Filter rows to match this parameter value
        if isinstance(value, float):
            filtered = filtered[np.isclose(filtered[param], value)]
        else:
            filtered = filtered[filtered[param] == value]
    
    return filtered
```

**Simple**: For each parameter in working_point, keep only rows matching that value.

## Key Insight

**Working point = "WHERE am I in parameter space?"**

If you don't specify it fully, you'll mix data from different parts of parameter space, which makes the plot meaningless.

## Practical Tip

For your data, always specify:
- `num_modes` (2, 3, or 5)
- `freq_sep` (usually 0.01 for SNR scans, or varies for freq_sep scans)
- `eig_mag` (usually 0.98)
- `snr_db` (varies for SNR scans, usually 8.0 for other scans)
- `spatial_dim` (usually 45)
- `delays_over_timesteps` (usually 0.32)
- `noise_mode` (specify if single scan, omit if panel_param)
- All the constants: `eig_mag_spread`, `rho_mode`, `temporal_dim`, etc.

The batch config does this for you - you just override the few params that change between plots.

