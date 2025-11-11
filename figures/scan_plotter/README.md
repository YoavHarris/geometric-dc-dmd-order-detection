# Parameter Scan Plotter

Clean, minimal system for plotting parameter scans. 223 lines of code, fully config-driven.

## Architecture

**Two core classes:**

1. **`SingleScanPlotter`** - Draws one scan on one axis
   - Input: axis, data, x_param, metric
   - Output: Lines on that axis

2. **`PanelComposer`** - Arranges multiple scans into panels
   - Input: List of panel specs
   - Output: Figure with panels

**All styling in `config.yaml`** - Colors, fonts, sizes, linestyles

## Usage

### Single Scan

```bash
python cli.py single \
  --csv_path=data.csv \
  --x_param=snr_db \
  --metric=order_hit_prob \
  --working_point="{'num_modes': 2, 'noise_mode': 'gaussian'}" \
  --output_path=output.png \
  --title="SNR Scan"
```

### Multi-Panel (Parameter Variation)

```bash
python cli.py multi \
  --csv_path=data.csv \
  --x_param=snr_db \
  --metric=order_hit_prob \
  --working_point="{'num_modes': 2}" \
  --panel_param=noise_mode \
  --panel_values="['gaussian', 'student_t', 'hetero', 'bi_gaussian']" \
  --output_path=output.png \
  --title="SNR Scan by Noise Type"
```

### Batch Processing

```bash
python cli.py batch --config_path=examples/simple_batch.yaml
```

## Batch Config Format

```yaml
base:
  csv_path: data.csv
  metric: order_hit_prob
  panel_param: noise_mode
  panel_values: [gaussian, student_t, hetero, bi_gaussian]
  
  working_point:
    # Common parameters
    rho_mode: random
    temporal_dim: 200

plots:
  # Each plot merges base + specific settings
  - working_point: {num_modes: 2, snr_db: 10}
    x_param: freq_sep
    output_path: output/freq_sep_nm2.png
    title: "Frequency Separation (num_modes=2)"
```

## Working Points

**A working point defines WHERE you are in parameter space.**

Your CSV has many parameter combinations (SNR, frequency separation, num_modes, noise_mode, etc.). A "scan" means:
1. **Fix** all parameters at specific values (the working point)
2. **Vary** just one parameter (x_param)
3. Plot how the metric changes

Example:
```yaml
working_point:
  freq_sep: 0.01      # Fixed
  num_modes: 2        # Fixed
  noise_mode: gaussian # Fixed
  # ... all other params fixed

x_param: snr_db       # This varies → becomes the X-axis
```

This gives you: "order_hit_prob vs SNR for num_modes=2, noise_mode=gaussian, freq_sep=0.01"

**See `WORKING_POINT_EXPLAINED.md` for detailed examples with data.**

### Rules
- **Single scan**: Specify ALL parameters except `x_param`
- **Multi-panel**: Specify ALL parameters except `x_param` and `panel_param`
- Each (x_value, panel_value, method) should map to exactly ONE row in your CSV

## Design Configuration

Edit `config.yaml` to change:
- Method colors and linestyles
- Figure dimensions and DPI
- Font sizes
- Line widths
- Legend styling

This file is rarely touched - it's your design system.

## Use Cases

All use cases use the same two building blocks:

- **Single scan** → 1 panel
- **Multi-panel by parameter** → N panels, vary parameter
- **Multi-metric** → N panels, vary metric (same pattern)
- **Grid layout** → 2D arrangement (horizontal + vertical)

No special-case code needed - just compose panels differently.

## File Structure

```
scan_plotter/
├── config.yaml          # Design configuration (colors, fonts, etc)
├── plotter.py          # Core classes (223 lines total)
├── cli.py              # Command-line interface
├── requirements.txt    
└── examples/
    └── simple_batch.yaml
```

## Installation

```bash
pip install -r requirements.txt
```

Requires: matplotlib, pandas, pyyaml, fire, numpy

## Philosophy

- **Minimal code, maximal config** - Static choices in YAML, not Python
- **Composable building blocks** - SingleScanPlotter + PanelComposer
- **No special cases** - All use cases use the same core classes
- **Config-driven** - Styling and design in config files, not code

