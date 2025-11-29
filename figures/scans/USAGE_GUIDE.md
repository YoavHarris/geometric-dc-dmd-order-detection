# Quick Start Guide: Custom Axis Labels

## Overview

You can now override axis labels for both X and Y axes, with full LaTeX support for mathematical notation.

## How It Works

By default, labels are auto-generated from parameter names:
- `snr_db` becomes "Snr Db"
- `order_hit_prob` becomes "Order Hit Prob"

With the new feature, you can specify custom labels:
- `xlabel: "SNR (dB)"` displays as "SNR (dB)"
- `ylabel: "$P_{\\mathrm{hit}}$"` displays as P_hit (rendered as LaTeX)

## Quick Examples

### 1. Basic YAML Config

```yaml
plots:
  - working_point: {num_modes: 2}
    x_param: snr_db
    metric: order_hit_prob
    # Custom labels (no LaTeX)
    xlabel: "Signal-to-Noise Ratio (dB)"
    ylabel: "Hit Probability"
    output_path: output/my_plot.png
```

### 2. LaTeX Math Notation

```yaml
plots:
  - working_point: {num_modes: 3}
    x_param: snr_db
    metric: order_hit_prob
    # LaTeX labels with math symbols
    xlabel: "$\\mathrm{SNR}$ (dB)"
    ylabel: "$P_{\\mathrm{hit}}$"
    output_path: output/latex_plot.png
```

### 3. Advanced LaTeX

```yaml
plots:
  - working_point: {num_modes: 2}
    x_param: freq_sep
    metric: order_hit_prob
    # Complex LaTeX notation
    xlabel: "$\\Delta f$ (normalized)"
    ylabel: "$\\mathbb{P}(\\hat{r} = r)$"
    output_path: output/advanced_latex.png
```

### 4. Partial Override

```yaml
plots:
  - working_point: {num_modes: 2}
    x_param: eig_mag
    metric: order_hit_prob
    # Only override xlabel, ylabel uses auto-format
    xlabel: "$|\\lambda|$ (eigenvalue magnitude)"
    output_path: output/partial.png
```

## LaTeX Cheat Sheet

| What You Want | YAML Code | Description |
|---------------|-----------|-------------|
| Greek letter | `$\\alpha$` | alpha |
| Delta | `$\\Delta f$` | Delta f |
| Subscript | `$P_{hit}$` | P with subscript hit |
| Superscript | `$\\sigma^2$` | sigma squared |
| Bold/Roman | `$\\mathrm{SNR}$` | SNR (upright) |
| Probability | `$\\mathbb{P}(...)$` | Probability notation |
| Hat | `$\\hat{r}$` | r with hat |
| Absolute value | `$|\\lambda|$` | absolute lambda |
| Fraction | `$\\frac{a}{b}$` | vertical a over b |

**Important:** Use double backslashes (`\\`) in YAML strings!

## Command-Line Usage

```bash
# With custom labels
python cli.py single \
  --csv_path=data.csv \
  --x_param=snr_db \
  --metric=order_hit_prob \
  --xlabel='$\mathrm{SNR}$ (dB)' \
  --ylabel='$P_{\mathrm{hit}}$' \
  --working_point="{'num_modes': 2}" \
  --output_path=output.png

# Without custom labels (uses auto-format)
python cli.py single \
  --csv_path=data.csv \
  --x_param=snr_db \
  --metric=order_hit_prob \
  --working_point="{'num_modes': 2}" \
  --output_path=output.png
```

## Multi-Panel Plots

Labels apply to all panels:

```yaml
plots:
  - working_point: {num_modes: 2}
    x_param: snr_db
    panel_param: noise_mode
    panel_values: [gaussian, student_t, hetero]
    xlabel: "$\\mathrm{SNR}$ (dB)"
    ylabel: "$P_{\\mathrm{hit}}$"
    output_path: output/multi_panel.png
```

All panels will have the same custom xlabel and ylabel.

## Combining with Other Features

You can combine label overrides with all other features:

```yaml
plots:
  - working_point: {num_modes: 2}
    x_param: snr_db
    metric: order_hit_prob
    # Custom labels
    xlabel: "$\\mathrm{SNR}$ (dB)"
    ylabel: "$P_{\\mathrm{hit}}$"
    # X-axis control
    xscale: linear
    xlim: [6.0, 10.0]
    # Method filtering
    methods: [ESL-Norm, FixedEigenvalueBVFit, NestedDMD]
    # Title
    title: "Performance vs SNR"
    output_path: output/full_control.png
```

## Tips

1. **Test LaTeX syntax**: If unsure, test your LaTeX in a simple document first
2. **Escaping in YAML**: Always use `\\` for backslashes in YAML strings
3. **Shell escaping**: Use single quotes in command-line when passing LaTeX: `--xlabel='$\alpha$'`
4. **Optional parameters**: Omit `xlabel` or `ylabel` to use auto-formatting
5. **Consistency**: In multi-panel plots, one label applies to all panels

## Full Example

See `examples/label_override_example.yaml` for a comprehensive example with 5 different use cases.

## Troubleshooting

**Problem**: LaTeX not rendering
- **Solution**: Check for double backslashes in YAML: `\\alpha` not `\alpha`

**Problem**: Shell escaping issues
- **Solution**: Use single quotes in command-line: `--xlabel='$\alpha$'`

**Problem**: Only one label showing
- **Solution**: Both `xlabel` and `ylabel` are independent - you can set just one

**Problem**: Labels look ugly
- **Solution**: Try wrapping text parts in `\\mathrm{...}`: `$\\mathrm{SNR}$ (dB)` looks better than `$SNR$ (dB)`

