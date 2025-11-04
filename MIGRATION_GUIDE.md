# Migration Guide: Block-Vandermonde Fit Refactoring

## What Changed

The `ModeNestedDMD` class with `mode` parameter has been refactored into:
- **Base class**: `BlockVandermondeFit` (shared functionality)
- **Strategy 1**: `NestedDMD` (nested rank-1 DMD)
- **Strategy 2**: `FixedEigenvalueBVFit` (closed-form FEBVF)
- **Deprecated**: `ModeNestedDMD` (kept for backward compatibility)

## Why?

The old name `ModeNestedDMD` was misleading because:
- It suggested only nested DMD, but also implemented FEBVF
- The `mode` parameter was a code smell (two algorithms in one class)

The new design:
- Clear names that describe what each class does
- Better separation of concerns
- Easier to extend with new strategies
- More maintainable

---

## Migration Path

### Old Code (Still Works, But Deprecated)

```python
from algorithms.block_vandermonde_fit_vec import ModeNestedDMD

# Nested DMD (default)
bv_fit = ModeNestedDMD(num_delays=10, spatial_dim=5, mode="nested_dmd")
features = bv_fit.compute_features(modes, eigenvalues)

# FEBVF
bv_fit = ModeNestedDMD(num_delays=10, spatial_dim=5, mode="febvf")
features = bv_fit.compute_features(modes, eigenvalues)
```

### New Code (Recommended)

```python
from algorithms.block_vandermonde_fit_vec import NestedDMD, FixedEigenvalueBVFit

# Nested DMD
nested = NestedDMD(num_delays=10, spatial_dim=5)
features = nested.compute_features(modes, eigenvalues)

# FEBVF
febvf = FixedEigenvalueBVFit(num_delays=10, spatial_dim=5)
features = febvf.compute_features(modes, eigenvalues)
```

---

## API Changes

### Constructor

**Old:**
```python
ModeNestedDMD(num_delays, spatial_dim, epsilon=..., mode="nested_dmd"|"febvf")
```

**New:**
```python
# Choose the right class instead of using mode parameter
NestedDMD(num_delays, spatial_dim, epsilon=...)
FixedEigenvalueBVFit(num_delays, spatial_dim, epsilon=...)
```

### compute_features()

**No changes!** Same signature and return format:

```python
features = strategy.compute_features(
    modes,           # (D*L, M)
    eigenvalues,     # (M,)
    plot=False,
    **kwargs         # For NestedDMD: fit_dmd_kwargs
)
```

**Returns:**
- `NestedDMD`: `{"Reconstruction": ..., "Eigenvalue-Consistency": ..., *_raw: ...}`
- `FixedEigenvalueBVFit`: `{"BV-Fit": ..., "BV-Fit_raw": ...}`

---

## Search and Replace Guide

### Simple Cases

If you're only using one strategy per file:

**For nested DMD users:**
```bash
# Find
from algorithms.block_vandermonde_fit_vec import ModeNestedDMD
# Replace with
from algorithms.block_vandermonde_fit_vec import NestedDMD

# Find
ModeNestedDMD(
# Replace with
NestedDMD(

# Remove mode parameter
, mode="nested_dmd"  # Delete this
```

**For FEBVF users:**
```bash
# Find
from algorithms.block_vandermonde_fit_vec import ModeNestedDMD
# Replace with
from algorithms.block_vandermonde_fit_vec import FixedEigenvalueBVFit

# Find
ModeNestedDMD(
# Replace with
FixedEigenvalueBVFit(

# Remove mode parameter
, mode="febvf"  # Delete this
```

### Dynamic Cases

If you're switching between strategies at runtime:

**Old:**
```python
strategy = "nested_dmd"  # or "febvf"
bv_fit = ModeNestedDMD(num_delays, spatial_dim, mode=strategy)
```

**New:**
```python
STRATEGIES = {
    "nested_dmd": NestedDMD,
    "febvf": FixedEigenvalueBVFit,
}
strategy = "nested_dmd"  # or "febvf"
bv_fit = STRATEGIES[strategy](num_delays, spatial_dim)
```

---

## Timeline

- **Now**: Old `ModeNestedDMD` still works but shows deprecation warning
- **Next release**: Deprecation warning continues
- **Future release**: `ModeNestedDMD` will be removed

**Action Required**: Update your code to use the new classes.

---

## Benefits of New Design

1. **Clear names**: Class name tells you what it does
2. **Type safety**: No invalid mode strings
3. **IDE support**: Better autocomplete and documentation
4. **Extensibility**: Easy to add new strategies
5. **Testability**: Each strategy is independently testable

---

## Need Help?

If you have code that's hard to migrate, please open an issue with:
- Your use case
- Code snippet showing the problem
- Expected behavior

We'll help you migrate!

