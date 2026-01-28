# Critical Issues Found & Fixed - Second Run

## Issues Discovered

**Issue #1: Learning Rate Still 1.0**
- Problem: Config file overrode default learning rate
- File: `exp_configs/lstm_debug.conf`
- Fix: Changed `lr="${lr:-1.0}"` → `lr="${lr:-0.01}"`

**Issue #2: Trainer Reference Not Propagating**
- Problem: `DistributedOptimizer` is a factory function that creates a dynamic class
- The `set_trainer()` method wouldn't be recognized via `hasattr()` on the dynamic class
- File: `main_trainer.py` line ~71
- Fix: Changed from conditional check to try/except with fallback:
  ```python
  try:
      optimizer.set_trainer(trainer)
  except AttributeError:
      # Fallback for dynamic class wrapper
      if hasattr(optimizer, '_allreducer'):
          optimizer._allreducer._trainer = trainer
  ```

**Issue #3: Sparsity Metrics Code in Wrong Scope**
- Problem: Metrics tracking code was placed AFTER all_gindexes went out of scope
- The code was placed after the `if self._profiling:` block at the SAME indentation as `elif self._sparse and self._compression.name == 'topkAopt':`
- This meant `all_gindexes` variable was unavailable when trying to access it with `'all_gindexes' in locals()`
- File: `allreducer.py`
- Fix: 
  - MOVED metrics code from line ~1092 (wrong location)
  - TO line ~1051 (inside the oktopk block, right after `result[all_gindexes_tensor] = all_gvalues_tensor`)
  - This ensures `all_gindexes` is available since it's still in scope within the oktopk block

## Files Modified

1. **exp_configs/lstm_debug.conf** - Changed lr default to 0.01
2. **main_trainer.py** - Improved trainer reference initialization with fallback
3. **allreducer.py** - Moved metrics code to correct scope inside oktopk block

## Expected Behavior Now

Next run should show:
```
✓ Average Sparsity: 0.XX (not nan)
✓ Compression ratio: X.X (not nan)
✓ loss: Y.YYY (not nan)
✓ val loss: Z.ZZZ (not 0.000)
✓ val acc: A.AAA (not 1.000)
```

