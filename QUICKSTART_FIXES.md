# Implementation Guide - Quick Start

## TL;DR (Too Long; Didn't Read)

You need to make changes in **3 files** to fix the NaN issue:

1. **allreducer.py** - Track sparsity metrics
2. **distributed_optimizer.py** - Pass trainer reference and fix deprecated API
3. **lstm_oktopk_debug_polaris.sh** - Reduce learning rate

---

## The Problem

Your training shows:
```
WARNING NaN detected! sparsities: []
loss: nan
val loss: 0.000000, val acc: 1.000000
```

This happens because:
1. ✗ Sparsity metrics never get calculated (`sparsities` list is empty)
2. ✗ Learning rate is too high (1.0 causes gradient explosion)
3. ✗ PyTorch API is deprecated (warning but not critical)

---

## The Solution (3 Steps)

### Step 1: Fix allreducer.py (Sparsity Tracking)

This is the MOST IMPORTANT fix.

**Edit: `/home/mrest/Ok-Topk-Analysis/LSTM/allreducer.py`**

**Change 1A:** Line 217 - Add `trainer=None` parameter
```python
# OLD:
def __init__(self, ..., writer=None):

# NEW:
def __init__(self, ..., writer=None, trainer=None):
```

**Change 1B:** Line ~233 - Store trainer reference
```python
# Add this line after self._norm_clip = norm_clip
self._trainer = trainer
```

**Change 1C:** Line ~1603 - Add sparsity tracking (INSERT BEFORE `elif self._sparse and self._compression.name == 'topkAopt':`)
```python
# Track compression metrics for the trainer
if self._trainer is not None and self._sparse and self._compression.name == 'oktopk':
    try:
        global_topk_count = all_gindexes.size if 'all_gindexes' in locals() and all_gindexes is not None else 0
        tensor_size = torch.numel(new_tensor.data)
        
        if tensor_size > 0 and global_topk_count > 0:
            sparsity = 1.0 - (float(global_topk_count) / float(tensor_size))
            compression_ratio = float(tensor_size) / float(global_topk_count)
            communication_size = global_topk_count * 2 * 4
            
            self._trainer.sparsities.append(sparsity)
            self._trainer.compression_ratios.append(compression_ratio)
            self._trainer.communication_sizes.append(communication_size)
    except Exception as e:
        if rank == 0:
            logger.warning('Failed to track compression metrics: %s', str(e))
```

---

### Step 2: Fix distributed_optimizer.py (Connect Trainer & Fix API)

**Edit: `/home/mrest/Ok-Topk-Analysis/LSTM/distributed_optimizer.py`**

**Change 2A:** Line ~80 - Add `trainer=None` to AllReducer instantiation
```python
# Find the AllReducer(...) call and add trainer=None as last parameter:
self._allreducer = ar.AllReducer(
    ...existing parameters...,
    writer=writer,
    trainer=None  # ADD THIS
)
```

**Change 2B:** Add setter method to _DistributedOptimizer class
```python
# Add this method in the _DistributedOptimizer class
def set_trainer(self, trainer):
    """Set trainer reference for metrics tracking"""
    if self._allreducer is not None:
        self._allreducer._trainer = trainer
```

**Change 2C:** Line 129 - Fix deprecated add_ call
```python
# OLD:
d_p.add_(weight_decay, p.data)

# NEW:
d_p.add_(p.data, alpha=weight_decay)
```

---

### Step 3: Fix Learning Rate in main_trainer.py

**Edit: `/home/mrest/Ok-Topk-Analysis/LSTM/main_trainer.py`**

**Change 3A:** After trainer.update_optimizer(optimizer) (around line 70)
```python
trainer.update_optimizer(optimizer)
if hasattr(optimizer, 'set_trainer'):
    optimizer.set_trainer(trainer)
```

**Change 3B:** Around line 170 - Default learning rate
```python
# OLD:
parser.add_argument('--lr', type=float, default=1.0, help='learning rate')

# NEW:
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
```

---

## File Summary

| File | Lines | What Changed | Why |
|------|-------|--------------|-----|
| allreducer.py | 217, 233, 1603 | Add trainer param, store ref, track metrics | So sparsity metrics get populated |
| distributed_optimizer.py | 80, ~150, 129 | Add trainer param, add setter, fix API | To connect trainer to allreducer and fix deprecated warning |
| main_trainer.py | ~70, ~170 | Call set_trainer, reduce lr | To initialize metrics tracking and prevent gradient explosion |

---

## Testing Your Fix

After making all changes:

```bash
cd /home/mrest/Ok-Topk-Analysis/LSTM

# Single worker test
python main_trainer.py \
  --dnn lstman4 \
  --dataset an4 \
  --data_dir ./audio_data \
  --nworkers 1 \
  --max_epochs 1 \
  --batch_size 8 \
  --lr 0.01 \
  --compression True \
  --compressor oktopk
```

**BEFORE FIX:**
```
WARNING NaN detected! sparsities: []
Average Sparsity: nan, compression ratio: nan, communication size: nan
loss: nan
```

**AFTER FIX:**
```
Average Sparsity: 0.98, compression ratio: 50.5, communication size: 123456
loss: 0.123
val loss: 2.345, val acc: 0.567
```

---

## If You Want to Apply All Fixes at Once

See `COPYPASTE_FIXES.md` for exact code blocks you can copy and paste.

See `DETAILED_CHANGES.md` for step-by-step line-by-line instructions.

---

## Priority Order

1. **MUST DO**: allreducer.py Changes (1A, 1B, 1C) - This fixes the NaN sparsities warning
2. **MUST DO**: main_trainer.py Changes (3A, 3B) - This prevents gradient explosion
3. **SHOULD DO**: distributed_optimizer.py Changes (2A, 2B, 2C) - This connects everything and removes warnings

---

## Common Mistakes to Avoid

❌ Don't forget the `trainer=None` parameter in ALL three places (allreducer __init__, AllReducer call, when calling it)

❌ Don't add the sparsity tracking code in the wrong place (must be after compress_t2 is set but before elif)

❌ Don't forget to call `optimizer.set_trainer(trainer)` in main_trainer.py

❌ Don't change BOTH the script AND main_trainer.py for learning rate (pick one to avoid confusion)

✓ Do test with `--nworkers 1` first before using MPI

✓ Do keep the `try/except` block around sparsity tracking (in case all_gindexes isn't defined in some code paths)

---

## Support Reference Files

- `FIXES_AND_CHANGES.md` - Complete problem analysis and solution strategy
- `DETAILED_CHANGES.md` - Exact line numbers and code context
- `COPYPASTE_FIXES.md` - Ready-to-copy code blocks

