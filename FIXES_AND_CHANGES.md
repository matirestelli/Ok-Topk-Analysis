# Ok-Topk Analysis: Code Fixes Required

## Summary
The training fails with NaN loss and `sparsities: []` error because the compression metrics are never collected and the learning rate is too aggressive (1.0). This document outlines all changes needed to fix the code.

---

## Issue 1: Empty Sparsity List

### Problem
- **File**: [dl_trainer.py](dl_trainer.py#L610-L613)
- **Lines**: 610-613
- The code attempts to compute `np.mean(self.sparsities)` but `self.sparsities` list is always empty
- This produces NaN which cascades through all training metrics

### Root Cause
The `AllReducer` class in [allreducer.py](allreducer.py) never populates the `self.sparsities`, `self.compression_ratios`, and `self.communication_sizes` lists in `DLTrainer`

### Solution
The `AllReducer` instance needs a reference to the trainer and should populate these lists after each gradient compression.

### Implementation

**File: [allreducer.py](allreducer.py)**

**Location**: Line ~217 in `__init__` method

Add these parameters to AllReducer constructor:
```python
def __init__(self, named_parameters, lock, key_lock, compression, sparse=False, 
             err_callback=None, layerwise_times=None, sigma_scale=2.5, density=0.001, 
             train_epoch=0, norm_clip=None, msg_queue=None, msg_queue2=None, writer=None,
             trainer=None):  # ADD THIS PARAMETER
```

Store trainer reference:
```python
self._trainer = trainer  # ADD THIS LINE after line 233
```

**Location**: Lines ~1603-1606 in `run()` method (after oktopk block ends)

After calculating compression metrics, add this code before `self._allreduce_counter[new_name] += 1`:

```python
                # Track sparsity metrics for trainer
                if self._trainer is not None and self._sparse:
                    if self._compression.name == 'oktopk':
                        # Calculate sparsity (compressed / total)
                        global_topk_size = all_gindexes.size if 'all_gindexes' in locals() else 0
                        tensor_size = torch.numel(new_tensor.data)
                        sparsity = 1.0 - (float(global_topk_size) / float(tensor_size)) if tensor_size > 0 else 0.0
                        compression_ratio = float(tensor_size) / float(global_topk_size) if global_topk_size > 0 else 1.0
                        communication_size = global_topk_size * 2 * 4  # indices (int32) + values (float32)
                        
                        self._trainer.sparsities.append(sparsity)
                        self._trainer.compression_ratios.append(compression_ratio)
                        self._trainer.communication_sizes.append(communication_size)
```

---

## Issue 2: Learning Rate Too High

### Problem
- **File**: [lstm_oktopk_debug_polaris.sh](LSTM/lstm_oktopk_debug_polaris.sh) or training command
- **Value**: `lr=1.0`
- This learning rate is too aggressive for LSTM training, causing gradient explosion

### Solution
Reduce learning rate by 10-100x

### Implementation

**Option A: Modify shell script** [lstm_oktopk_debug_polaris.sh](LSTM/lstm_oktopk_debug_polaris.sh)

Find the line with `--lr 1.0` and change to:
```bash
--lr 0.01    # For conservative training
```
or
```bash
--lr 0.1     # For moderate training
```

**Option B: Modify main_trainer.py** [main_trainer.py](main_trainer.py#L25-L35)

In `argparse` section, change default lr:
```python
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')  # Changed from 1.0
```

---

## Issue 3: Gradient Clipping May Be Inadequate

### Problem
- **File**: [main_trainer.py](main_trainer.py#L95-L97)
- Current implementation uses `torch.nn.utils.clip_grad_norm_` but only after synchronization
- With NaN gradients, clipping may not help

### Solution
Apply gradient clipping before synchronization in distributed_optimizer.py

### Implementation

**File: [distributed_optimizer.py](distributed_optimizer.py)**

**Location**: Before line 129 in the `step()` method

Add gradient clipping:
```python
# Clip gradients before sync
if self._norm_clip is not None:
    torch.nn.utils.clip_grad_norm_(self.param_groups[0]['params'], self._norm_clip)
```

---

## Issue 4: Deprecated PyTorch API

### Problem
- **File**: [distributed_optimizer.py](distributed_optimizer.py#L129)
- **Error**: `add_(weight_decay, p.data)` is deprecated
- Should use: `add_(p.data, alpha=weight_decay)`

### Solution
Update the deprecated API call

### Implementation

**File: [distributed_optimizer.py](distributed_optimizer.py)**

**Location**: Line 129

Change from:
```python
d_p.add_(weight_decay, p.data)
```

To:
```python
d_p.add_(p.data, alpha=weight_decay)
```

---

## Summary of Code Changes

| File | Line(s) | Change | Priority |
|------|---------|--------|----------|
| allreducer.py | 217 | Add `trainer` param to `__init__` | **HIGH** |
| allreducer.py | ~233 | Store `self._trainer = trainer` | **HIGH** |
| allreducer.py | ~1603 | Add sparsity tracking code | **HIGH** |
| distributed_optimizer.py | 26 | Add `norm_clip` parameter pass | **MEDIUM** |
| distributed_optimizer.py | 129 | Fix deprecated `add_()` API | **MEDIUM** |
| lstm_oktopk_debug_polaris.sh | ~lr param | Change `--lr 1.0` to `--lr 0.01` | **HIGH** |
| main_trainer.py | Line with `--lr` default | Change default from `1.0` to `0.01` | **HIGH** |

---

## Testing Recommendations

After implementing changes, run:

1. **Single iteration test**:
   ```bash
   python main_trainer.py --dnn lstman4 --dataset an4 --data_dir ./audio_data --nworkers 1 --max_epochs 1 --batch_size 8 --lr 0.01 --compression True --compressor oktopk
   ```

2. **Check metrics after first epoch**:
   - `sparsities` should NOT be empty
   - `loss` should NOT be NaN
   - `val loss` should be reasonable (not 0.0)

3. **Multi-worker test** (if available):
   ```bash
   mpirun -np 2 python main_trainer.py --dnn lstman4 --dataset an4 --data_dir ./audio_data --nworkers 2 --max_epochs 1 --batch_size 8 --lr 0.01 --compression True --compressor oktopk
   ```

---

## Validation Checklist

- [ ] `AllReducer` accepts `trainer` parameter
- [ ] Sparsity metrics are populated after oktopk compression
- [ ] Learning rate reduced to 0.01
- [ ] Deprecated API warning removed
- [ ] First epoch completes without NaN loss
- [ ] Validation metrics are reasonable (not 0.0 or 1.0)
- [ ] `sparsities` list contains values after first epoch

