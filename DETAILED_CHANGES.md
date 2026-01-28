# Detailed Code Change Locations

## Change #1: Add Trainer Reference to AllReducer

### File: `/home/mrest/Ok-Topk-Analysis/LSTM/allreducer.py`

### Location A: Line ~217 (Constructor signature)

```python
# BEFORE (current code around line 217)
def __init__(self, named_parameters, lock, key_lock, compression, sparse=False, err_callback=None, layerwise_times=None, sigma_scale=2.5, density=0.001, train_epoch=0, norm_clip=None, msg_queue=None, msg_queue2=None, writer=None):

# AFTER (modified)
def __init__(self, named_parameters, lock, key_lock, compression, sparse=False, err_callback=None, layerwise_times=None, sigma_scale=2.5, density=0.001, train_epoch=0, norm_clip=None, msg_queue=None, msg_queue2=None, writer=None, trainer=None):
    # ... rest of init code ...
    self._writer = writer
    self._trainer = trainer  # ADD THIS LINE
```

### Location B: Line ~1603-1606 (After oktopk allreduce, before counter increment)

```python
# CURRENT CODE (lines 1603-1606)
                if self._profiling:
                    force_insert_item(self._compression_timers, new_name, compress_t1+compress_t2)

                elif self._sparse and self._compression.name == 'topkAopt':

# MODIFIED CODE - INSERT THIS BLOCK BEFORE THE "elif" STATEMENT:
                if self._profiling:
                    force_insert_item(self._compression_timers, new_name, compress_t1+compress_t2)

                # Track compression metrics for the trainer
                if self._trainer is not None and self._sparse and self._compression.name == 'oktopk':
                    try:
                        # Only count actual global topk elements sent
                        global_topk_count = all_gindexes.size if 'all_gindexes' in locals() and all_gindexes is not None else 0
                        tensor_size = torch.numel(new_tensor.data)
                        
                        if tensor_size > 0 and global_topk_count > 0:
                            sparsity = 1.0 - (float(global_topk_count) / float(tensor_size))
                            compression_ratio = float(tensor_size) / float(global_topk_count)
                            communication_size = global_topk_count * 2 * 4  # Each value is float32, each index is int32
                            
                            self._trainer.sparsities.append(sparsity)
                            self._trainer.compression_ratios.append(compression_ratio)
                            self._trainer.communication_sizes.append(communication_size)
                    except Exception as e:
                        if rank == 0:
                            logger.warning('Failed to track compression metrics: %s', str(e))

                elif self._sparse and self._compression.name == 'topkAopt':
```

---

## Change #2: Pass Trainer to AllReducer Constructor

### File: `/home/mrest/Ok-Topk-Analysis/LSTM/distributed_optimizer.py`

### Location: Find where AllReducer is instantiated (~line 70-90)

```python
# BEFORE: Find the line that creates AllReducer
self._allreducer = ar.AllReducer(
    named_parameters=named_parameters,
    lock=self._lock,
    key_lock=self._key_lock,
    compression=compressor,
    sparse=is_sparse,
    err_callback=err_handler,
    layerwise_times=layerwise_times,
    sigma_scale=sigma_scale,
    density=density,
    norm_clip=norm_clip,
    msg_queue=self._msg_queue,
    msg_queue2=self._msg_queue2,
    writer=writer
)

# AFTER: Add trainer=None parameter (will be set later)
self._allreducer = ar.AllReducer(
    named_parameters=named_parameters,
    lock=self._lock,
    key_lock=self._key_lock,
    compression=compressor,
    sparse=is_sparse,
    err_callback=err_handler,
    layerwise_times=layerwise_times,
    sigma_scale=sigma_scale,
    density=density,
    norm_clip=norm_clip,
    msg_queue=self._msg_queue,
    msg_queue2=self._msg_queue2,
    writer=writer,
    trainer=None  # ADD THIS LINE
)
```

### Then add a setter method in DistributedOptimizer class:

```python
def set_trainer(self, trainer):
    """Set trainer reference for metrics tracking"""
    self._allreducer._trainer = trainer
```

### And in dl_trainer.py, after creating the optimizer, call:

Find in `main_trainer.py` around line 66-68:
```python
optimizer = dopt.DistributedOptimizer(trainer.optimizer, trainer.net.named_parameters(), 
                                       compression=compressor, is_sparse=is_sparse, 
                                       err_handler=_error_handler, layerwise_times=None, 
                                       sigma_scale=sigma_scale, density=density, 
                                       norm_clip=norm_clip, writer=writer)

trainer.update_optimizer(optimizer)

# ADD THIS LINE:
optimizer.set_trainer(trainer)
```

---

## Change #3: Fix Deprecated PyTorch API

### File: `/home/mrest/Ok-Topk-Analysis/LSTM/distributed_optimizer.py`

### Location: Line 129

```python
# BEFORE
d_p.add_(weight_decay, p.data)

# AFTER
d_p.add_(p.data, alpha=weight_decay)
```

---

## Change #4: Reduce Learning Rate

### Option A: Edit Shell Script
### File: `/home/mrest/Ok-Topk-Analysis/LSTM/lstm_oktopk_debug_polaris.sh`

Find the line with `--lr` and change:
```bash
# BEFORE
--lr 1.0

# AFTER  
--lr 0.01
```

### Option B: Edit main_trainer.py
### File: `/home/mrest/Ok-Topk-Analysis/LSTM/main_trainer.py`

Find argparse section (around line 170):
```python
# BEFORE
parser.add_argument('--lr', type=float, default=1.0, help='learning rate')

# AFTER
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
```

---

## Change #5: Optional - Better Gradient Clipping

### File: `/home/mrest/Ok-Topk-Analysis/LSTM/distributed_optimizer.py`

### Location: In step() method around line 125-135

```python
# BEFORE
for group in self.param_groups:
    weight_decay = group['weight_decay']
    for p in group['params']:
        if p.grad is None:
            continue
        d_p = p.grad.data
        if weight_decay != 0:
            d_p.add_(p.data, alpha=weight_decay)

# AFTER - Add gradient clipping before weight decay
for group in self.param_groups:
    weight_decay = group['weight_decay']
    for p in group['params']:
        if p.grad is None:
            continue
        d_p = p.grad.data
        
        # Apply gradient clipping if enabled
        if self._norm_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(p, self._norm_clip)
        
        if weight_decay != 0:
            d_p.add_(p.data, alpha=weight_decay)
```

---

## Summary: Line-by-Line Changes

1. **allreducer.py:217** - Add `trainer=None` to function signature
2. **allreducer.py:~233** - Add `self._trainer = trainer` 
3. **allreducer.py:~1603** - Add sparsity tracking block
4. **distributed_optimizer.py:~80** - Add `trainer=None` to AllReducer call
5. **distributed_optimizer.py:~90** - Add setter method `set_trainer()`
6. **main_trainer.py:~70** - Add `optimizer.set_trainer(trainer)`
7. **distributed_optimizer.py:129** - Fix deprecated API call
8. **lstm_oktopk_debug_polaris.sh** OR **main_trainer.py:170** - Change `--lr 1.0` to `--lr 0.01`

