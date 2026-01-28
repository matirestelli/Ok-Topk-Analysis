# Quick Copy-Paste Code Fixes

These are the exact code blocks you can copy and paste directly into the files.

---

## Fix #1: allreducer.py Line 217 - Constructor Signature

**Find this line:**
```python
def __init__(self, named_parameters, lock, key_lock, compression, sparse=False, err_callback=None, layerwise_times=None, sigma_scale=2.5, density=0.001, train_epoch=0, norm_clip=None, msg_queue=None, msg_queue2=None, writer=None):
```

**Replace with:**
```python
def __init__(self, named_parameters, lock, key_lock, compression, sparse=False, err_callback=None, layerwise_times=None, sigma_scale=2.5, density=0.001, train_epoch=0, norm_clip=None, msg_queue=None, msg_queue2=None, writer=None, trainer=None):
```

---

## Fix #2: allreducer.py Around Line 233 - Store Trainer Reference

**Find this section:**
```python
        self._norm_clip = norm_clip

        self._allreduce_counter = {}
```

**Add this line between them:**
```python
        self._trainer = trainer
```

**Result should look like:**
```python
        self._norm_clip = norm_clip
        self._trainer = trainer
        
        self._allreduce_counter = {}
```

---

## Fix #3: allreducer.py Line ~1603 - Add Sparsity Tracking

**Find this section:**
```python
                if self._profiling:
                    force_insert_item(self._compression_timers, new_name, compress_t1+compress_t2)

                elif self._sparse and self._compression.name == 'topkAopt':
```

**Replace with:**
```python
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

## Fix #4: distributed_optimizer.py Line ~80 - Add Trainer Parameter

**Find this section (the AllReducer instantiation):**
```python
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
```

**Replace with:**
```python
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
            trainer=None
        )
```

---

## Fix #5: distributed_optimizer.py - Add Setter Method

**Find the class `_DistributedOptimizer` and locate the end of `__init__` method.**

**Add this method after `__init__`:**
```python
    def set_trainer(self, trainer):
        """Set trainer reference for metrics tracking"""
        if self._allreducer is not None:
            self._allreducer._trainer = trainer
```

---

## Fix #6: main_trainer.py Line ~70 - Call Setter

**Find this section in the `robust_ssgd` function:**
```python
    optimizer = dopt.DistributedOptimizer(trainer.optimizer, trainer.net.named_parameters(), compression=compressor, is_sparse=is_sparse, err_handler=_error_handler, layerwise_times=None, sigma_scale=sigma_scale, density=density, norm_clip=norm_clip, writer=writer)

    trainer.update_optimizer(optimizer)
```

**Add this line after `trainer.update_optimizer(optimizer)`:**
```python
    if hasattr(optimizer, 'set_trainer'):
        optimizer.set_trainer(trainer)
```

**Result:**
```python
    optimizer = dopt.DistributedOptimizer(trainer.optimizer, trainer.net.named_parameters(), compression=compressor, is_sparse=is_sparse, err_handler=_error_handler, layerwise_times=None, sigma_scale=sigma_scale, density=density, norm_clip=norm_clip, writer=writer)

    trainer.update_optimizer(optimizer)
    if hasattr(optimizer, 'set_trainer'):
        optimizer.set_trainer(trainer)
```

---

## Fix #7: distributed_optimizer.py Line 129 - Fix Deprecated API

**Find this line:**
```python
                            d_p.add_(weight_decay, p.data)
```

**Replace with:**
```python
                            d_p.add_(p.data, alpha=weight_decay)
```

---

## Fix #8: Learning Rate - Option A (Shell Script)

**File: lstm_oktopk_debug_polaris.sh**

**Find:**
```bash
--lr 1.0
```

**Replace with:**
```bash
--lr 0.01
```

---

## Fix #8: Learning Rate - Option B (main_trainer.py)

**File: main_trainer.py around line 170**

**Find:**
```python
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate')
```

**Replace with:**
```python
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
```

---

## Test After Making Changes

```bash
cd /home/mrest/Ok-Topk-Analysis/LSTM

# Test with single worker
python main_trainer.py --dnn lstman4 --dataset an4 --data_dir ./audio_data --nworkers 1 --max_epochs 1 --batch_size 8 --lr 0.01 --compression True --compressor oktopk

# Watch for:
# 1. No "sparsities: []" warning
# 2. Loss values are NOT nan
# 3. val loss is NOT 0.0
# 4. Sparsity metrics are printed
```

---

## Expected Output After Fixes

```
2026-01-27 22:52:29,703 [dl_trainer.py:612] INFO Average Sparsity: 0.98, compression ratio: 45.2, communication size: 12345
```

Instead of:
```
2026-01-27 22:52:29,703 [dl_trainer.py:612] WARNING NaN detected! sparsities:  []
```

