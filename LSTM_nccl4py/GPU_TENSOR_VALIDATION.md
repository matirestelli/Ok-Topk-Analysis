# GPU Tensor Validation Checklist

## 1. Simple `.is_cuda` Checks

Add assertions at function entry/exit points:

```python
def topk_sparse_allreduce(comm, sparse_tensor, storage, indexes=None, dtype=torch.float32):
    # Validate input is on GPU
    assert sparse_tensor.is_cuda, f"sparse_tensor must be on GPU, got device={sparse_tensor.device}"
    assert indexes.is_cuda, f"indexes must be on GPU, got device={indexes.device}"
    
    # ... function body ...
    
    # Validate outputs are on GPU
    assert values_gathered.is_cuda, f"Output must be on GPU, got {values_gathered.device}"
    assert indexes_gathered.is_cuda, f"Output must be on GPU, got {indexes_gathered.device}"
    
    return values_gathered, indexes_gathered, None
```

---

## 2. Device Placement Logging

Add debug logging to track tensor locations throughout execution:

```python
def dense_allreduce(comm, tensor):
    # Log device placement
    logger.debug(f"dense_allreduce INPUT: device={tensor.device}, shape={tensor.shape}, is_cuda={tensor.is_cuda}")
    
    result = torch.zeros_like(tensor)
    stream = nccl4py.get_stream()
    
    comm.all_reduce(sendbuf=tensor.data_ptr(), 
                   recvbuf=result.data_ptr(),
                   count=tensor.numel(), 
                   datatype='float32', 
                   op='sum', 
                   stream=stream)
    stream.synchronize()
    
    # Log output device
    logger.debug(f"dense_allreduce OUTPUT: device={result.device}, shape={result.shape}, is_cuda={result.is_cuda}")
    
    return result
```

---

## 3. GPU Memory Usage Tracking

Monitor GPU memory to detect unexpected CPU→GPU transfers:

```python
def topk_sparse_allreduce_with_memory_check(comm, sparse_tensor, storage, indexes=None, dtype=torch.float32):
    # Record initial GPU memory
    torch.cuda.reset_peak_memory_stats()
    initial_mem = torch.cuda.memory_allocated()
    logger.info(f"Initial GPU memory: {initial_mem / 1e9:.2f} GB")
    
    tensor = sparse_tensor
    k = int(tensor.numel() * 0.01)
    _, indexes = torch.topk(torch.abs(tensor), k)
    values = tensor[indexes]
    
    # Check memory after topk (should still be on GPU)
    after_topk = torch.cuda.memory_allocated()
    logger.info(f"After topk GPU memory: {after_topk / 1e9:.2f} GB (delta: {(after_topk - initial_mem) / 1e9:.2f} GB)")
    
    num_workers = comm.size
    nnz = k
    
    values_gathered = torch.zeros(k * num_workers, dtype=dtype, device=tensor.device)
    indexes_gathered = torch.zeros(k * num_workers, dtype=torch.int32, device=tensor.device)
    
    after_alloc = torch.cuda.memory_allocated()
    logger.info(f"After buffer allocation: {after_alloc / 1e9:.2f} GB (delta: {(after_alloc - after_topk) / 1e9:.2f} GB)")
    
    stream = nccl4py.get_stream()
    
    # Collectives
    comm.all_gather(sendbuf=values.data_ptr(), 
                   recvbuf=values_gathered.data_ptr(),
                   count=nnz, 
                   datatype='float32', 
                   stream=stream)
    
    comm.all_gather(sendbuf=indexes.data_ptr(), 
                   recvbuf=indexes_gathered.data_ptr(),
                   count=nnz, 
                   datatype='int32', 
                   stream=stream)
    
    stream.synchronize()
    
    after_collective = torch.cuda.memory_allocated()
    peak_mem = torch.cuda.max_memory_allocated()
    logger.info(f"After collective: {after_collective / 1e9:.2f} GB, Peak: {peak_mem / 1e9:.2f} GB")
    
    return values_gathered, indexes_gathered, None
```

---

## 4. Type Validation Helper

Create a utility function to validate tensor states:

```python
def validate_gpu_tensor(tensor, name="tensor"):
    """Validate that tensor is GPU-resident and properly formatted."""
    checks = {
        f"{name}.is_cuda": tensor.is_cuda,
        f"{name}.dtype": tensor.dtype,
        f"{name}.device": tensor.device,
        f"{name}.shape": tensor.shape,
        f"{name}.is_contiguous": tensor.is_contiguous(),
    }
    
    for check_name, value in checks.items():
        if "is_cuda" in check_name and not value:
            raise RuntimeError(f"❌ {check_name} = {value} (MUST be True)")
        logger.debug(f"✓ {check_name} = {value}")
    
    return True

# Usage in functions
def topk_sparse_allreduce(comm, sparse_tensor, ...):
    validate_gpu_tensor(sparse_tensor, "sparse_tensor")
    validate_gpu_tensor(indexes, "indexes")
    
    # ... function body ...
    
    validate_gpu_tensor(values_gathered, "values_gathered")
    return values_gathered, indexes_gathered, None
```

---

## 5. Enable NCCL Debug Logging

Run with environment variables to see NCCL activity:

```bash
# Before running training:
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Or for more detail:
export NCCL_DEBUG=TRACE
```

This will show in logs:
- Which devices are used
- Collective operation details
- Memory movements

---

## 6. Pytest for GPU Tensor Validation

Create a simple test file:

```python
# test_gpu_tensors.py
import torch
import pytest
from allreducer import dense_allreduce, topk_sparse_allreduce

def test_dense_allreduce_stays_on_gpu():
    """Verify dense_allreduce keeps tensors on GPU."""
    tensor = torch.randn(1000, 1000, device='cuda', requires_grad=True)
    assert tensor.is_cuda, "Input should be on GPU"
    
    result = dense_allreduce(comm, tensor)
    
    assert result.is_cuda, f"Output should be on GPU, got {result.device}"
    assert result.shape == tensor.shape
    print(f"✓ Output device: {result.device}")

def test_topk_sparse_allreduce_stays_on_gpu():
    """Verify topk_sparse_allreduce keeps tensors on GPU."""
    tensor = torch.randn(10000, device='cuda')
    assert tensor.is_cuda, "Input should be on GPU"
    
    values, indexes, _ = topk_sparse_allreduce(comm, tensor, {})
    
    assert values.is_cuda, f"Values should be on GPU, got {values.device}"
    assert indexes.is_cuda, f"Indexes should be on GPU, got {indexes.device}"
    print(f"✓ Values device: {values.device}")
    print(f"✓ Indexes device: {indexes.device}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
```

Run with:
```bash
pytest test_gpu_tensors.py -v -s
```

---

## 7. Quick Inline Checks

Add these one-liners throughout code:

```python
# Check input
assert sparse_tensor.is_cuda, f"Expected GPU tensor, got {sparse_tensor.device}"

# After allocation
values_gathered = torch.zeros(..., device=tensor.device)  # Always specify device!
assert values_gathered.device == tensor.device

# Before communication
assert indexes.is_contiguous(), "Non-contiguous tensor may cause issues"

# After collective
assert values_gathered.is_cuda, "Output should remain on GPU"
```

---

## 8. Summary Checklist

For each function, verify:

- ✅ **Input**: `assert input_tensor.is_cuda`
- ✅ **Allocation**: `torch.zeros(..., device=tensor.device)`  ← Always specify device
- ✅ **No `.cpu()`**: Never call `.cpu()` in the hot path
- ✅ **No `.numpy()`**: Never convert to numpy (except for logging metadata)
- ✅ **Output**: `assert output.is_cuda` before return
- ✅ **Stream sync**: `stream.synchronize()` after nccl4py calls
- ✅ **Memory**: GPU memory should grow with batch size, not peak unexpectedly

---

## Common Issues to Watch For

| Issue | Sign | Fix |
|-------|------|-----|
| CPU tensor input | `AssertionError: tensor must be on GPU` | Call `.cuda()` on input before calling function |
| Unspecified device | `RuntimeError: expected all tensors on same device` | Add `device=tensor.device` to all allocations |
| CPU conversion | GPU memory stays high, then drops | Remove `.cpu()`, `.numpy()` calls |
| Non-contiguous | NCCL errors or silent failures | Add `.contiguous()` before `.data_ptr()` |
| Stream leak | Memory grows over iterations | Ensure `stream.synchronize()` is called |

---

## Example: Complete Validation Pattern

```python
def topk_sparse_allreduce_validated(comm, sparse_tensor, storage, indexes=None, dtype=torch.float32):
    # VALIDATE INPUT
    assert sparse_tensor.is_cuda, f"Input must be GPU tensor, got {sparse_tensor.device}"
    assert sparse_tensor.is_contiguous(), "Input must be contiguous"
    logger.info(f"INPUT: {sparse_tensor.device}, shape={sparse_tensor.shape}")
    
    # EXTRACT TOPK (stays on GPU)
    tensor = sparse_tensor
    k = int(tensor.numel() * 0.01)
    _, indexes = torch.topk(torch.abs(tensor), k)
    values = tensor[indexes]
    assert values.is_cuda and indexes.is_cuda
    
    # ALLOCATE ON SAME GPU
    num_workers = comm.size
    values_gathered = torch.zeros(k * num_workers, dtype=dtype, device=tensor.device)
    indexes_gathered = torch.zeros(k * num_workers, dtype=torch.int32, device=tensor.device)
    assert values_gathered.device == tensor.device
    assert indexes_gathered.device == tensor.device
    
    # COMMUNICATE
    stream = nccl4py.get_stream()
    comm.all_gather(sendbuf=values.data_ptr(), 
                   recvbuf=values_gathered.data_ptr(),
                   count=k, datatype='float32', stream=stream)
    comm.all_gather(sendbuf=indexes.data_ptr(), 
                   recvbuf=indexes_gathered.data_ptr(),
                   count=k, datatype='int32', stream=stream)
    stream.synchronize()
    
    # VALIDATE OUTPUT
    assert values_gathered.is_cuda, f"Output must be GPU, got {values_gathered.device}"
    assert indexes_gathered.is_cuda, f"Output must be GPU, got {indexes_gathered.device}"
    logger.info(f"OUTPUT: {values_gathered.device}, shape={values_gathered.shape}")
    
    return values_gathered, indexes_gathered, None
```

This gives you full confidence everything stays on GPU! 🚀

