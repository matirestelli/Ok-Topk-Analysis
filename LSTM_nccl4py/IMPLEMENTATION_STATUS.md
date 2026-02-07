# nccl4py Implementation Status for allreducer.py

## ✅ Completed Changes

### 1. Import Statement (Lines 1-14)
- ✅ Added `import nccl4py` with comment "nccl4py mapping: GPU-native collective communications"

### 2. dense_allreduce() Function  
- ✅ Replaced `MPI.Allreduce` with `nccl4py.all_reduce`
- ✅ GPU-resident tensors (PyTorch)
- ✅ Added stream management with `nccl4py.get_stream()`
- ✅ Removed `.Barrier()` call

### 3. topk_sparse_allreduce() Function
- ✅ Replaced 2x `MPI.Allgather` with 2x `nccl4py.all_gather`
- ✅ GPU-resident tensors throughout (no numpy conversion)
- ✅ Uses `torch.topk()` instead of `utils.topk()`
- ✅ Stream-based synchronization

### 4. gtopk_sparse_allreduce() Function  
- ✅ Replaced `MPI.Send`/`MPI.Recv` with `nccl4py.send`/`nccl4py.recv`
- ✅ Replaced `MPI.Bcast` with `nccl4py.broadcast`
- ✅ GPU-resident computation with PyTorch topk
- ✅ Binary tree reduction all on GPU

---

## ⏳ Remaining Implementations

These paths in `AllReducer.run()` need nccl4py mappings:

### oktopk path (Lines ~700-1000)
- Replace `comm.Alltoall(ssizes, rsizes)` → `comm.all_to_all(...)`
- Replace P2P loop with grouped operations (lines ~750-800)
- Replace `comm.Allreduce(region_boundaries, ...)` → `comm.all_reduce(...)`

### topkAopt path (Lines ~1200-1250)  
- Replace `comm.Allgather(send_size, recv_sizes)` → `comm.all_gather(...)`
- Replace 2x `comm.Allgatherv(...)` → 2x `comm.all_gather_v(...)`

### topkSA path (Lines ~1270-1500)
- Same patterns as oktopk + topkAopt combined

### gaussiank/gaussiankconcat/gaussiankSA paths (Lines ~1525-1745)
- Replace `comm.Allgather(send_size, recv_sizes)` → `comm.all_gather(...)`
- Replace `comm.Allgatherv(...)` → `comm.all_gather_v(...)`

---

## Implementation Pattern

For each MPI collective, use this pattern:

```python
# nccl4py mapping: <MPI_CALL> -> <nccl4py_call>
stream = nccl4py.get_stream()

# Convert to GPU tensors (keep on GPU, don't use numpy)
tensor_gpu = torch.from_numpy(tensor).cuda() if isinstance(tensor, np.ndarray) else tensor

# Call nccl4py collective
comm.<nccl4py_function>(sendbuf=tensor_gpu.data_ptr(),
                       recvbuf=result_gpu.data_ptr(),
                       count=tensor_gpu.numel(),
                       datatype='float32',
                       op='sum',  # for reductions
                       stream=stream)

stream.synchronize()

# Return GPU tensor (no conversion back to numpy for GPU code path)
return tensor_gpu
```

---

## Key Principles

1. **GPU-resident tensors**: Keep everything as PyTorch tensors on GPU
2. **Stream management**: Use `nccl4py.get_stream()` and `.synchronize()`
3. **No numpy roundtrips**: Avoid CPU↔GPU conversions for sparse tensors
4. **Comments**: Add "nccl4py mapping: <MPI_CALL>" comments for clarity
5. **Datatype mapping**:
   - `MPI.FLOAT` → `'float32'`
   - `MPI.INT` → `'int32'`
   - `MPI.DOUBLE` → `'float64'`

---

## Testing

After implementation:
1. Run with `CONDA_ENV=py310_nccl source env_nccl4py.sh`
2. Test single iteration to validate communication
3. Check NCCL environment: `NCCL_DEBUG=INFO`
4. Monitor GPU utilization during AllReduce operations

