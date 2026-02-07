# allreducer.py - MPI to nccl4py Mapping

This document shows all MPI calls found in `allreducer.py` with their nccl4py equivalents.

## MPI Calls Found in allreducer.py

### 1. **dense_allreduce()** - Line ~305

**Current MPI code:**
```python
def dense_allreduce(comm, tensor):
    result = np.zeros_like(tensor)
    op = MPI.SUM
    comm.Allreduce(tensor, result, op)  # <-- MPI_Allreduce
    comm.Barrier()
    return result
```

**nccl4py equivalent:**
```python
def dense_allreduce_nccl(comm, tensor):
    import nccl4py
    
    # Tensor is already GPU-resident PyTorch tensor in practice
    # If input is numpy (edge case), caller should convert before calling
    assert tensor.is_cuda, "Tensor must be on GPU for nccl4py"
    
    result = torch.zeros_like(tensor)
    stream = nccl4py.get_stream()
    
    # nccl4py AllReduce
    comm.all_reduce(sendbuf=tensor.data_ptr(), 
                   recvbuf=result.data_ptr(),
                   count=tensor.numel(), 
                   datatype='float32', 
                   op='sum', 
                   stream=stream)
    stream.synchronize()
    
    return result
```

**Status:** ✅ **Direct mapping** - Assumes GPU-resident input (standard for training loops)

---

### 2. **topk_sparse_allreduce()** - Lines ~25-46

**Current MPI code:**
```python
def topk_sparse_allreduce(comm, sparse_tensor, storage, indexes=None, dtype=np.float32):
    # ... setup ...
    comm.Allgather(values, values_1d[:num_workers*nnz])     # <-- MPI_Allgather
    comm.Allgather(indexes, indexes_1d[:num_workers*nnz])    # <-- MPI_Allgather
    return values_1d, indexes_1d, None
```

**nccl4py equivalent:**
```python
def topk_sparse_allreduce_nccl(comm, sparse_tensor, storage, indexes=None, dtype=torch.float32):
    """
    Keep everything on GPU - no numpy conversion needed!
    Input: GPU tensors (PyTorch)
    Output: GPU tensors (PyTorch)
    """
    import nccl4py
    import torch
    
    tensor = sparse_tensor
    if indexes is None:
        k = int(tensor.numel() * 0.01)
        _, indexes = torch.topk(torch.abs(tensor), k)
        values = tensor[indexes]
    else:
        k = len(indexes)
        values = tensor[indexes]
    
    num_workers = comm.size
    nnz = k
    
    # Allocate GPU buffers directly (no numpy!)
    values_gathered = torch.zeros(k * num_workers, dtype=dtype, device=tensor.device)
    indexes_gathered = torch.zeros(k * num_workers, dtype=torch.int32, device=tensor.device)
    
    stream = nccl4py.get_stream()
    
    # nccl4py AllGather - values
    comm.all_gather(sendbuf=values.data_ptr(), 
                   recvbuf=values_gathered.data_ptr(),
                   count=nnz, 
                   datatype='float32', 
                   stream=stream)
    
    # nccl4py AllGather - indexes
    comm.all_gather(sendbuf=indexes.data_ptr(), 
                   recvbuf=indexes_gathered.data_ptr(),
                   count=nnz, 
                   datatype='int32', 
                   stream=stream)
    
    stream.synchronize()
    
    # Return GPU tensors directly - no conversion back to numpy!
    return values_gathered, indexes_gathered, None
```

**Status:** ✅ **Direct mapping** - GPU-resident end-to-end, zero CPU conversion

---

### 3. **gtopk_sparse_allreduce()** - Lines ~48-138

**Current MPI code:**
```python
def gtopk_sparse_allreduce(comm, sparse_tensor, storage=None, indexes=None, dtype=np.float32):
    # ... setup ...
    for i in range(num_round):
        if rank in participate_ranks:
            local_rank = participate_ranks.index(rank)
            if local_rank % 2 == 0:
                source = participate_ranks[local_rank+1]
                comm.Recv([recv_values, MPI.FLOAT], source=source)  # <-- MPI_Recv
                # ... computation ...
            else:
                target = participate_ranks[local_rank-1]
                comm.Send([send_values, MPI.FLOAT], dest=target)  # <-- MPI_Send
    # ... more computation ...
    comm.Bcast(send_values, root=0)  # <-- MPI_Bcast
    # ...
```

**nccl4py equivalent:**
```python
def gtopk_sparse_allreduce_nccl(comm, sparse_tensor, storage=None, indexes=None, dtype=torch.float32):
    """
    Keep everything on GPU - no numpy conversions!
    Binary tree reduction for sparse tensors.
    """
    import nccl4py
    import torch
    
    num_workers = comm.size
    rank = comm.rank
    tensor = sparse_tensor
    
    if indexes is None:
        k = int(tensor.numel() * 0.001)
        _, indexes = torch.topk(torch.abs(tensor), k)
        values = tensor[indexes]
    else:
        k = len(indexes)
        values = tensor[indexes]
    
    original_indexes = indexes
    
    # GPU tensors for send/recv (concatenate indexes + values)
    send_data = torch.cat([indexes.float(), values])  # combined GPU tensor
    recv_data = torch.zeros_like(send_data)
    
    stream = nccl4py.get_stream()
    
    num_round = int(np.log2(num_workers))
    step = 1
    participate_ranks = list(range(0, num_workers, step))
    
    for i in range(num_round):
        if rank in participate_ranks:
            local_rank = participate_ranks.index(rank)
            if local_rank % 2 == 0:
                source = participate_ranks[local_rank + 1]
                
                # nccl4py P2P Recv
                comm.recv(recvbuf=recv_data.data_ptr(),
                         peer=source,
                         count=len(recv_data),
                         datatype='float32',
                         stream=stream)
                stream.synchronize()
                
                # GPU-based topk merging (all operations on GPU)
                recv_indexes = recv_data[:k].long()
                recv_values = recv_data[k:2*k]
                
                # Merge: combine local and received values
                all_indexes = torch.cat([indexes, recv_indexes])
                all_values = torch.cat([values, recv_values])
                
                # GPU topk to select top-k from merged
                _, topk_idx = torch.topk(torch.abs(all_values), min(k, len(all_values)))
                indexes = all_indexes[topk_idx]
                values = all_values[topk_idx]
                
                # Update send_data for next round
                send_data = torch.cat([indexes.float(), values])
            else:
                target = participate_ranks[local_rank - 1]
                
                # nccl4py P2P Send
                comm.send(sendbuf=send_data.data_ptr(),
                         peer=target,
                         count=len(send_data),
                         datatype='float32',
                         stream=stream)
                stream.synchronize()
        
        step *= 2
        participate_ranks = list(range(0, num_workers, step))
    
    # Final broadcast from rank 0 (GPU tensor)
    if rank == 0:
        final_data = torch.cat([indexes.float(), values])
    else:
        final_data = torch.zeros(2*k, dtype=dtype, device=tensor.device)
    
    comm.broadcast(sendbuf=final_data.data_ptr() if rank == 0 else None,
                  recvbuf=final_data.data_ptr() if rank != 0 else None,
                  count=len(final_data),
                  root=0,
                  datatype='float32',
                  stream=stream)
    stream.synchronize()
    
    # Extract final results (still GPU tensors)
    if rank != 0:
        indexes = final_data[:k].long()
        values = final_data[k:2*k]
    
    # GPU intersection for included_indexes
    _, c1, c2 = torch.sort(torch.cat([original_indexes, indexes]).unique(return_inverse=True))
    included_indexes = c1[:len(original_indexes)]
    
    # Return GPU tensors - no conversion!
    return values, indexes, included_indexes
```

**Status:** ✅ **Direct mapping** - GPU-resident end-to-end
- All topk/merge operations on GPU
- No numpy roundtrips
- Efficient GPU memory access patterns

---

### 4. **AllReducer.run()** - oktopk path (Lines ~363-745)

**Current MPI code:**
```python
# Line ~460: Alltoall for transposing sizes
comm.Alltoall(ssizes, rsizes)  # <-- MPI_Alltoall

# Lines ~483-506: P2P Send/Recv with throttling
reqs = []
for i in range(0, throttle):
    dst = dsts[i]
    src = srcs[i]
    # ... buffer setup ...
    reqs.append(comm.Isend(...))  # <-- MPI_Isend
    reqs.append(comm.Irecv(...))  # <-- MPI_Irecv
MPI.Request.Waitall(reqs)

# Lines ~544-588: Allreduce for boundaries
comm.Allreduce(region_boundaries, global_boundaries, MPI.SUM)  # <-- MPI_Allreduce
```

**nccl4py equivalent:**
```python
# 4a. Alltoall mapping
stream = nccl4py.get_stream()

ssizes_gpu = torch.from_numpy(ssizes).cuda().int()
rsizes_gpu = torch.from_numpy(rsizes).cuda().int()

comm.all_to_all(sendbuf=ssizes_gpu.data_ptr(),
               recvbuf=rsizes_gpu.data_ptr(),
               count=num_workers,
               datatype='int32',
               stream=stream)
stream.synchronize()

rsizes = rsizes_gpu.cpu().numpy()


# 4b. P2P Send/Recv with grouping (instead of throttled Isend/Irecv)
comm_stream = nccl4py.get_stream()
compute_stream = torch.cuda.current_stream()

throttle = min(4, num_workers)
for i in range(0, throttle):
    dst = dsts[i]
    src = srcs[i]
    
    # Get buffer pointers
    send_idx_gpu = torch.from_numpy(all_index_sbuffers[i]).cuda()
    send_val_gpu = torch.from_numpy(all_value_sbuffers[i]).cuda()
    recv_idx_gpu = torch.zeros(recv_sizes[src], dtype=torch.int32).cuda()
    recv_val_gpu = torch.zeros(recv_sizes[src], dtype=torch.float32).cuda()
    
    with comm_stream.group():
        comm.send(sendbuf=send_idx_gpu.data_ptr(),
                 peer=dst,
                 count=len(send_idx_gpu),
                 datatype='int32',
                 stream=comm_stream)
        comm.recv(recvbuf=recv_idx_gpu.data_ptr(),
                 peer=src,
                 count=recv_sizes[src],
                 datatype='int32',
                 stream=comm_stream)
        comm.send(sendbuf=send_val_gpu.data_ptr(),
                 peer=dst,
                 count=len(send_val_gpu),
                 datatype='float32',
                 stream=comm_stream)
        comm.recv(recvbuf=recv_val_gpu.data_ptr(),
                 peer=src,
                 count=recv_sizes[src],
                 datatype='float32',
                 stream=comm_stream)

comm_stream.synchronize()


# 4c. Allreduce mapping for boundaries
region_boundaries_gpu = torch.from_numpy(region_boundaries).cuda().int()
global_boundaries_gpu = torch.zeros_like(region_boundaries_gpu)

comm.all_reduce(sendbuf=region_boundaries_gpu.data_ptr(),
               recvbuf=global_boundaries_gpu.data_ptr(),
               count=len(region_boundaries),
               datatype='int32',
               op='sum',
               stream=stream)
stream.synchronize()

global_boundaries = global_boundaries_gpu.cpu().numpy()
global_boundaries //= num_workers
```

**Status:** ✅ **Direct mapping** (with stream management)

---

### 5. **AllReducer.run()** - topkAopt path (Lines ~748-798)

**Current MPI code:**
```python
comm.Allgather(send_size, recv_sizes)  # <-- MPI_Allgather (metadata)

# Lines ~777-780
comm.Allgatherv(kindexes, [all_indexes, recv_sizes, offsets, MPI.INT])      # <-- MPI_Allgatherv
comm.Allgatherv(kvalues, [all_values, recv_sizes, offsets, MPI.FLOAT])      # <-- MPI_Allgatherv
```

**nccl4py equivalent:**
```python
# Allgather metadata (sizes)
send_size_gpu = torch.from_numpy(send_size).cuda().int()
recv_sizes_gpu = torch.from_numpy(recv_sizes).cuda().int()

comm.all_gather(sendbuf=send_size_gpu.data_ptr(),
               recvbuf=recv_sizes_gpu.data_ptr(),
               count=1,
               datatype='int32',
               stream=stream)
stream.synchronize()

recv_sizes = recv_sizes_gpu.cpu().numpy()

# Allgatherv for variable-length chunks
offsets_gpu = torch.from_numpy(offsets).cuda().int()
kindexes_gpu = torch.from_numpy(kindexes).cuda().int()
all_indexes_gpu = torch.zeros(total_size, dtype=torch.int32).cuda()

comm.all_gather_v(sendbuf=kindexes_gpu.data_ptr(),
                 recvbuf=all_indexes_gpu.data_ptr(),
                 recvcounts=recv_sizes_gpu.data_ptr(),
                 displs=offsets_gpu.data_ptr(),
                 datatype='int32',
                 stream=stream)

# Same for values
kvalues_gpu = torch.from_numpy(kvalues).cuda()
all_values_gpu = torch.zeros(total_size, dtype=torch.float32).cuda()

comm.all_gather_v(sendbuf=kvalues_gpu.data_ptr(),
                 recvbuf=all_values_gpu.data_ptr(),
                 recvcounts=recv_sizes_gpu.data_ptr(),
                 displs=offsets_gpu.data_ptr(),
                 datatype='float32',
                 stream=stream)

stream.synchronize()

all_indexes = all_indexes_gpu.cpu().numpy()
all_values = all_values_gpu.cpu().numpy()
```

**Status:** ✅ **Direct mapping**

---

### 6. **AllReducer.run()** - topkSA path (Lines ~820-981)

**Current MPI code:**
```python
comm.Alltoall(ssizes, rsizes)  # <-- MPI_Alltoall

# Lines ~890-901: P2P with throttling (same as oktopk)
reqs = []
for i in range(0, throttle):
    # ... Isend/Irecv ...
MPI.Request.Waitall(reqs)

# Lines ~952-963: Final sparse gather
comm.Allgather(send_size, recv_sizes)       # <-- MPI_Allgather
comm.Allgatherv(gindexes, [..., MPI.INT])   # <-- MPI_Allgatherv
comm.Allgatherv(gvalues, [..., MPI.FLOAT])  # <-- MPI_Allgatherv
```

**nccl4py equivalent:** (Same patterns as #4 and #5 above)

**Status:** ✅ **Direct mapping** (reuse patterns from oktopk and topkAopt)

---

### 7. **AllReducer.run()** - gaussiank path (Lines ~1014-1068)

**Current MPI code:**
```python
comm.Allgather(send_size, recv_sizes)       # <-- MPI_Allgather
comm.Allgatherv(local_topk_indexes, [...])  # <-- MPI_Allgatherv
comm.Allgatherv(local_topk_values, [...])   # <-- MPI_Allgatherv
```

**nccl4py equivalent:** (Same as topkAopt #5)

**Status:** ✅ **Direct mapping**

---

### 8. **AllReducer.run()** - gaussiankconcat path (Lines ~1070-1100)

**Current MPI code:**
```python
comm.Allgather(send_size, recv_sizes)       # <-- MPI_Allgather
comm.Allgatherv(send_buffer, [...])         # <-- MPI_Allgatherv (single combined buffer)
```

**nccl4py equivalent:** (Same as topkAopt #5, but with combined buffer)

**Status:** ✅ **Direct mapping**

---

### 9. **AllReducer.run()** - gaussiankSA path (Lines ~1100-1208)

**Current MPI code:**
```python
comm.Allgather(send_size, recv_sizes)       # <-- MPI_Allgather
comm.Allgatherv(send_buffer, [...])         # <-- MPI_Allgatherv (combined indexes + values)
```

**nccl4py equivalent:** (Same as gaussiankconcat)

**Status:** ✅ **Direct mapping**

---

## Summary Table

| Function/Path | MPI Call | Type | nccl4py Equivalent | Status |
|---|---|---|---|---|
| dense_allreduce | Allreduce | Collective | all_reduce | ✅ |
| topk_sparse_allreduce | Allgather (2x) | Collective | all_gather (2x) | ✅ |
| gtopk_sparse_allreduce | Send/Recv, Bcast | P2P + Collective | send, recv, broadcast | ✅ |
| oktopk path | Alltoall, Isend/Irecv, Allreduce | Mixed | all_to_all, grouped P2P, all_reduce | ✅ |
| topkAopt path | Allgather, Allgatherv (2x) | Collective | all_gather, all_gather_v (2x) | ✅ |
| topkSA path | Alltoall, Isend/Irecv, Allgather, Allgatherv | Mixed | all_to_all, grouped P2P, all_gather, all_gather_v | ✅ |
| gaussiank path | Allgather, Allgatherv (2x) | Collective | all_gather, all_gather_v (2x) | ✅ |
| gaussiankconcat path | Allgather, Allgatherv | Collective | all_gather, all_gather_v | ✅ |
| gaussiankSA path | Allgather, Allgatherv | Collective | all_gather, all_gather_v | ✅ |

---

## Critical Notes for Review

1. **GPU Memory Transfer**: All paths require converting numpy arrays ↔ GPU tensors
2. **Stream Management**: Use dedicated `comm_stream` for communication, separate `compute_stream` for compute
3. **Synchronization**: Must call `.synchronize()` after each nccl4py collective
4. **gtopk_sparse_allreduce**: Requires GPU-resident topk() and intersect1d() (use cupy or PyTorch equivalents)
5. **Throttling P2P**: Replace `Isend/Irecv` loops with `comm.group()` context manager
6. **Datatype Mapping**: 
   - `MPI.FLOAT` → `'float32'`
   - `MPI.INT` → `'int32'`
   - `MPI.DOUBLE` → `'float64'`

