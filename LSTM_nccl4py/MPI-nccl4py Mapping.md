# MPI to nccl4py Mapping Guide

## Extracted MPI Collectives & P2P Calls

Only the primitives used in your code. Each includes:
- **MPI usage in your code**
- **nccl4py equivalent**
- **Theory**: how NCCL implements it under the hood
- **GPU differences** vs CPU MPI

---

## 1. MPI_Allreduce (denseallreduce, sparseallreduce)

### Your MPI usage:
```python
comm.Allreduce(tensor, result, op=MPI.SUM)
```
Dense sum across all workers.

### nccl4py equivalent:
```python
stream = nccl4py.get_stream()  # dedicated comm stream
comm.all_reduce(sendbuf=tensor.data_ptr(), recvbuf=result.data_ptr(),
                count=tensor.numel(), datatype='float32', op='sum', stream=stream)
stream.synchronize()
```

### Theory:
- **NCCL AllReduce** = ReduceScatter + AllGather
- Ring/tree/CollNet algorithms for different scales
- GPU‑optimized: pipelined kernels, NVLink/RDMA direct

### GPU differences vs CPU MPI:
- Buffers stay on GPU (no H2D/D2H)
- Async via CUDA streams: overlap with compute
- NVLink provides ~10-100x faster intra-node bandwidth than Ethernet

---

## 2. MPI_Allgather (topksparseallreduce)

### Your MPI usage:
```python
comm.Allgather(values, values1d, numworkers*nnz)  # [nnz] → [numworkers, nnz]
comm.Allgather(indexes, indexes1d, numworkers*nnz)
```

### nccl4py equivalent:
```python
comm.all_gather(sendbuf=local_chunk.data_ptr(), recvbuf=gathered.data_ptr(),
                count=nnz, datatype='float32', stream=stream)
```

### Theory:
- **Double‑ring algorithm**: send ring + recv ring
- Each rank sends its chunk to all others
- Pipelines with GPU memory copies
- Low synchronization overhead at scale

### GPU differences vs CPU MPI:
- NVLink makes intra‑node AllGather **~10x faster** than InfiniBand
- Sparse topk pattern especially benefits from GPU memory bandwidth
- Supports variable-length segments efficiently

---

## 3. MPI_Alltoall (AllReducer.run)

### Your MPI usage:
```python
comm.Alltoall(ssizes, rsizes)  # transpose sparse chunk sizes
```

### nccl4py equivalent:
```python
comm.all_to_all(sendbuf=ssizes.data_ptr(), recvbuf=rsizes.data_ptr(),
                count=numworkers, datatype='int32', stream=stream)
```

### Theory:
- **Double‑binary tree** or pairwise exchange
- Everyone sends distinct chunk to everyone else
- NCCL 2.18+ has dedicated AllToAll kernel (highly optimized)
- Latency-sensitive operation for small messages

### GPU differences vs CPU MPI:
- Your throttling (4x Isend) maps to grouped P2P launches
- GPU can execute multiple sends/recvs in parallel
- Better for sparse communication patterns

---

## 4. MPI_Allgatherv (variable sparse chunks)

### Your MPI usage:
```python
comm.Allgatherv(sendbuffer, recvbuffer, recvsizes, offsets, MPI.FLOAT/INT)
```

### nccl4py equivalent:
```python
comm.all_gather_v(sendbuf=local_sparse.data_ptr(), recvbuf=global_sparse.data_ptr(),
                  recvcounts=recvsizes, displs=offsets, datatype='float32', stream=stream)
```

### Theory:
- **AllGather with irregular sizes**
- Precomputes offsets, uses segmented ring
- Supports different chunk sizes per rank
- More efficient than multiple AllGather calls

### GPU differences vs CPU MPI:
- Efficient for sparse gradients (your topk pattern)
- Avoids padding to uniform sizes
- GPU memory layout optimized for segmented access

---

## 5. MPI_Isend/Irecv + Waitall (throttled AlltoAll)

### Your MPI usage:
```python
reqs = []
reqs.append(comm.Isend(buffer, MPI.FLOAT, dst, tag=1))
reqs.append(comm.Irecv(buffer, MPI.FLOAT, src, tag=1))
MPI.Request.Waitall(reqs)
```

### nccl4py equivalent:
```python
with comm.group(stream=comm_stream):
    comm.send(sendbuf=send_idx.data_ptr(), peer=dst, count=len(send_idx), datatype='uint32')
    comm.recv(recvbuf=recv_idx.data_ptr(), peer=src, count=recv_len, datatype='uint32')
    comm.send(sendbuf=send_val.data_ptr(), peer=dst, count=len(send_val), datatype='float32')
    comm.recv(recvbuf=recv_val.data_ptr(), peer=src, count=recv_len, datatype='float32')
comm_stream.synchronize()
```

### Theory:
- **ncclSend/ncclRecv** = direct GPU‑GPU RDMA
- No host involvement; pure kernel launches
- Grouped calls fuse into single kernel graph
- Enables pipelining and overlapping

### GPU differences vs CPU MPI:
- **Overlap**: launch on `comm_stream`, compute on `compute_stream`
- Zero-copy GPU-to-GPU direct access (NVLink or InfiniBand)
- No PCIe bottleneck (unlike CPU MPI)
- Grouped operations reduce launch overhead

---

## Migration Strategy

### 1. **For Dense Reduction** (entire gradient)
Replace `MPI.Allreduce` → `nccl4py.all_reduce`
- Same semantics, GPU native
- ~1-2x faster due to NVLink

### 2. **For Top-K Sparse**
Replace `MPI.Allgather` + `MPI.Alltoall` → `nccl4py.all_gather_v` + grouped P2P
- Variable-length chunks handled efficiently
- Indices and values separate for flexibility

### 3. **For P2P Throttling**
Replace `Isend/Irecv` loops → `comm.group()` context
- Groups multiple messages
- Single synchronization point
- Better GPU resource utilization

### Key Implementation Notes:
- Always use **dedicated CUDA streams** for communication
- **Synchronize streams** before accessing results in compute kernels
- Use `nccl4py.get_unique_id()` for multi-process initialization
- Polaris: ensure `NCCL_NET_GDR_LEVEL=PHB` for optimal P2P performance
