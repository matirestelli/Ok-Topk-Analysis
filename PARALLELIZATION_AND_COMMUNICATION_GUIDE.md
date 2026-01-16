# Parallelization and Worker Communication Guide

This guide explains how the Ok-Topk system parallelizes work across multiple workers and how they communicate during distributed training.

## Quick Reference: Key Files for Worker Communication

Each file has a specific role in coordinating worker processes:

| File | What It Represents | Key Responsibility |
|------|-------------------|-------------------|
| **`*.sh`** scripts | Job launcher | Spawns multiple MPI processes (workers) via SLURM |
| **`main_trainer.py`** | Worker initialization | Each MPI process runs this; sets up MPI communicator, assigns GPU to process, broadcasts initial model |
| **`distributed_optimizer.py`** | Gradient interceptor | Runs in each worker; hooks into backpropagation to catch gradients and queue them for communication |
| **`allreducer.py`** | Communication engine | Background thread in each worker; performs actual MPI send/receive operations to synchronize gradients |
| **`compression.py`** | Sparsification logic | Computes which gradient elements to send (top-k selection, thresholding, residual tracking) |

**To understand worker-to-worker communication:** Focus on **`allreducer.py`** - this is where MPI processes exchange data.

**Simple mental model:**
1. Each worker = One MPI process = One Python instance running `main_trainer.py`
2. Worker computes gradients → **`distributed_optimizer.py`** catches them
3. Background thread in **`allreducer.py`** exchanges sparse gradients between workers using MPI
4. After synchronization, worker updates its model and continues training

## Overall Architecture

The system uses **data parallelism** with **sparse gradient allreduce** for distributed training. 

### What is a "Worker"?

Each **worker is an MPI process**. Specifically:

- **One worker = One MPI process = One Python process**
- Each process runs the same `main_trainer.py` script
- Each process gets a unique **rank** (ID) from MPI: 0, 1, 2, ..., n-1
- Each process is typically assigned to one GPU
- All processes run simultaneously and communicate via MPI

**Example from `lstm_oktopk.sh`:**
```bash
#SBATCH --nodes=32          # Request 32 compute nodes
#SBATCH --ntasks=32         # Launch 32 total MPI processes (workers)
**What happens:**
- **SLURM** (`srun`) launches `nworkers` separate Python processes across the allocated nodes
- Each process imports `mpi4py` which automatically calls `MPI_Init()` to join the MPI environment
- Each MPI process gets a unique **rank** (ID from 0 to nworkers-1) via `MPI.COMM_WORLD.rank`
- All processes execute the same code but behave differently based on their rank
```

This launches **32 MPI processes** (32 workers), one per node, each running on its own GPU.

### How Workers Operate

Each worker:
1. Trains on a different subset of data (different mini-batches)
2. Computes gradients locally on its own GPU
3. Communicates sparse gradients to synchronize model updates
4. Uses MPI (Message Passing Interface) for inter-worker communication

## Key Files for Understanding Parallelization

### 1. **Launch and Initialization** 
**File:** `LSTM/lstm_oktopk.sh` (and similar `.sh` files)

```bash
srun $PY -m mpi4py main_trainer.py --dnn $dnn --dataset $dataset \
  --nworkers $nworkers --batch-size $batch_size
```

- Uses **SLURM** (`srun`) to launch multiple processes across nodes
- Uses **MPI4PY** to initialize MPI communication
- Each worker is assigned a unique **rank** (ID from 0 to nworkers-1)

### 2. **Training Coordinator**
**File:** [LSTM/main_trainer.py](LSTM/main_trainer.py)

**Key aspects:**
- **Lines 12-13:** Initialize MPI communicator
  ```python
  from mpi4py import MPI
  comm = MPI.COMM_WORLD  # Global communicator for all workers
  ```MPI process (worker) sets its GPU based on its rank
  ```python
  torch.cuda.set_device(dopt.rank() % nwpernode)
  # If rank=0 and nwpernode=1: uses GPU 0
  # If rank=1 and nwpernode=1: uses GPU 1
  # Each process has its own CUDA context on a separate GPUed on rank
  ```python
  torch.cuda.set_device(dopt.rank() % nwpernode)
  ```

- **Lines 49-53:** Model parameters are broadcast from rank 0 to all workers
  ```python
  model_state = comm.bcast(trainer.net.state_dict(), root=0)
  trainer.net.load_state_dict(model_state)
  comm.Barrier()  # Synchronization point
  ```

- **Line 65:** Creates distributed optimizer that handles gradient communication
  ```python
  optimizer = dopt.DistributedOptimizer(trainer.optimizer, 
                                        trainer.net.named_parameters(),
                                        compression=compressor, ...)
  ```

### 3. **Distributed Optimizer** 
**File:** [LSTM/distributed_optimizer.py](LSTM/distributed_optimizer.py)

**Key mechanism:**

- **Lines 66-72:** Registers hooks on all model parameters to intercept gradients
  ```python
  def _register_hooks(self):
      for param_group in self.param_groups:
          for p in param_group['params']:
              if p.requires_grad:
                  grad_acc.register_hook(self._make_hook(p))
  ```

- **Lines 74-93:** Hook function that triggers when gradients are computed
  ```python
  def _make_hook(self, p):
      def hook(*ignore):
          # Gradient is ready for parameter p
          self._handles[p] = self._allreducer.add_tensor(name, d_p)
          self._msg_queue.put(name)  # Signal allreducer thread
      return hook
  ```

- **Lines 59-61:** Background thread for allreduce operations
  ```python
  self.allreducer_thread = threading.Thread(name='allreducer', 
                                             target=self._allreducer.run)
  self.allreducer_thread.start()
  ```

**Communication Pattern:**
- Main training thread computes gradients
- Hooks intercept gradients and queue them
- Background thread performs communication asynchronously

### 4. **Allreduce Communication Core**
**File:** [LSTM/allreducer.py](LSTM/allreducer.py)

This is where the **actual inter-worker communication** happens.

#### **Initialization (Lines 196-260):**

```python
class AllReducer():
    def __init__(self, ...):
        self._comm = MPI.COMM_WORLD  # MPI communicator
        self._comm.Set_errhandler(MPI.ERRORS_RETURN)
        
        # Set up communication pattern
        dsts = list(range(self._comm.size))  # Destination ranks
        srcs = dsts[::-1]                     # Source ranks
        dsts = list_rotate(dsts, -self._comm.rank)
        srcs = list_rotate(srcs, self._comm.rank+1)
```

**Key variables:**
- `self._comm.size` - Total number of workers
- `self._comm.rank` - This worker's ID (0 to n-1)
- `self._dsts` - Destination workers for sending data
- `self._srcs` - Source workers for receiving data

#### **Main Communication Loop (Lines 554-700+):**

```python
def run(self):
    while self._running:
        name = self._msg_queue.get()  # Wait for gradient from hook
        tensor = self._entries[name]
        
        # Perform sparse allreduce
        if self._compression.name == 'oktopk':
            # Ok-Topk algorithm (detailed below)
```

## Ok-Topk Communication Protocol (Lines 580-850)

The Ok-Topk algorithm is the main contribution. Here's how workers communicate:

### **Phase 1: Local Compression** (Lines 595-610)

Each worker independently:
1. Computes a local threshold based on gradient magnitude
2. Selects top-k elements above the threshold
3. Maintains residuals (unselected gradients) for next iteration

```python
local_threshold = self._compression.ratio2threshold(
    tensor=new_tensor, ratio=density)
indexes, values = self._compression.compressbythreshold(
    tensor=split_tensors[i], thres=local_threshold)
```

### **Phase 2: Region Partitioning** (Lines 640-670)

The gradient tensor is divided into **regions** (one per worker):

1. Each worker computes local top-k indexes
2. Divide indexes into equal chunks (one per worker)
3. Use boundaries to partition the gradient space
4. Workers use **Allreduce** to agree on global boundaries

```python
# Every worker finds boundaries in their local top-k
index_chunk = local_topk_indexes.size // num_workers
index_boundaries = np.zeros(num_workers)
for i in range(num_workers):
    index_boundaries[i] = index_chunk * i

# Allreduce to get global boundaries
comm.Allreduce(region_boundaries, global_boundaries, MPI.SUM)
global_boundaries //= num_workers
```

**Result:** Gradient space is partitioned, and each worker is responsible for one region.

### **Phase 3: All-to-All Exchange** (Lines 672-720)

Workers exchange sparse gradients using **personalized communication**:

```python
# Split tensor according to regions
split_tensors = torch.split(new_tensor, boundaries)

# Each worker compresses their portion for each region
for i in range(num_workers):
    indexes, values = self._compression.compressbythreshold(
        tensor=split_tensors[i], thres=local_threshold)
    all_index_sbuffers.append(send_index_buffer)
    all_value_sbuffers.append(send_value_buffer)
```

**Communication primitive:** Point-to-point (Isend/Irecv)

```python
# Exchange buffer sizes first
comm.Alltoall(ssizes, rsizes)

# Then exchange actual sparse data
for j in range(throttle):
    dst = dsts[j]
    src = srcs[j]
    comm.Isend([all_index_sbuffers[dst], MPI.INT], dest=dst)
    comm.Irecv([all_index_rbuffers[j], MPI.INT], source=src)
    comm.Isend([all_value_sbuffers[dst], MPI.FLOAT], dest=dst)
    comm.Irecv([all_value_rbuffers[j], MPI.FLOAT], source=src)
```

**Key optimization:** Throttling communication into chunks to overlap computation and communication.

### **Phase 4: Local Aggregation** (Lines 720-800)

Each worker aggregates received sparse gradients for their assigned region:

```python
for k in range(inner_chunk_offset.size):
    reduced_t[tmp_indexes[inner_chunk_offset[k]:
                          inner_chunk_offset[k]+inner_chunk_size[k]]] += \
             tmp_values[inner_chunk_offset[k]:
                        inner_chunk_offset[k]+inner_chunk_size[k]]
```

### **Phase 5: Global Top-K Selection** (Lines 800-850)

Periodically (every `global_threshold_recompute_interval` iterations):

1. Each worker identifies non-zero elements in their region
2. **Allgatherv** collects all non-zero gradients globally
3. Compute global top-k from all aggregated values
4. Broadcast result back to reconstruct full gradient

```python
# Gather all non-zero gradients
comm.Allgatherv(send_buffer, [recv_buffer, recv_sizes, offsets, MPI.FLOAT])

# Compute global top-k
gtopk_values, gtopk_values_indexes, global_threshold = \
    self._compression.k2globalthreshold(all_gvalues_tensor, topk_value)

# Update result tensor
result[gtopk_gindexes_tensor] = gtopk_values / num_workers
```

## Communication Patterns Summary

| Algorithm | Communication Primitives | Volume | Key Features |
|-----------|-------------------------|---------|--------------|
| **Dense** | `Allreduce` | O(d) | Standard data-parallel training |
| **TopkA** | `Allgather` | O(nk) | Each worker sends top-k, all receive all |
| **gTopk** | `Send/Recv` (tree) | O(k log n) | Tree-based reduction |
| **Ok-Topk** | `Alltoall` + `Allgatherv` | **O(k)** | Region-based partitioning (**optimal**) |

Where:
- `d` = model dimension (total parameters)
- `n` = number of workers
- `k` = sparsity parameter (k << d)

## Key Communication Operations

### **MPI Operations Used:**

1. **`Bcast`** (Line 49, main_trainer.py)
   - One-to-all: Broadcasts initial model from rank 0

2. **`Allreduce`** (Line 667, allreducer.py)
   - All-to-all: Computes global sum/average
   - Used for: Dense gradients, boundary calculation

3. **`Alltoall`** (Line 704, allreducer.py)
   - All-to-all with personalized messages
   - Used for: Exchanging buffer sizes in Ok-Topk

4. **`Isend/Irecv`** (Lines 730-750, allreducer.py)
   - Non-blocking point-to-point
   - Used for: Sparse gradient exchange
   - Allows computation-communication overlap

5. **`Allgatherv`** (Line 819, allreducer.py)
   - All-to-all with variable-size messages
   - Used for: Global top-k selection

6. **`Barrier`** (Line 53, main_trainer.py)
   - Synchronization point
   - Ensures all workers finish before proceeding

## How to Trace Communication Flow

To understand the full communication flow, follow this path:

1. **Start:** [LSTM/main_trainer.py](LSTM/main_trainer.py) - Line 82 (`trainer.train(1)`)
   - Forward pass, backward pass → gradients computed

2. **Hook triggered:** [LSTM/distributed_optimizer.py](LSTM/distributed_optimizer.py) - Line 74
   - `_make_hook()` intercepts gradient
   - Queues gradient for communication

3. **Background thread:** [LSTM/allreducer.py](LSTM/allreducer.py) - Line 554 (`run()`)
   - Receives queued gradient
   - Performs compression and communication

4. **Ok-Topk algorithm:** [LSTM/allreducer.py](LSTM/allreducer.py) - Lines 580-850
   - Local compression → Region partition → All-to-all → Aggregation → Global top-k

5. **Gradient update:** [LSTM/main_trainer.py](LSTM/main_trainer.py) - Line 96 (`trainer.update_model()`)
   - Apply synchronized gradients to model

## Debugging and Monitoring

**Useful variables to inspect:**

- `comm.rank` - Which worker am I?
- `comm.size` - Total number of workers
- `ssizes` - Amount of data each worker sends
- `rsizes` - Amount of data each worker receives
- `local_topk_indexes.size` - Local sparsity
- `gtopk_gindexes.size` - Global sparsity

**Profiling outputs:**
- Lines 709-710: Prints local top-k statistics
- Lines 848-849: Prints global top-k statistics
Process Model:**
- **Each worker = One MPI process = One running Python instance**
- Processes run independently but coordinate via MPI messages
- Each process has its own memory space and GPU

**Data Parallelism:**
- Each MPI process (worker) gets different mini-batch from dataset
- All processes maintain a full copy of the model
- Gradients are synchronized across processes
- Each worker gets different mini-batch
- All workers have full model copy
- Gradients synchronized via sparse allreduce

**Communication Strategy:**
- **Asynchronous:** Background thread for communication
- **Sparse:** Only top-k gradients communicated
- **Partitioned:** Gradient space divided among workers
- **Overlapped:** Computation and communication pipelined

**Ok-Topk Advantage:**
- Optimal communication volume: O(k) instead of O(nk)
- Scales well with number of workers
- Maintains convergence through error feedback (residuals)
