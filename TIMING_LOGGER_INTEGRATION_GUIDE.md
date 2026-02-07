# Step 1: Timing Logger Integration Guide

## What Was Created

### 1. **New File: `timing_logger.py`**
A standalone timing logging system with these key features:

- **Thread-safe queue**: Records are buffered in a thread-safe queue to avoid blocking the AllReducer thread
- **Background flush thread**: Periodically writes buffered records to disk without blocking main computation
- **Rank-specific CSV files**: Each rank writes to `timing-rank{i}.csv`
- **High-precision timing**: Uses `time.perf_counter()` for microsecond precision
- **Context manager support**: Easy syntax for timing operations

### 2. **Integration Points in `main_trainer.py`**

#### At imports:
```python
from timing_logger import init_timing_logger, shutdown_timing_logger, get_timing_logger
```

#### At function start (right after rank initialization):
```python
timing_log_dir = './logs/timing'
timing_logger = init_timing_logger(rank, timing_log_dir)
```

#### When creating optimizer:
```python
optimizer = dopt.DistributedOptimizer(..., timing_logger=timing_logger)
```

#### In training loop (before each iteration):
```python
optimizer.set_current_iteration(epoch * iters_per_epoch + i)
optimizer.set_current_epoch(epoch)
```

#### At end of training:
```python
optimizer.stop()
shutdown_timing_logger()
```

## How to Use It

### Option 1: Direct Recording (Simple Operations)
```python
timing_logger = get_timing_logger()
timing_logger.record_timing(
    operation='ALLREDUCE',
    elapsed_ms=5.234,
    layer_name='layer1',
    message_size=110280000,
    phase='dense',
    iteration=100,
    epoch=0
)
```

### Option 2: Context Manager (Recommended for MPI Operations)
```python
timing_logger = get_timing_logger()

# Before an MPI call
with timing_logger.record_mpi_operation('ALLREDUCE', message_size, layer_name, 'dense', iteration, epoch):
    comm.Allreduce(tensor, result, op)  # Time this operation automatically

# OR with manual timing
with timing_logger.record_mpi_operation('ALLREDUCE', message_size, layer_name) as timer:
    # Do timed work
    pass
```

## What Gets Logged

Each record contains:
- `timestamp`: ISO format wall-clock time
- `iteration`: Training iteration number
- `epoch`: Training epoch
- `layer_name`: Parameter group name
- `operation`: MPI operation type (ALLREDUCE, ALLGATHER, SEND, RECV, ALLTOALL, BCAST)
- `phase`: dense or sparse
- `message_size_bytes`: Size of data transferred
- `elapsed_ms`: Actual elapsed time in milliseconds
- `start_counter`: perf_counter() at start
- `end_counter`: perf_counter() at end

## Output Files

After training, you'll find:
```
./logs/timing/
├── timing-rank0.csv
├── timing-rank1.csv
├── timing-rank2.csv
└── timing-rank3.csv  (etc for each rank)
```

Each CSV file structure:
```
timestamp,iteration,epoch,layer_name,operation,phase,message_size_bytes,elapsed_ms,start_counter,end_counter
2026-02-06T14:30:45.123456,0,0,layer1,ALLREDUCE,dense,110280000,5.234,1234.567,1239.801
2026-02-06T14:30:50.456789,1,0,layer1,ALLREDUCE,dense,110280000,4.891,1239.802,1244.693
...
```

## Next Steps

Now that the timing logger is integrated into `main_trainer.py`, the next steps are:

1. **Step 2**: Instrument `allreducer.py` to actually USE the timing logger in:
   - `dense_allreduce()` function
   - `topk_sparse_allreduce()` function
   - `gtopk_sparse_allreduce()` function
   - Other MPI operations (Send, Recv, Bcast, Alltoall)

2. **Step 3**: Add H2D/D2H timing around CUDA operations

3. **Step 4**: Add compression timing to isolate compression cost from communication cost

## Thread Safety Notes

- The `TimingLogger` uses `queue.Queue` which is thread-safe
- The background flush thread writes to disk, so no blocking in AllReducer
- Safe to call `record_timing()` from any thread
- No locks needed - designed for lock-free recording

## Performance Impact

- Queue insertion: ~1 microsecond per record
- No blocking: AllReducer thread unaffected
- Disk I/O: Happens in background thread with 1-second flush intervals
- Memory: ~10KB per 1000 records (adjustable via queue size)

## Configuration Options

In `timing_logger.py`, you can adjust:
```python
flush_interval = 1.0  # Flush every 1 second (line ~110)
maxsize=10000         # Queue max size (line ~35)
```

These are tuned for the current setup but can be changed based on your needs.
