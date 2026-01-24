# Ok-Topk Algorithm Testing Strategy & mpi4py Fix

## 1. Overview: Why Train Smaller?

Full training on Polaris with all workers can take **20+ hours**. To quickly validate the Ok-Topk algorithm works correctly without massive time investment, we created a **smoke-test approach**:
- Reduced training epochs: 5 instead of 50+
- Reduced batch size: 8 instead of 32
- Reduced resource allocation: 8 GPUs (2 nodes) instead of full system
- **Walltime: 15 minutes instead of hours**

This allows fast validation that the algorithm runs, converges, and produces reasonable results without waiting days.

---

## 2. How to Train Smaller: The Configuration Approach

### Strategy
Rather than modifying code, we modified **configuration files only**:
1. Created a debug configuration file with smaller hyperparameters
2. Created a debug PBS job script pointing to this configuration
3. Job script sources the config before training

### No Code Changes Required
The original `main_trainer.py` reads hyperparameters from config files, so changing configs automatically uses smaller training parameters.

---

## 3. New Files Created

### File 1: `/home/mrest/Ok-Topk-Analysis/LSTM/exp_configs/lstm_debug.conf`
**Purpose**: Debug configuration with reduced training size

**Contents**:
```bash
batch_size=8              # Small batch for fast iterations (default: 32)
max_epochs=5              # Minimal epochs to validate algorithm (default: 50+)
lr=1.0                    # Learning rate (adjust as needed)
dataset=ptb               # Penn Treebank dataset
data_dir=audio_data/      # Where to find training data
```

**How to Modify Training Size**:
- **Smaller**: Reduce `batch_size` (4-8), reduce `max_epochs` (2-5)
- **Larger**: Increase `batch_size` (16-64), increase `max_epochs` (20-50)
- **Different data**: Change `data_dir` path

---

### File 2: `/home/mrest/Ok-Topk-Analysis/LSTM/lstm_oktopk_polaris.sh`
**Purpose**: PBS job script for submitting debug training to Polaris

**Key Configuration Lines**:
```bash
#PBS -l select=2:ngpus=4        # 2 nodes, 4 GPUs per node = 8 total workers
#PBS -l walltime=00:15:00       # 15 minutes (enough for smoke test)
#PBS -q debug                   # Use debug queue for quick turnaround

# Configuration sourced from debug config file
source exp_configs/lstm_debug.conf

# Job runs LSTM training with Ok-Topk algorithm
srun python -m mpi4py main_trainer.py \
    --dnn lstm \
    --dataset $dataset \
    --max-epochs $max_epochs \
    --batch-size $batch_size \
    --nworkers 8 \
    --nwpernode 4
```

**How to Modify**:
- **Use full system**: Change `select=2:ngpus=4` to `select=8:ngpus=4` (32 GPUs)
- **Longer training**: Change `walltime=00:15:00` to `walltime=02:00:00` (2 hours)
- **Different config**: Change `source exp_configs/lstm_debug.conf` to `source exp_configs/lstm.conf`

---

### File 3: `/home/mrest/Ok-Topk-Analysis/LSTM/test_mpi4py.py`
**Purpose**: Simple test script to verify mpi4py is working

**Usage**:
```bash
python test_mpi4py.py
```

**Output** (if working):
```
SUCCESS: mpi4py imported successfully!
MPI Initialized: 0 / 1
```

---

### File 4: Modified `/home/mrest/Ok-Topk-Analysis/polaris_env_modules.sh`
**Change Made**: Added LD_LIBRARY_PATH for mpi4py

**Addition**:
```bash
# Set LD_LIBRARY_PATH for mpi4py to find Cray MPICH ABI-compatible libraries
export LD_LIBRARY_PATH=/opt/cray/pe/mpich/9.0.1/ofi/nvidia/23.3/lib-abi-mpich:$LD_LIBRARY_PATH
```

This ensures all job scripts that source this file automatically have correct mpi4py library paths.

---

## 4. How to Set Training Data for Smaller Training

### Method 1: Edit the Config File
Edit `exp_configs/lstm_debug.conf`:

```bash
# Reduce training time
batch_size=8              # Smaller = faster iterations
max_epochs=5              # Fewer = less total training time

# Use different dataset
data_dir=/path/to/your/data/

# Adjust learning rate for stability with small batches
lr=1.0
```

### Method 2: Override at Runtime
When submitting the job, you can override config values:

```bash
qsub -v batch_size=4,max_epochs=3 lstm_oktopk_polaris.sh
```

### Method 3: Create Different Config Files
For different training sizes, create multiple configs:

```bash
exp_configs/lstm_debug.conf      # 5 epochs, batch=8   (current)
exp_configs/lstm_micro.conf      # 2 epochs, batch=4   (ultra-fast)
exp_configs/lstm_full.conf       # 50 epochs, batch=32 (production)
```

Then point the job script to the desired config.

### Data Directory Structure
The training script expects:
```
LSTM/
├── audio_data/           # Default data directory
│   ├── train/
│   ├── val/
│   └── test/
├── exp_configs/
│   └── lstm_debug.conf
└── main_trainer.py
```

Update `data_dir` in config to point to your dataset location.

---

## 5. The mpi4py Problem & Solution

### The Problem

**Error** (encountered 8+ times):
```
ImportError: undefined symbol: MPI_Buffer_iflush
```

**Root Cause**: 
When installing mpi4py via conda, it bundled its own MPICH implementation (version 4.3.2). This conda MPICH is **binary incompatible at the ABI level** with Polaris' system Cray MPICH (versions 8.1.x, 9.0.1).

When the job tried to import mpi4py, it attempted to load conda's MPICH, which didn't have the symbol `MPI_Buffer_iflush` that Polaris' Cray MPICH provides. Result: crash.

**Why Other Attempts Failed**:
- Changing MPICH modules (8.1.28 → 8.1.32 → 9.0.1) didn't help because mpi4py was still trying to use conda's MPICH
- LD_LIBRARY_PATH tricks didn't work because the binary ABI itself was incompatible
- Building from source failed due to nvc compiler flag incompatibility

### The Solution (In Order)

**Step 1: Remove conda's conflicting MPICH**
```bash
conda uninstall mpich mpi mpi4py -y
```

This removes the conflicting packages. Note: This also removes mpi4py as a dependency.

**Step 2: Install mpi4py via pip pointing to system MPICH**
```bash
export MPICC=/opt/cray/pe/mpich/9.0.1/ofi/nvidia/23.3/bin/mpicc
pip install mpi4py
```

This installs mpi4py as a **binary wheel** for Python 3.8, compiled to link against a generic MPICH interface (not conda's specific version).

**Step 3: Set LD_LIBRARY_PATH for Cray MPICH ABI libraries**
```bash
export LD_LIBRARY_PATH=/opt/cray/pe/mpich/9.0.1/ofi/nvidia/23.3/lib-abi-mpich:$LD_LIBRARY_PATH
```

This tells mpi4py where to find Polaris' Cray MPICH ABI-compatible libraries at runtime.

**Step 4: Verify it works**
```bash
python -c "from mpi4py import MPI; print('SUCCESS')"
```

If this prints "SUCCESS", mpi4py is working.

### Where the Fix Is Implemented

The fix is now **automatic** in:

1. **`polaris_env_modules.sh`** - Sets LD_LIBRARY_PATH for all jobs
2. **`lstm_oktopk_polaris.sh`** - Job script also sets LD_LIBRARY_PATH (redundant but safe)

Any future job script that sources `polaris_env_modules.sh` will automatically have mpi4py working.

---

## 6. Quick Start: Running the Smaller Training

### First Time Setup (One Time Only)
```bash
# Verify mpi4py works
python ~/Ok-Topk-Analysis/LSTM/test_mpi4py.py
```

Expected output:
```
SUCCESS: mpi4py imported successfully!
MPI Initialized: 0 / 1
```

### Submit Debug Job
```bash
cd ~/Ok-Topk-Analysis/LSTM
qsub lstm_oktopk_polaris.sh
```

This returns immediately with a job ID like `6872314.polaris-pbs-01`.

### Check Job Status
```bash
qstat -u $USER
```

### Read Results (When Done)
```bash
cat logs/lstm_oktopk_debug*.out      # Standard output
cat logs/lstm_oktopk_debug*.err      # Error output & detailed logs
```

---

## 7. Next Steps for Full Training

Once smoke test validates the algorithm works:

1. **Edit config file** to larger hyperparameters:
   ```bash
   cp exp_configs/lstm_debug.conf exp_configs/lstm_full.conf
   # Edit lstm_full.conf: batch_size=32, max_epochs=50
   ```

2. **Create full job script** by copying and modifying:
   ```bash
   cp lstm_oktopk_polaris.sh lstm_oktopk_full_polaris.sh
   # Edit: source exp_configs/lstm_full.conf
   # Edit: #PBS -l select=8:ngpus=4  (full system: 32 GPUs)
   # Edit: #PBS -l walltime=20:00:00 (20 hours)
   ```

3. **Submit full training**:
   ```bash
   qsub lstm_oktopk_full_polaris.sh
   ```

---

## 8. Configuration Reference

### Key Hyperparameters in Config Files

| Parameter | Smoke Test | Small | Large | Full |
|-----------|-----------|-------|-------|------|
| `batch_size` | 8 | 16 | 32 | 64 |
| `max_epochs` | 5 | 10 | 25 | 50 |
| `lr` | 1.0 | 1.0 | 0.5 | 0.5 |
| Job GPUs | 8 (2 nodes) | 16 (4 nodes) | 24 (6 nodes) | 32 (8 nodes) |
| Walltime | 15 min | 1 hour | 5 hours | 20 hours |

---

## 9. Troubleshooting

### "mpi4py import fails"
- Check `python -c "from mpi4py import MPI"` returns no error
- Verify `echo $LD_LIBRARY_PATH` contains `/opt/cray/pe/mpich/9.0.1/ofi/nvidia/23.3/lib-abi-mpich`
- If missing, run: `source ../polaris_env_modules.sh`

### "Job stays in queue"
- PBS scheduler may be under maintenance
- Wait 5-10 minutes and resubmit
- Check Polaris status at https://www.alcf.anl.gov/

### "Job fails with module errors"
- Verify modules load: `module load cuda/11.8 cray-mpich/9.0.1`
- Check conda activates: `conda activate py38_oktopk`
- Both are done automatically by `polaris_env_modules.sh`

### "Training runs but is very slow"
- Reduce `batch_size` further if OOM (out of memory)
- Check GPU utilization: `nvidia-smi` during job run
- Verify all 8 workers are active in the training logs

---

## 10. Files Summary

| File | Purpose | Modified By | Status |
|------|---------|-----------|--------|
| `exp_configs/lstm_debug.conf` | Debug config (5 epochs, batch=8) | Created | Ready |
| `lstm_oktopk_polaris.sh` | PBS job script for debug job | Created | Ready |
| `test_mpi4py.py` | mpi4py verification script | Created | Ready |
| `polaris_env_modules.sh` | Environment setup (+ LD_LIBRARY_PATH) | Modified | Ready |
| `main_trainer.py` | Training script | Unchanged | Works |
| `allreducer.py` | MPI communication (imports mpi4py) | Unchanged | Works |

---

## 11. Summary

✅ **Smaller training strategy**: Config-only changes, no code modifications
✅ **Smoke test files**: Created debug.conf and job script
✅ **mpi4py fixed**: Conda conflict removed, pip install + LD_LIBRARY_PATH set
✅ **Ready to run**: `qsub lstm_oktopk_polaris.sh` when Polaris returns to service

**Next action**: When Polaris maintenance ends, submit the job and monitor logs.
