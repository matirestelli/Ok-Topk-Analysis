# Polaris Environment Setup Guide

This guide explains how to set up and run Ok-Topk experiments on the **Polaris supercomputer** at Argonne Leadership Computing Facility (ALCF).

## Important Differences from Original Setup

The original job scripts were written for **Piz Daint** at CSCS. Polaris has different:
- Module names and versions
- GPU architecture (NVIDIA A100 vs V100)
- MPI implementation (Cray MPICH instead of vendor MPI)
- SLURM partition names and constraints

## One-Time Environment Setup

### Step 1: Run the Setup Script

From the repository root, execute:

```bash
cd /home/mrest/Ok-Topk-Analysis
bash setup_polaris_env.sh
```

This script will:
1. Load the correct Polaris modules (conda, CUDA toolkit)
2. Create a conda environment named `py38_oktopk`
3. Install Python 3.8 and pip 20.2.4
4. Install all requirements from `requirements.txt`
5. Compile and install mpi4py with Cray MPICH
6. Clone and install NVIDIA Apex with CUDA extensions

**Note:** This only needs to be done once, unless you need to rebuild the environment.

### Step 2: Verify Installation

After setup completes, test the environment:

```bash
module load conda/2024-04-29
conda activate py38_oktopk
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from mpi4py import MPI; print(f'MPI version: {MPI.Get_version()}')"
```

## Running Jobs on Polaris

### Important: Job Scripts Need Modification

The existing job scripts (e.g., `lstm_oktopk.sh`, `bert_oktopk.sh`, `vgg16_oktopk.sh`) were written for Piz Daint and need to be updated for Polaris.

### Key Changes Required:

1. **SLURM Headers** - Update partition, constraints, and resource requests:
   ```bash
   #!/bin/bash -l
   #SBATCH --job-name=oktopk_lstm
   #SBATCH --account=YOUR_PROJECT_ACCOUNT  # REQUIRED on Polaris
   #SBATCH --partition=prod                # or 'debug' for testing
   #SBATCH --nodes=32
   #SBATCH --ntasks-per-node=4             # Polaris has 4 GPUs per node
   #SBATCH --cpus-per-task=8
   #SBATCH --time=01:20:00
   #SBATCH --output=logs/%x-%j.out
   ```

2. **Module Loading** - Replace Daint modules:
   ```bash
   # OLD (Piz Daint):
   # module load daint-gpu
   # module load cudatoolkit/10.2.89_3.28-2.1__g52c0314
   
   # NEW (Polaris):
   module load conda/2024-04-29
   module load cudatoolkit-standalone/12.4.1
   module load craype-accel-nvidia80
   conda activate py38_oktopk
   ```

3. **Environment Variables**:
   ```bash
   export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
   
   # For NCCL (if using NCCL communication):
   export NCCL_NET_GDR_LEVEL=PHB
   export NCCL_CROSS_NIC=1
   export NCCL_COLLNET_ENABLE=1
   ```

## Troubleshooting

### MPI Issues
If you see MPI initialization errors:
```bash
# Check MPI is properly linked
ldd $(which python) | grep mpi
# Should show Cray MPICH libraries
```

### CUDA Issues
If CUDA is not detected:
```bash
# Verify CUDA module is loaded
module list
# Check CUDA is accessible
nvcc --version
nvidia-smi
```

### Module Not Found
```bash
# List available module versions
module spider conda
module spider cuda
``