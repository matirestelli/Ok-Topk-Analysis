#!/bin/bash
# Polaris-specific environment module loader
# Source this file in your job scripts or call directly before running training
# This sets up modules and activates the conda environment created by setup_polaris_env.sh

# Load required Polaris modules (must come before activating conda env)
module load cuda/11.8 || module load cuda/12.9
module load craype-accel-nvidia80 2>/dev/null || module load craype-accel-nvidia70 2>/dev/null || true
module load cray-pals
# Activate Miniconda and the conda environment
# (created by setup_polaris_env.sh)
MINICONDA_DIR="$HOME/miniconda3"
source "$MINICONDA_DIR/etc/profile.d/conda.sh"
conda activate py38_oktopk

# Install required packages if missing
pip install psutil --quiet 2>/dev/null || true

# Set LD_LIBRARY_PATH for mpi4py to find Cray MPICH ABI-compatible libraries
export LD_LIBRARY_PATH=/opt/cray/pe/mpich/9.0.1/ofi/nvidia/23.3/lib-abi-mpich:$LD_LIBRARY_PATH

# Set OpenMP threads (adjust based on your job requirements)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Optional NCCL optimizations for Polaris
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1

# Workaround for tensorboardX/protobuf incompatibility (if needed)
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Print environment info for debugging
echo "========================================="
echo "Polaris environment loaded"
echo "========================================="
echo "Modules loaded:"
module list
echo ""
echo "Python: $(which python)"
echo "Python version: $(python --version 2>&1)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>&1 || echo 'NOT FOUND')"
echo "NVCC: $(which nvcc 2>&1 || echo 'NOT FOUND')"
echo "CUDA version: $(nvcc --version 2>&1 | grep -i release || echo 'NOT DETECTED')"
echo "mpi4py: $(python -c 'from mpi4py import MPI; print(\"OK\")' 2>&1 || echo 'NOT FOUND')"
echo "========================================="
echo ""
