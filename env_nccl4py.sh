#!/bin/bash
# Polaris-specific NCCL4Py environment loader
# Source this file in your job scripts to load NCCL4Py dependencies
# Designed for Python 3.10+ with nccl4py installed

# Load required Polaris modules (must come before activating conda env)
module load cuda/12.9 || module load cuda/11.8
module load craype-accel-nvidia80 2>/dev/null || module load craype-accel-nvidia70 2>/dev/null || true
module load cray-pals

# Activate Miniconda and the Python 3.10 conda environment
MINICONDA_DIR="$HOME/miniconda3"
source "$MINICONDA_DIR/etc/profile.d/conda.sh"
conda activate py310_nccl

# Install required packages if missing
pip install psutil --quiet 2>/dev/null || true

# Optional: keep these if you still use mpi4py in the same job (hybrid MPI+NCCL)
export LD_LIBRARY_PATH=/opt/cray/pe/mpich/9.0.1/ofi/nvidia/23.3/lib-abi-mpich:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Debug info
echo "========================================="
echo "Polaris NCCL4Py environment loaded"
echo "========================================="
module list
which python
python --version
python -c "import nccl4py; print('nccl4py OK')" 2>&1 || echo "nccl4py NOT FOUND"
