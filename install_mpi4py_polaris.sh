#!/bin/bash
# Install mpi4py on Polaris with Cray MPICH

set -e

echo "=========================================="
echo "Installing mpi4py for Polaris (Cray MPICH)"
echo "=========================================="

# Load Miniconda
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate py38_oktopk

# Load CUDA and GPU accelerator (required for MPI libraries)
echo "Loading CUDA and GPU modules..."
module load cuda/11.8
module load craype-accel-nvidia80

# Verify MPI compiler is available
echo "Checking for MPI compiler..."
which cc || (echo "ERROR: 'cc' MPI compiler wrapper not found"; exit 1)

# Show MPI setup
echo "MPI Compiler Info:"
cc --version 2>&1 | head -5 || true

# Uninstall any existing mpi4py
echo "Removing any existing mpi4py..."
pip uninstall -y mpi4py || true

# Install mpi4py
echo "Building mpi4py with Cray MPICH..."
MPICC="cc -shared" pip install --no-binary=mpi4py -v mpi4py

# Verify
echo ""
echo "Verifying mpi4py installation..."
python -c "from mpi4py import MPI; print(f'✓ mpi4py installed'); print(f'  MPI version: {MPI.Get_version()}')"

echo ""
echo "=========================================="
echo "mpi4py installation successful!"
echo "=========================================="
