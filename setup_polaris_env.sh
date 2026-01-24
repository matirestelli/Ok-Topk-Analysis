#!/bin/bash
# Environment setup script for Polaris supercomputer (ALCF)
# This script creates a conda environment with all required dependencies
# Single unified installation script - everything in one place

set -e

LOGFILE="setup_polaris_env.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=========================================="
echo "Setting up environment for Polaris"
echo "Log: $LOGFILE"
echo "=========================================="

# Variables
ENV_NAME="py38_oktopk"
MINICONDA_DIR="$HOME/miniconda3"
FORCE=${FORCE:-0}   # set FORCE=1 to remove existing env non-interactively

echo "Detecting available modules..."

# Try to load a system conda module if available
echo "Attempting to load a system 'conda' module (if present)"
set +e
module load conda 2>/dev/null
CONDA_OK=$?
set -e
if [ $CONDA_OK -eq 0 ] && command -v conda >/dev/null 2>&1; then
    echo "System conda module loaded and 'conda' is available"
else
    echo "No usable 'conda' module found or 'conda' not in PATH. Will fallback to Miniconda in $MINICONDA_DIR"
    if [ -x "$MINICONDA_DIR/bin/conda" ]; then
        echo "Found existing Miniconda at $MINICONDA_DIR"
        source "$MINICONDA_DIR/etc/profile.d/conda.sh"
    else
        echo "Miniconda not found. Installing Miniconda to $MINICONDA_DIR"
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
        bash /tmp/miniconda.sh -b -p "$MINICONDA_DIR"
        rm -f /tmp/miniconda.sh
        source "$MINICONDA_DIR/etc/profile.d/conda.sh"
        conda config --set auto_activate_base false || true
    fi
fi

# Load CUDA module first (required before loading craype-accel-nvidia)
echo "Loading CUDA module (required for GPU support)"
set +e
module load cuda/11.8 2>/dev/null && echo "Loaded cuda/11.8" || (module load cuda/12.9 2>/dev/null && echo "Loaded cuda/12.9") || echo "WARNING: Could not load cuda module"
set -e

# Try to find a suitable craype-accel module for NVIDIA GPUs and load it if present
echo "Looking for available 'craype-accel' NVIDIA modules"
ACCEL_CHOICE=""
if module avail craype-accel 2>&1 | grep -q craype-accel-nvidia80; then
    ACCEL_CHOICE=craype-accel-nvidia80
elif module avail craype-accel 2>&1 | grep -q craype-accel-nvidia70; then
    ACCEL_CHOICE=craype-accel-nvidia70
fi
if [ -n "$ACCEL_CHOICE" ]; then
    echo "Loading $ACCEL_CHOICE"
    module load "$ACCEL_CHOICE"
else
    echo "No craype-accel-nvidia* module auto-detected; you may need to load an accelerator module in your job script"
fi

# === ENVIRONMENT CREATION ===
echo ""
echo "Checking for existing conda environment: $ENV_NAME"
if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
    if [ "$FORCE" = "1" ]; then
        echo "Removing existing environment $ENV_NAME (FORCE=1)"
        conda env remove -n ${ENV_NAME} || true
    else
        echo "Environment ${ENV_NAME} already exists. Remove it? (y/n)"
        read -r response
        if [[ "$response" == "y" ]]; then
            conda env remove -n ${ENV_NAME}
        else
            echo "Keeping existing environment. Exiting setup."
            exit 0
        fi
    fi
fi

echo "Creating conda environment: ${ENV_NAME} (Python 3.8)"
# Accept TOS before creating environment to avoid interactive prompt
$MINICONDA_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
$MINICONDA_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
conda create -n ${ENV_NAME} python=3.8 -y

# === ENVIRONMENT ACTIVATION ===
echo "Activating environment"
source "$MINICONDA_DIR/etc/profile.d/conda.sh" || true
conda activate ${ENV_NAME}

# Accept conda TOS to avoid interactive prompts
echo "Accepting conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# === DEPENDENCIES INSTALLATION ===
echo ""
echo "=== Installing Python dependencies ==="

echo "Step 1: Installing pip and build tools..."
pip install --upgrade pip setuptools wheel

echo "Step 2: Installing requirements from requirements.txt..."
if [ -f requirements.txt ]; then
    echo "Checking if PyTorch is already installed..."
    if python -c "import torch; print(f'PyTorch {torch.__version__} found')" 2>/dev/null; then
        echo "PyTorch already installed, skipping requirements.txt to avoid reinstall"
    else
        echo "Installing requirements..."
        pip install -r requirements.txt
    fi
else
    echo "WARNING: requirements.txt not found in current directory; continuing"
fi

echo "Step 3: Installing NVIDIA Apex..."
if python -c "import apex; print(f'Apex installed')" 2>/dev/null; then
    echo "✓ Apex already installed, skipping"
else
    echo "Building and installing Apex (this may take several minutes)..."
    if [ ! -d "apex" ]; then
        echo "  - Cloning Apex repository..."
        git clone https://github.com/NVIDIA/apex || echo "  WARNING: git clone failed"
    fi
    if [ -d "apex" ]; then
        cd apex
        echo "  - Installing Apex with CUDA extensions..."
        pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ || echo "  WARNING: Apex installation may have failed"
        cd ..
        if python -c "import apex; print('✓ Apex verified!')" 2>/dev/null; then
            echo "✓ Apex installed successfully!"
        fi
    else
        echo "WARNING: Apex directory not found; skipping Apex install"
    fi
fi

# === SUMMARY ===
echo ""
echo "=========================================="
echo "Environment setup script finished!"
echo "=========================================="
echo ""
echo "Setup log saved to: $LOGFILE"
echo ""
echo "To activate this environment, use:"
echo "  source $MINICONDA_DIR/etc/profile.d/conda.sh"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "In your SLURM job scripts, load modules and activate:"
echo "  module load cuda/11.8"
echo "  module load $ACCEL_CHOICE"
echo "  source $MINICONDA_DIR/etc/profile.d/conda.sh"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To verify the installation:"
echo "  source $MINICONDA_DIR/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"
echo "  python -c \"import torch; print(f'PyTorch: {torch.__version__}')\" "
echo "  python -c \"from mpi4py import MPI; print(f'mpi4py: OK')\" "
echo "  python -c \"import apex; print(f'Apex: OK')\" "
echo ""
echo "To recreate the environment from scratch:"
echo "  FORCE=1 bash setup_polaris_env.sh"
echo ""
