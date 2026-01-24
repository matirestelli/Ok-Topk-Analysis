# Ok-Topk Polaris Setup - Complete Summary

## ✅ Environment Status

Your Polaris environment is now **ready to use**!

### Installed Components
- **Miniconda**: `~/miniconda3`
- **Conda Environment**: `py38_oktopk` (Python 3.8)
- **PyTorch**: 2.2.0 (CUDA 12.1) ✅
- **Apex**: ✅
- **Requirements**: All from `requirements.txt` ✅
- **mpi4py**: To be built on compute nodes (see below)

### Key Files
- `setup_polaris_env.sh` - Full environment setup (run once)
- `polaris_env_modules.sh` - Load modules + activate env (source in job scripts)
- `LSTM/lstm_oktopk_polaris.sh` - Example Polaris job script

## Quick Start

### Activate the Environment
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py38_oktopk
```

Or use the helper script:
```bash
source polaris_env_modules.sh
```

### Build mpi4py on a Compute Node

mpi4py cannot be built on login nodes due to Cray compiler limitations. Build it on a compute node:

```bash
# Interactive allocation (for testing)
qsub -I -l select=1 -l walltime=00:30:00

# Then in the compute node:
module load cuda/11.8
module load craype-accel-nvidia80
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py38_oktopk
MPICC="cc -shared" pip install --no-binary=mpi4py mpi4py
```

Or include in your job script:
```bash
srun -n 1 bash -c 'module load cuda/11.8 craype-accel-nvidia80 && MPICC="cc -shared" pip install --no-binary=mpi4py mpi4py'
```

## Running Jobs

### Basic Job Script Template

```bash
#!/bin/bash -l
#SBATCH --job-name=oktopk_test
#SBATCH --account=YOUR_PROJECT_ACCOUNT
#SBATCH --partition=prod          # or 'debug' for testing
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1       # 1 GPU per node (modify for 4 GPUs)
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/job_%j.out

mkdir -p logs

# Load Polaris environment
source polaris_env_modules.sh

# Your training command
cd ~/Ok-Topk-Analysis/LSTM
srun python -m mpi4py main_trainer.py \
    --dnn lstman4 \
    --dataset an4 \
    --max-epochs 1 \
    --batch-size 32 \
    --nworkers 2 \
    --data-dir ./audio_data \
    --nwpernode 1 \
    --compression \
    --density 0.02 \
    --compressor oktopk
```

## Module Versions (Polaris)

- **CUDA**: `cuda/11.8`
- **GPU Accelerator**: `craype-accel-nvidia80` (A100 GPUs)
- **Python**: 3.8
- **PyTorch**: 2.2.0 (updated from 1.6.0 for Polaris compatibility)
- **Apex**: Latest (with CUDA extensions)

## Troubleshooting

### PyTorch Import Error
If you get `ImportError: numpy.core.multiarray failed to import`:
```bash
pip install "numpy<1.20" --force-reinstall
```

### Module Not Found
Ensure you've sourced the environment:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py38_oktopk
```

### mpi4py Issues
- **On login node**: Expected to fail - build on compute node only
- **In job scripts**: Load modules and use `MPICC="cc -shared"` when building

### GPU Not Available
Check that modules are loaded:
```bash
module list
nvidia-smi
```

## Testing the Setup

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py38_oktopk

# Test PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__} on CUDA {torch.version.cuda}')"

# Test Apex
python -c "import apex; print('Apex OK')"

# Test requirements
python -c "import librosa, numpy, scipy, sklearn; print('All dependencies OK')"
```

## Notes

1. **PyTorch Version**: Updated to 2.2.0 (from 1.6.0) for better Polaris compatibility
2. **warpctc-pytorch**: Has PyPI packaging issues - skipped (only needed for LSTM acoustic models)
3. **mpi4py**: Must be built on compute nodes due to Cray compiler wrapper limitations
4. **Requirements.txt**: Kept as reference but updated in this setup for Polaris

## Next Steps

1. Run a test job to verify everything works:
   ```bash
   sbatch LSTM/lstm_oktopk_polaris.sh
   ```

2. Check logs:
   ```bash
   tail -f logs/job_*.out
   ```

3. For larger experiments, adjust `--nodes`, `--ntasks-per-node`, and dataset paths as needed.

## References

- Polaris User Guide: https://docs.alcf.anl.gov/polaris/
- Ok-Topk Paper: https://doi.org/10.1145/3503221.3508399
- Setup Guide: POLARIS_SETUP_GUIDE.md
