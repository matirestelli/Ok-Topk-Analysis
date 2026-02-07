#!/bin/bash -l
#PBS -l select=1:ngpus=4
#PBS -l walltime=00:05:00
#PBS -l filesystems=home:grand
#PBS -q debug-scaling
#PBS -A UIC-HPC
#PBS -N gpu_diagnostics
#PBS -o /home/mrest/Ok-Topk-Analysis/LSTM/logs/gpu_diagnostics_${PBS_JOBID}.out
#PBS -e /home/mrest/Ok-Topk-Analysis/LSTM/logs/gpu_diagnostics_${PBS_JOBID}.err

cd ${PBS_O_WORKDIR}
mkdir -p logs

echo "=========================================="
echo "GPU Diagnostics on Compute Node"
echo "Job started at: $(date)"
echo "=========================================="
echo ""

# Source the environment to get PyTorch
source ../polaris_env_modules.sh

echo "=== NVIDIA-SMI Full Output ==="
nvidia-smi
echo ""

echo "=== GPU List ==="
nvidia-smi --list-gpus
echo ""

echo "=== CUDA Info ==="
nvidia-smi -q | grep -A 5 "CUDA"
echo ""

echo "=== GPU Memory Info ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv
echo ""

echo "=== GPU Processes ==="
nvidia-smi pmon -c 1
echo ""

echo "=== GPU Clock Info ==="
nvidia-smi --query-gpu=index,clocks.current.graphics,clocks.current.memory --format=csv
echo ""

echo "=== Test CUDA Device Access ==="
PY=$HOME/miniconda3/envs/py38_oktopk/bin/python
$PY -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
print('CUDA Device Count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    try:
        print(f'Device {i}: {torch.cuda.get_device_name(i)}')
        torch.cuda.set_device(i)
        print(f'  - Set device {i}: SUCCESS')
        t = torch.zeros(10, device=f'cuda:{i}')
        print(f'  - Allocate tensor on device {i}: SUCCESS')
    except Exception as e:
        print(f'  - ERROR: {e}')
"
echo ""

echo "=== MIG Status ==="
nvidia-smi -L
echo ""

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
