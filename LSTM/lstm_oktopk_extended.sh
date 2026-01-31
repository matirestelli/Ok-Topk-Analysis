#!/bin/bash -l
#PBS -l select=2:ngpus=4
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:grand
#PBS -q debug-scaling
#PBS -A UIC-HPC
#PBS -N lstm_oktopk_extended
#PBS -o /home/mrest/Ok-Topk-Analysis/LSTM/logs/lstm_oktopk_extended_${PBS_JOBID}.out
#PBS -e /home/mrest/Ok-Topk-Analysis/LSTM/logs/lstm_oktopk_extended_${PBS_JOBID}.err

# Change to the directory where the job was submitted
cd ${PBS_O_WORKDIR}

# Create logs directory if it doesn't exist
mkdir -p logs

# Source the Polaris environment setup
source ../polaris_env_modules.sh

# Print environment information
echo "=========================================="
echo "Job started at: $(date)"
echo "Extended training run - should reach sparse phase (>128 iterations)"
echo "=========================================="

# Configuration - use extended config with 10 epochs
dnn="${dnn:-lstm}"
density="${density:-0.02}"
source exp_configs/lstm_extended.conf
compressor="${compressor:-oktopk}"
nworkers="${nworkers:-8}"

echo "Running with $nworkers workers, $max_epochs epochs (140+ iterations total)"
echo "First 128 iterations: Dense warm-up"
echo "After iteration 128: Sparse compression begins"
nwpernode=4
sigmascale=2.5
PY=$HOME/miniconda3/envs/py38_oktopk/bin/python

# Reset CUDA before running (clears any stale CUDA contexts)
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# MIG mode is system-configured, skip admin-level commands
# Just let PyTorch/CUDA driver handle GPU allocation

# Run the training
mpirun -np $nworkers $PY main_trainer.py \
    --dnn $dnn \
    --dataset $dataset \
    --max-epochs $max_epochs \
    --batch-size $batch_size \
    --nworkers $nworkers \
    --data-dir $data_dir \
    --lr $lr \
    --nwpernode $nwpernode \
    --nsteps-update $nstepsupdate \
    --compression \
    --sigma-scale $sigmascale \
    --density $density \
    --compressor $compressor

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
