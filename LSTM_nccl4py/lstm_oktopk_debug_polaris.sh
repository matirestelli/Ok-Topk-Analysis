#!/bin/bash -l
#PBS -l select=2:ngpus=4
#PBS -l walltime=00:15:00
#PBS -l filesystems=home:grand
#PBS -q debug-scaling
#PBS -A UIC-HPC
#PBS -N lstm_oktopk_debug
#PBS -o /home/mrest/Ok-Topk-Analysis/LSTM/logs/lstm_oktopk_debug_${PBS_JOBID}.out
#PBS -e /home/mrest/Ok-Topk-Analysis/LSTM/logs/lstm_oktopk_debug_${PBS_JOBID}.err

# Change to the directory where the job was submitted
cd ${PBS_O_WORKDIR}

# Create logs directory if it doesn't exist
mkdir -p logs

# Source the Polaris environment setup
# This loads modules and activates the conda environment
source ../polaris_env_modules.sh

# Print environment information
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: ${PBS_JOBID:-not-in-job}"
if [ -f "$PBS_NODEFILE" ]; then
    echo "Nodes: $(cat $PBS_NODEFILE | wc -l)"
else
    echo "Nodes: running on login node (not in job)"
fi
echo "Queue: ${PBS_QUEUE:-not-in-job}"
echo "=========================================="
module list
echo ""
which python
python --version
which nvcc
nvcc --version
echo "=========================================="

# Configuration - use debug config
dnn="${dnn:-lstm}"
density="${density:-0.02}"
source exp_configs/lstm_debug.conf
compressor="${compressor:-oktopk}"
nworkers="${nworkers:-8}"  # 2 nodes * 4 GPUs per node = 8 workers total

echo "Running with $nworkers workers"
nwpernode=4  # Using 4 GPUs per node (max capability)
sigmascale=2.5
PY=$HOME/miniconda3/envs/py38_oktopk/bin/python

# Run the training (use mpirun for MPI environment)
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
