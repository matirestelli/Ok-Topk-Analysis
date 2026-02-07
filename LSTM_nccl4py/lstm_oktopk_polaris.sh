#!/bin/bash -l
#PBS -l select=2:ngpus=4
#PBS -l walltime=00:15:00
#PBS -l filesystems=home:grand
#PBS -q debug
#PBS -A UIC-HPC
#PBS -N lstm_oktopk_debug
#PBS -o logs/lstm_oktopk_debug${PBS_JOBID}.out
#PBS -e logs/lstm_oktopk_debug${PBS_JOBID}.err

# Change to the directory where the job was submitted
cd ${PBS_O_WORKDIR}

# Create logs directory if it doesn't exist
mkdir -p logs

# Source the Polaris environment setup
# This loads modules and activates the conda environment
source ../polaris_env_modules.sh

# Set LD_LIBRARY_PATH for mpi4py to find Cray MPICH libraries
export LD_LIBRARY_PATH=/opt/cray/pe/mpich/9.0.1/ofi/nvidia/23.3/lib-abi-mpich:$LD_LIBRARY_PATH

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

# Configuration
dnn="${dnn:-lstm}"
density="${density:-0.02}"
source exp_configs/lstm_debug.conf
compressor="${compressor:-oktopk}"
nworkers="${nworkers:-8}"  # 2 nodes * 4 GPUs per node = 8 workers total

echo "Running with $nworkers workers"
nwpernode=4  # Using 4 GPUs per node (max capability)
sigmascale=2.5
PY=python

# Run the training (srun provides MPI environment automatically)
srun python -m mpi4py main_trainer.py \
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
