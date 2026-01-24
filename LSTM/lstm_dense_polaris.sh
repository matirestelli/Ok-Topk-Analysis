#!/bin/bash -l
#PBS -l select=32:ngpus=1
#PBS -l walltime=01:20:00
#PBS -l filesystems=home:grand
#PBS -q backfill
#PBS -A UIC-HPC
#PBS -N lstm_dense
#PBS -o logs/lstm_dense_${PBS_JOBID}.out
#PBS -e logs/lstm_dense_${PBS_JOBID}.err

mkdir -p logs

source ../polaris_env_modules.sh

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $PBS_JOBID"
echo "Nodes: $(cat $PBS_NODEFILE | wc -l)"
echo "Queue: $PBS_QUEUE"
echo "=========================================="
module list
echo ""
which python
python --version
which nvcc
nvcc --version
echo "=========================================="

dnn="${dnn:-lstman4}"
source exp_configs/$dnn.conf
nworkers="${nworkers:-32}"

echo "Running with $nworkers workers (dense, no compression)"
nwpernode=1
sigmascale=2.5
PY=python

srun python main_trainer.py \
    --dnn $dnn \
    --dataset $dataset \
    --max-epochs 10 \
    --batch-size $batch_size \
    --nworkers $nworkers \
    --data-dir $data_dir \
    --lr $lr \
    --nwpernode $nwpernode \
    --nsteps-update $nstepsupdate \
    --sigma-scale $sigmascale

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
