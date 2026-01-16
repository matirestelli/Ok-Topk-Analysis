# Repository Structure Guide: Ok-Topk Sparse Allreduce

This repository implements **Ok-Topk**, a near-optimal sparse allreduce algorithm for distributed deep learning (published in PPoPP'22).

## Key Files and Their Roles

Each file in this repository has a specific purpose in the distributed training system:

| File | What It Represents | Key Responsibility |
|------|-------------------|-------------------|
| **`*.sh`** scripts | Job launcher | Spawns multiple MPI processes (workers) via SLURM; configures experiment parameters |
| **`main_trainer.py`** | Worker initialization | Each MPI process runs this; sets up MPI communicator, assigns GPU to process, broadcasts initial model |
| **`distributed_optimizer.py`** | Gradient interceptor | Runs in each worker; hooks into backpropagation to catch gradients and queue them for communication |
| **`allreducer.py`** | Communication engine | Background thread in each worker; performs actual MPI send/receive operations to synchronize gradients; **core of Ok-Topk algorithm** |
| **`compression.py`** | Sparsification logic | Computes which gradient elements to send (top-k selection, thresholding, residual tracking) |
| **`dl_trainer.py`** | Training loop | Handles forward/backward passes, data loading, and model updates |
| **`model_builder.py`** | Model architecture | Defines neural network structure (VGG, LSTM, BERT) |
| **`datasets.py`** / **`ptb_reader.py`** | Data loading | Loads and preprocesses training data for each worker |
| **`settings.py`** | Configuration | Global settings and hyperparameters |
| **`utils.py`** | Helper functions | Utility functions for top-k selection, timing, logging |

**Key insight:** Each worker is an MPI process. The files work together to enable workers to train independently while synchronizing gradients through sparse communication.

## Root Level

- **README.md** - Setup instructions and overview
- **requirements.txt** - Python dependencies for the project
- **LICENSE** - Project license
- **apex/** - NVIDIA's library for mixed precision training (external dependency, cloned during setup)

## Three Main Experiment Folders

The repository contains three main folders (BERT/, LSTM/, VGG/), each implementing complete training code for a different neural network model. Each folder has a **similar structure** with the core allreduce algorithms replicated.

### Core Allreduce Implementation Files

These files are duplicated in each of the three folders and contain the main algorithmic contributions:

#### **`allreducer.py`** - Main Sparse Allreduce Algorithms
- **Ok-Topk implementation** - The novel sparse allreduce algorithm
- **Baseline algorithms**:
  - TopkDSA
  - gTopk
  - GaussianK
- Communication primitives using MPI
- Functions like `topk_sparse_allreduce`, `oktopk_sparse_allreduce`

#### **`compression.py`** - Gradient Compression Strategies
- TopK compressor
- Gaussian compressor
- Various sparsification methods
- Residual accumulation logic
- Error feedback mechanisms

#### **`distributed_optimizer.py`** - Distributed SGD Optimizer
- SGD optimizer integrated with gradient compression
- Handles communication and compression coordination

### Training Infrastructure

Each folder contains:

- **`main_trainer.py`** / **`dl_trainer.py`** - Main training loops and experiment orchestration
- **`horovod_trainer.py`** - Horovod-based distributed training alternative
- **`model_builder.py`** - Neural network architecture definitions specific to each model
- **`datasets.py`** / **`ptb_reader.py`** - Data loading and preprocessing utilities
- **`settings.py`** - Configuration parameters and hyperparameters
- **`utils.py`** - Helper functions and utilities
- **`evaluate.py`** - Model evaluation logic

### Experiment Scripts

Each folder contains bash scripts to run different compression methods:

- **`*_dense.sh`** - Run baseline without compression (standard allreduce)
- **`*_oktopk.sh`** - Run **Ok-Topk algorithm** (main contribution)
- **`*_gaussiank.sh`** - Run GaussianK baseline
- **`*_gtopk.sh`** - Run gTopk baseline
- **`*_topkA.sh`** - Run TopkA baseline
- **`*_topkDSA.sh`** - Run TopkDSA baseline
- **`sbatch_*_jobs.sh`** - SLURM job submission scripts for batch experiments

### Data Directories

- **VGG/vgg_data/** - CIFAR-10 dataset location
- **LSTM/audio_data/** - AN4 audio dataset location
- **BERT/bert/bert_data/** - Wikipedia dataset location

## BERT/ - Special Structure

The BERT folder has some unique files for pipeline parallelism:

- **`communication.py`** - Pipeline parallelism communication handlers using PyTorch distributed
- **`optimizer.py`** - Base optimizer implementation
- **`optimizer_with_aggregation.py`** - Optimizer with gradient aggregation
- **`optimizer_with_stashing.py`** - Optimizer with gradient stashing
- **`optimizer_with_stashing_and_aggregation.py`** - Combined approach
- **`runtime.py`** - Pipeline execution runtime and scheduling
- **`runtime_utilities.py`** - Runtime helper functions
- **`threadsafe_queue.py`** / **`threadsafe_counter.py`** - Thread-safe data structures
- **`launch.py`** - Distributed launch utility
- **`bert/`** subdirectory - BERT model implementation and data

## Understanding the Code

### To Understand the Core Algorithm

Focus on these files (use LSTM/ or VGG/ versions as they're simpler):

1. **LSTM/compression.py** - How gradients are compressed and sparsified
2. **LSTM/allreducer.py** - How the sparse allreduce communication works
3. **LSTM/distributed_optimizer.py** - How compression integrates with the optimizer

### Key Functions to Look At

In **allreducer.py**:
- `oktopk_sparse_allreduce()` - The main Ok-Topk algorithm implementation
- `topk_sparse_allreduce()` - Baseline topk algorithm
- `gtopk_sparse_allreduce()` - gTopk baseline
- `gaussiank_sparse_allreduce()` - GaussianK baseline

In **compression.py**:
- `TopKCompressor.compress()` - Top-k gradient selection
- `GaussianKCompressor.compress()` - Gaussian-based sparsification

### Code Duplication

The core allreduce and compression code is **essentially duplicated across all three folders** (BERT, LSTM, VGG) to run experiments on different models. The algorithms are the same, but adapted for:
- Different model architectures
- Different dataset loaders
- Different evaluation metrics

### Running Experiments

Each model folder has scripts to compare all methods:

```bash
# For VGG experiments
cd ./VGG
./sbatch_vgg_jobs.sh

# For LSTM experiments
cd ./LSTM
./sbatch_lstm_jobs.sh

# For BERT experiments
cd ./BERT/bert/
./sbatch_bert_jobs.sh
```

These scripts run all compression methods (dense, oktopk, gaussiank, gtopk, topkA, topkDSA) for comparison.

## Algorithm Comparison

The repository implements and compares:

1. **Ok-Topk** (Main contribution) - Near-optimal sparse allreduce with <6k communication volume
2. **Dense** - Standard allreduce without compression (baseline)
3. **TopkA** - Top-k aggregation baseline
4. **TopkDSA** - Top-k with decentralized stochastic averaging
5. **gTopk** - Generalized Top-k
6. **GaussianK** - Gaussian-based sparsification

## Publication

The work is published in PPoPP'22: [DOI](https://doi.org/10.1145/3503221.3508399)

## Quick Start for Code Understanding

1. **Start with**: [LSTM/compression.py](LSTM/compression.py) - Understand gradient compression
2. **Then read**: [LSTM/allreducer.py](LSTM/allreducer.py) - See how sparse communication works
3. **Check**: [LSTM/distributed_optimizer.py](LSTM/distributed_optimizer.py) - Integration with training
4. **Run**: `LSTM/lstm_oktopk.sh` - See how experiments are configured
