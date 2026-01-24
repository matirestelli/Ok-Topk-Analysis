# Ok-Topk Analysis and Study

This repository contains an analysis of the Ok-Topk algorithm for near-optimal sparse allreduce in distributed deep learning. The work focuses on:

- **Understanding Ok-Topk's communication patterns and worker coordination** - Analysis of how MPI processes communicate during distributed training
- **Reproducing the original results on smaller-scale test cases** - Validation and experimentation with reduced cluster sizes
- **Analyzing alternative communication approaches** - Exploring modern primitives such as NCCL and NVSHMEM for collective operations
- **Comparing communication efficiency** - Evaluating different implementations of sparse allreduce algorithms

## Documentation

- **[REPOSITORY_GUIDE.md](REPOSITORY_GUIDE.md)** - Overview of repository structure and key files
- **[PARALLELIZATION_AND_COMMUNICATION_GUIDE.md](PARALLELIZATION_AND_COMMUNICATION_GUIDE.md)** - Detailed explanation of worker communication and MPI implementation
- **[POLARIS_SETUP_GUIDE.md](POLARIS_SETUP_GUIDE.md)** - Setup instructions for running on Polaris supercomputer (ALCF)

## Near-Optimal Sparse Allreduce for Distributed Deep Learning (published in PPoPP'22)

Ok-Topk is a scheme for distributed training with sparse gradients. Ok-Topk integrates a novel sparse allreduce algorithm (less than 6k communication volume which is asymptotically optimal) with the decentralized parallel Stochastic Gradient Descent (SGD) optimizer, and its convergence is proved theoretically and empirically. All baselines, such as TopkDSA, gTopk, and Guassiank, are already integrated in the repo.

## Setup the environment

### For Polaris Supercomputer (ALCF)
**See [POLARIS_SETUP_GUIDE.md](POLARIS_SETUP_GUIDE.md) for detailed Polaris-specific instructions.**

Quick setup on Polaris:
```bash
bash setup_polaris_env.sh
```

### For Other Systems (Generic Setup)
To install the required Python modules: 

`conda create --name py38_oktopk python=3.8`

`conda activate py38_oktopk`

`pip3 install pip==20.2.4`

`pip install -r requirements.txt`

`MPICC="cc -shared" pip install --no-binary=mpi4py mpi4py`

`git clone https://github.com/NVIDIA/apex`

`cd apex`

`pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./`

## Prepare Datasets

### Cifar-10 for VGG
`cd ./VGG/vgg_data`

`wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`

`tar -zxvf cifar-10-python.tar.gz`

### AN4 for LSTM
`cd ./LSTM/audio_data`

`wget https://www.dropbox.com/s/l5w4up20u5pfjxf/an4.zip`

`unzip an4.zip`

### Wikipedia for BERT
`cd ./BERT/bert/bert_data/`

Prepare the dataset according to the README file.

## Run jobs
We run experiments on GPU clusters with SLURM job scheduler.
To evaluate the performance of `Ok-Topk`, `Gaussiank`, `gtopk`, `topkA`, `topkDSA`, and `dense`, run the jobs as follows.

### To run VGG jobs
`cd ./VGG`

`./sbatch_vgg_jobs.sh`

### To run LSTM jobs
`cd ./LSTM`

`./sbatch_lstm_jobs.sh`

### To run BERT jobs
`cd ./BERT/bert/`

`./sbatch_bert_jobs.sh`

## Publication

The work of Ok-Topk is pulished in PPoPP'22. [DOI](https://doi.org/10.1145/3503221.3508399)

## License

See [LICENSE](LICENSE).
