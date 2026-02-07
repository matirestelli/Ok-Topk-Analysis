#!/usr/bin/env python
from mpi4py import MPI
print("SUCCESS: mpi4py imported successfully!")
comm = MPI.COMM_WORLD
print(f"MPI Initialized: {comm.Get_rank()} / {comm.Get_size()}")
