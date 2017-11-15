#!/bin/bash

# Graph sizes
P_min=14
P_max=18
P_step=1

# Number of MPI processes
Ns=(1 2 4 8 16 32 64 128)

# Types of graphs
TYPES=('RMAT' 'SSCA2')

DATA_PATH=../data/

PROG=./run
MPIRUN=mpisubmit.bg
