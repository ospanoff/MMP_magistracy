#!/bin/bash

# Graph sizes
# P_min=5
# P_max=25
# P_step=5
P_min=20
P_max=21
P_step=1

# Number of MPI processes
# Ns=(1 2 4 8 16)
Ns=(1 2)

# Types of graphs
# TYPES=('RMAT' 'SSCA2')
TYPES=('RMAT')

PROG=./run.out
MPIRUN=mpirun
