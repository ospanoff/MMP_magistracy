#!/bin/bash
module add openmpi/1.8.4-icc

Gs=(1000 2000)
Ns=(8 16 32 64 128)
PROG=dirichlet

for N in ${Ns[@]}
do
    for G in ${Gs[@]}
    do
        echo sbatch -n $N -p test -o res/${N}_${G}.txt ompi $PROG $G $G
    done
done

module rm openmpi/1.8.4-icc
