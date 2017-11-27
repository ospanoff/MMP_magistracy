#!/bin/bash
module add slurm
module add openmpi cuda

Gs=(1000 2000)
Ns=(1 2 4 8 16)
PROG=dirichlet_cuda

for N in ${Ns[@]}
do
    for G in ${Gs[@]}
    do
        sbatch -N ${N} --ntasks-per-node=2 -p gputest -o ${N}_${G}.txt ompi ${PROG} ${G} ${G}
        while [ $(squeue --user $USER | wc -l) -gt 3 ]
        do
            sleep 60
            echo Checking again...
        done
    done
done

module rm openmpi cuda
