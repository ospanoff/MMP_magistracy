#!/bin/bash
module add slurm
module add openmpi/1.8.4-icc

Gs=(1000 2000)
Ns=(8 16 32 64 128)
PROG=dirichlet

for N in ${Ns[@]}
do
    for G in ${Gs[@]}
    do
        sbatch -n ${N} -p test -o ${N}_${G}.txt -t 0-00:05:00 ompi ${PROG} ${G} ${G}
        while [ $(squeue --user $USER | wc -l) -gt 3 ]
        do
            sleep 60
            echo Checking again...
        done
    done
done

module rm openmpi/1.8.4-icc
