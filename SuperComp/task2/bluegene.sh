#!/bin/bash
Gs=(1000 2000)

Ns=(128 256 512)
PROG=dirichlet
Env=\"\"
if [ "$1" = 'OMP' ];
then
    Ns=(32 64 128)
    PROG=dirichlet_omp
    Env=\"OMP_NUM_THREADS=3\"
fi

for N in ${Ns[@]}
do
    for G in ${Gs[@]}
    do
        echo mpisubmit.bg -n $N -w 00:05:00 --stdout res/${N}_${G}_$1.txt -e $Env $PROG -- $G $G
    done
done
