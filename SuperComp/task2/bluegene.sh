#!/bin/bash
Gs=(1000 2000)
Ns=(128 256 512)

PROG=dirichlet
Env=\"\"
MODE='CPU'
if [ "$1" = 'OMP' ];
then
    PROG=dirichlet_omp
    Env=\"OMP_NUM_THREADS=3\"
    MODE='OMP'
else
    MODE='CPU'
fi

for N in ${Ns[@]}
do
    for G in ${Gs[@]}
    do
        echo mpisubmit.bg -n ${N} -w 00:05:00 --stdout res/${N}_${G}_${MODE}.txt -e ${Env} ${PROG} -- ${G} ${G}
    done
done
