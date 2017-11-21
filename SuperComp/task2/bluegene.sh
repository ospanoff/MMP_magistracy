#!/bin/bash
Gs=(1000 2000)
Ns=(1 128 256 512)

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
        if [ "$N" = 1 ]; then
            T='02:00'
        else
            T='00:15'
        fi
        echo mpisubmit.bg -n ${N} -w ${T}:00 --stdout res/${N}_${G}_${MODE}.txt -e ${Env} ${PROG} -- ${G} ${G}
    done
done
