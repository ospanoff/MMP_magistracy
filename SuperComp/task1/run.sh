#!/bin/bash
set -e

source params.sh

make

for TYPE in ${TYPES[@]}
do
    echo $TYPE
    for P in $(eval echo {$P_min..$P_max..$P_step})
    do
        FILE_NAME=graph_"$TYPE"_$P.bin
        for N in ${Ns[@]}
        do
            echo 'Starting: ' $N $P
            $MPIRUN -np $N $PROG $FILE_NAME $P 50
        done
    done
done
