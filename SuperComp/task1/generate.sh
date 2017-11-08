#!/bin/bash
set -e

source params.sh

make generator

for TYPE in ${TYPES[@]}
do
    for P in $(eval echo {$P_min..$P_max..$P_step})
    do
        FILE_NAME=graph_"$TYPE"_$P.bin
        ./generator.out -s $P -directed -weighted -file $FILE_NAME -type $TYPE
    done
done
