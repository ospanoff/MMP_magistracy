#!/bin/bash
set -e

source params.sh

#make

TYPE=${TYPES[$1]}

echo $TYPE

for P in $(seq $P_min $P_step $P_max)
do
    FILE_NAME=${DATA_PATH}graph_${TYPE}_$P.bin
    for N in ${Ns[@]}
    do
        ./check ${FILE_NAME}_${P}.mp_res ${FILE_NAME}.st_res
    done
done

echo 'Pass'
