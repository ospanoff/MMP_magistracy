#!/bin/bash
set -e

source params.sh

TYPE=${TYPES[$1]}

echo $TYPE

for P in $(seq $P_min $P_step $P_max)
do
    FILE_NAME=${DATA_PATH}graph_${TYPE}_$P.bin
    echo 'Running' $P
    LD_LIBRARY_PATH=/home/edu-cmc-skmodel17-617-07/libs/boost/lib:$LD_LIBRARY_PATH ./run_st ${FILE_NAME} $P
    echo 'Done'
done
