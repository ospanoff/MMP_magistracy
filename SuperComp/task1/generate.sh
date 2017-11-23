#!/bin/bash
set -e

source params.sh

make generator

for TYPE in ${TYPES[@]}
do
    for P in $(seq ${P_min} ${P_step} ${P_max})
    do
        FILE_NAME=${DATA_PATH}graph_${TYPE}_${P}.bin
        echo Generating ${P} ${TYPE}
        ./generator -s ${P} -directed -unweighted -file ${FILE_NAME} -type ${TYPE}
    done
done
