#!/bin/bash
set -e

source params.sh

OUT_PATH=../out
TYPE=${TYPES[$1]}

echo ${TYPE}

for P in $(seq ${P_min} ${P_step} ${P_max})
do
    FILE_NAME=${DATA_PATH}graph_${TYPE}_${P}.bin
    for N in ${Ns[@]}
    do
        OUT_FILE=${P}_${N}_${TYPE}
        $MPIRUN -n ${N} --stdout ${OUT_PATH}/${OUT_FILE}.out --stderr ${OUT_PATH}/${OUT_FILE}.err -w 00:10:00 ${PROG} -- ${FILE_NAME} ${P}
        echo -e ""
    done
done
