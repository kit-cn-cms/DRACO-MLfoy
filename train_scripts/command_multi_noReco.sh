#!/bin/bash

set=MergeAll_S02
echo $set
for i in 01 #01 02 #04 05 06 07 08 09 10
do
    for cat in 4j_ge3t 5j_ge3t ge6j_ge3t ge4j_3t ge4j_ge4t #le5j_3t ge6j_3t 
    do
        #if [ ${i} == 01 ];
        #then
            python ttZ_train.py -v ${set} -o ${set}/exp2_v${i} -i DNN_Input_splitTTToSemiLeptonic -c $cat -n Experimental -e 300 -p -s -1 -P -R -S ttZ #2> /storage/8/lbosch/${set}_v${i}_$cat.err
        #else
            #python ttZ_train.py -v ${set} -o ${set}/sp_v${i} -i DNN_Input_splitTTToSemiLeptonic -c $cat -n ttZAnalysis -e 300 -S ttZ 2> /storage/8/lbosch/${set}_v${i}_$cat.err
        #fi
    done
done 
