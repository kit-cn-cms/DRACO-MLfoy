#!/bin/bash

for i in 1  #2 3 4 5
do
	for cat in  4j_ge3t # 5j ge3t ge6j_ge3t #ge4j_3t ge4j_ge4t
	do
    	python ttZ_train.py -v topVariables_validated_decorr -o topVariables_validated_decorr_2018vars/splitTTToSemiLeptonic_v${i} -i DNN_Input_splitTTToSemiLeptonic -c $cat -n ttZAnalysis -e 120 -p -s -1 -P -R -S ttZ  #2> /storage/8/lbosch/S04_bin_$cat.err
	done
done 
