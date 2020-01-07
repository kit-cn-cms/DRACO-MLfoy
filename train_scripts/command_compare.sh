#!/bin/bash

for i in 3 4 5 6 
do
	for cat in  4j_ge3t 5j_ge3t ge6j_ge3t #ge4j_3t ge4j_ge4t
	do
    	python ttZ_train.py -v topVariables_validated_decorr -o topVariables_validated_decorr_2018vars/splitTTToSemiLeptonic_v${i} -i DNN_Input_splitTTToSemiLeptonic -c $cat -n ttZAnalysis -e 300 -p -s -1 -P -R -S ttZ 2> /storage/8/lbosch/compare_v${i}_${cat}.err
	done
done 
