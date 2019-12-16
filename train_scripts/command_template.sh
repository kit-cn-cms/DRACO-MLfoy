#!/bin/bash

for i in 1 #2 3 4 5
do
	for cat in ge4j_3t ge4j_ge4t 4j_ge3t 5j_ge3t ge6j_ge3t 
	#le5j_ge4t ge6j_ge4t le5j_ge3t as an idea, but does not really make sense
	do
    	python ttZ_train.py -v Raw_Variables_noRecoBoson -o Raw_Vars_prio/bin_noRecoBoson_v${i} -i DNN_Input -c $cat -n ttZAnalysis_bin -e 120 -p -s -1 -P -R --binary -S ttZ -a ttZ,ttH #2> /storage/8/lbosch/S04_bin_$cat.err
	done
done 
