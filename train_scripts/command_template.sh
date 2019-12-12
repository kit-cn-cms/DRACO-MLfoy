#!/bin/bash

for i in 4 5 6
do
	for cat in 4j_ge3t 5j_ge3t ge6j_ge3t 
	do
    	python ttZ_train.py -v S04 -o S04_test/bin_v${i} -i DNN_Input_allVar -c $cat -n ttZAnalysis_bin -e 80 -p -s -1 -P -R --binary -S ttZ -a ttZ,ttH #2> /storage/8/lbosch/S04_bin_$cat.err
	done
done 
