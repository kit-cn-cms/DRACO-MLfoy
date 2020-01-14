#!/bin/bash

set=RecoVarsOnly
echo $set
for i in 1 2 3 #4 5
do
	for cat in ge4j_ge4t
	#le5j_ge4t ge6j_ge4t le5j_ge3t as an idea, but does not really make sense
	do
    	python ttZ_train.py -v ${set} -o ${set}/bin_splitTTToSemileptonic_v${i} -i DNN_Input_splitTTToSemiLeptonic -c $cat -n ttZAnalysis_bin -e 300 -p -s -1 -P -R --binary -S ttZ -a ttZ,ttH #2> /storage/8/lbosch/S04_bin_$cat.err
	done
done 

##multiclass:
set=RecoVarsOnly
echo $set
for i in 1 2 3 #4 5
do
 	for cat in  ge4j_ge4t
 	do
     	python ttZ_train.py -v ${set} -o ${set}/splitTTToSemileptonic_v${i} -i DNN_Input_splitTTToSemiLeptonic -c $cat -n ttZAnalysis -e 300 -p -s -1 -P -R  -S ttZ  #2> /storage/8/lbosch/S04_bin_$cat.err
 	done
 done 
