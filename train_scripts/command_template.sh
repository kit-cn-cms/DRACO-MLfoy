#!/bin/bash

for cat in 4j_ge3t 5j_ge3t ge6j_ge3t
do
    python ttZ_train.py -v S03 -o S03/v4 --balanceSamples -i DNN_Input_allVar -c $cat -n ttZAnalysis4 -e 80 -p -s -1 -P -R -S ttZ,ttH 2> /storage/8/lbosch/S03_v4_$cat.err
done

