#!/bin/bash

for cat in 4j_ge3t 5j_ge3t ge6j_ge3t
do
    python ttZ_train.py -v secondSelec -o secondSelec/v1 --balanceSamples -i DNN_Input_allVar -c $cat -n ttZAnalysis -e 70 -p -s -1 -P -R -S ttZ,ttH
done
# 
# 
# python ttZ_train.py -v secondSelec -o secondSelec/v1 --balanceSamples -i DNN_Input_allVar -c 4j_ge3t -n ttZAnalysis -e 70 -p -s -1 -P -R -S ttZ,ttH
# 
# python ttZ_train.py -v secondSelec -o secondSelec/v1 --balanceSamples -i DNN_Input_allVar -c 5j_ge3t -n ttZAnalysis -e 70 -p -s -1 -P -R -S ttZ,ttH
# 
# python ttZ_train.py -v secondSelec -o secondSelec/v1 --balanceSamples -i DNN_Input_allVar -c ge6j_ge3t -n ttZAnalysis -e 70 -p -s -1 -P -R -S ttZ,ttH
