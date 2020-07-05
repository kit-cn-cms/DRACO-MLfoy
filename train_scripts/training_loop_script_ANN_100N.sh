#!/bin/bash
#TODO: Check whether necessary to delete best_epoch.csv!

name=binary_crossentropy_Adam_modified_100
output1=QT_ANN_100N_training_
output2=ANN_100N_training_

epochs=4000
variableset=new_variableset_25_v2
iteration=100

for ((i=0; i<$iteration; i+=1)); do
    python train_dnn.py -o $output1"$i" -i /local/scratch/ssd/ycung/new_h5_files_2017_v2 -c ge4j_ge3t -v $variableset -n "$name" -P -p --binary --signal ttH -e $epochs -q
    python train_dnn.py -o $output2"$i" -i /local/scratch/ssd/ycung/new_h5_files_2017_v2 -c ge4j_ge3t -v $variableset -n "$name" -P -p --binary --signal ttH -e $epochs
done