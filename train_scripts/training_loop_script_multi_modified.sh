#!/bin/bash
#TODO: Check whether necessary to delete best_epoch.csv!

name=multiANN
output1=modified_MultiANN_QT_1e5_subsamples_training_
# output2=modified_MultiANN_training_
variableset=new_variableset_25_v2
epochs=4000
iteration=100

for ((i=0; i<$iteration; i+=1)); do
    python train_dnn.py -o $output1"$i" -i /local/scratch/ssd/ycung/new_h5_files_2017_v2 -c ge4j_ge3t -v $variableset -n "$name" -p -P -e $epochs -q
    # python train_dnn.py -o $output2"$i" -i /local/scratch/ssd/ycung/new_h5_files_2017_v2 -c ge4j_ge3t -v $variableset -n "$name" -p -P -e $epochs
done