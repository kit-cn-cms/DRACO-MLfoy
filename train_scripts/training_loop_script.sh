#!/bin/bash
#TODO: Check whether necessary to delete best_epoch.csv!

name=binary_crossentropy_Adam_modified
output1=ANN_training_
epochs=4000
iteration=100

for ((i=0; i<$iteration; i+=1)); do
    python train_dnn.py -o $output1"$i" -i /local/scratch/ssd/ycung/new_h5_files_2017 -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -P --binary --signal ttH -e $epochs
done