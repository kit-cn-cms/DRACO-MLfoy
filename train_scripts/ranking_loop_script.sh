#!/bin/bash

names=('ANN_' 'QT_ANN_' 'BNN_' 'QT_BNN_' 'MultiANN_' 'QT_MultiANN_')
input=('training_ANN/ANN_training_*_JTSTRING' 'training_ANN/QT_ANN_training_*_JTSTRING' 'training_BNN/BNN_training_*_JTSTRING' 'training_BNN/QT_BNN_training_*_JTSTRING' 'training_MultiANN/MultiANN_training_*_JTSTRING' 'training_MultiANN/QT_MultiANN_training_*_JTSTRING')
output=/home/ycung/Desktop/DRACO-MLfoy/workdir/variable_ranking_06-05-2020
nplot=15

cd /home/ycung/Desktop/DRACO-MLfoy/utils/

for i in "${!names[@]}"; do
    python getTopVariables.py -w /home/ycung/Desktop/DRACO-MLfoy/workdir -i ${input[$i]} -o $output -p -t first_layer --filename ${names[$i]} --nplot $nplot ge4j_ge3t
done

python getTopVariables.py -o $output --nplot $nplot --count