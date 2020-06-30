#!/bin/bash
#TODO: Check whether necessary to delete best_epoch.csv!

name=BNN
output1=Flipout_QT_BNN_training_
output2=Flipout_BNN_training_
output3=Variational_QT_BNN_training_
output4=Variational_BNN_training_

epochs=4000
iteration=155

cd /home/ycung/Desktop/DRACO-MLfoy/train_scripts/

python train_template_bnn_denseflipout.py -o $output2"150","150","150"_"0" -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs --layers 150,150,150
