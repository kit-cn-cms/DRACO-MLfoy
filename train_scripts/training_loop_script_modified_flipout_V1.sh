#!/bin/bash
#TODO: Check whether necessary to delete best_epoch.csv!

name=BNN_Flipout_modified_V1
output1=modified_Flipout_V1_QT_BNN_training_
output2=modified_Flipout_V1_BNN_training_
variableset=new_variableset_25_v2
epochs=4000

#Training 1 Layer
for ((i=50; i<250; i+=50)); do
    python train_template_bnn_denseflipout.py -o $output1"$i" -i /local/scratch/ssd/ycung/new_h5_files_2017_v2 -c ge4j_ge3t -v $variableset -n "$name" -p -P --binary --signal ttH -e $epochs -q --layers $i
    python train_template_bnn_denseflipout.py -o $output2"$i" -i /local/scratch/ssd/ycung/new_h5_files_2017_v2 -c ge4j_ge3t -v $variableset -n "$name" -p -P --binary --signal ttH -e $epochs --layers $i
done

#Training 2 Layer
for ((i=50; i<150; i+=50)); do
    python train_template_bnn_denseflipout.py -o $output1"$i","$i" -i /local/scratch/ssd/ycung/new_h5_files_2017_v2 -c ge4j_ge3t -v $variableset -n "$name" -p -P --binary --signal ttH -e $epochs -q --layers $i,$i
    python train_template_bnn_denseflipout.py -o $output2"$i","$i" -i /local/scratch/ssd/ycung/new_h5_files_2017_v2 -c ge4j_ge3t -v $variableset -n "$name" -p -P --binary --signal ttH -e $epochs --layers $i,$i
done

#Training 3 Layer
for ((i=50; i<150; i+=50)); do
    python train_template_bnn_denseflipout.py -o $output1"$i","$i","$i" -i /local/scratch/ssd/ycung/new_h5_files_2017_v2 -c ge4j_ge3t -v $variableset -n "$name" -p -P --binary --signal ttH -e $epochs -q --layers $i,$i,$i
    python train_template_bnn_denseflipout.py -o $output2"$i","$i","$i" -i /local/scratch/ssd/ycung/new_h5_files_2017_v2 -c ge4j_ge3t -v $variableset -n "$name" -p -P --binary --signal ttH -e $epochs --layers $i,$i,$i
done   
