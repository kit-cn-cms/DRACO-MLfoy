#!/bin/bash
#TODO: Check whether necessary to delete best_epoch.csv!

name=BNN
output1=Flipout_QT_BNN_training_
output2=Flipout_BNN_training_
output3=Variational_QT_BNN_training_
output4=Variational_BNN_training_

epochs=4000
iteration=155
iteration_2=205
iteration_3=255


#Ziel:
#50 - 250 in 5er schritten for variational und flipout
#"10,10" "10,10,10" - "150,150" "150,150,150" for densevariational und flipout
#layers=("10" "10,10,10")

cd /home/ycung/Desktop/DRACO-MLfoy/train_scripts/

for ((l=0; l<2; l+=1)); do
    for ((i=10; i<$iteration; i+=5)); do
        python train_template_bnn_denseflipout.py -o $output1"$i","$i"_"$l" -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs -q --layers "$i","$i"
        python train_template_bnn_denseflipout.py -o $output2"$i","$i"_"$l" -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs --layers "$i","$i"
        python train_template_bnn_denseflipout.py -o $output1"$i","$i","$i"_"$l" -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs -q --layers "$i","$i","$i"
        python train_template_bnn_denseflipout.py -o $output2"$i","$i","$i"_"$l" -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs --layers "$i","$i","$i"
    done

    for ((j=50; j<$iteration_2; j+=5)); do
        python train_template_bnn_denseflipout.py -o $output1"$j"_"$l" -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs -q --layers "$j"
        python train_template_bnn_denseflipout.py -o $output2"$j"_"$l" -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs --layers "$j"
        python train_template_bnn.py -o $output3"$j"_"$l" -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs -q --layers "$j"
        python train_template_bnn.py -o $output4"$j"_"$l" -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs --layers "$j"
    done

    for ((j=$iteration_2; j<$iteration_3; j+=5)); do
        python train_template_bnn_denseflipout.py -o $output1"$j"_"$l" -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs -q --layers "$j"
        python train_template_bnn_denseflipout.py -o $output2"$j"_"$l" -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs --layers "$j"
    done

    for ((j=$iteration_2; j<$iteration_3; j+=5)); do
        python train_template_bnn.py -o $output3"$j"_"$l" -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs -q --layers "$j"
        python train_template_bnn.py -o $output4"$j"_"$l" -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs --layers "$j"
    done

    for ((k=10; k<$iteration; k+=5)); do
        python train_template_bnn.py -o $output3"$k","$k"_"$l" -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs -q --layers "$k","$k"
        python train_template_bnn.py -o $output4"$k","$k"_"$l" -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs --layers "$k","$k"
        python train_template_bnn.py -o $output3"$k","$k","$k"_"$l" -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs -q --layers "$k","$k","$k"
        python train_template_bnn.py -o $output4"$k","$k","$k"_"$l" -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs --layers "$k","$k","$k"
    done
done
