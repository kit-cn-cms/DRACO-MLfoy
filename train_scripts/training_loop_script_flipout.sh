#!/bin/bash
#TODO: Check whether necessary to delete best_epoch.csv!

#_v2 = n_train_samples = batch_size
#_V3 = early_stopping_percentage = 0.01 und n_train_samples = batch size
#_V4 = early_stopping_percentage = 0.03 und n_train_samples = n_train_samples
#_V5 = early_stopping_percentage = 0.02 und learning_rate = 1e-4 und n_train_samples = n_train_samples
#_V6 = early_stopping_percentage = 0.02 und learning_rate = 1e-3 und n_train_samples = 0.75*n_train_samples 
#_V7 = early_stopping_percentage = 0.02 und learning_rate = 1e-3 und n_train_samples = 0.75*n_train_samples; Korrektur output_activation
#_V8 = early_stopping_percentage = 0.02 und learning_rate = 1e-3 und n_train_samples = 0.75*n_train_samples; Korrektur output_activation; loss in model.compile = neg_log_likelihood + kl mit kl = sum(model.losses)




name=BNN
output1=TEST_Flipout_QT_BNN_training_
output2=TEST_Flipout_BNN_training_

epochs=4000

cd /home/ycung/Desktop/DRACO-MLfoy/train_scripts/
layers=("50" "100" "200" "50,50")
for i in "${!layers[@]}"; do
    python train_template_bnn_denseflipout.py -o $output1"${layers[$i]}"_v8 -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs -q --layers ${layers[$i]}
    python train_template_bnn_denseflipout.py -o $output2"${layers[$i]}"_v8 -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs --layers ${layers[$i]}
done

# #ANN
# name=binary_crossentropy_Adam
# output1=QT_ANN_training_
# output2=ANN_training_
# epochs=4000
# iteration=200

# #BNN
# names=('BNN_L2_5050' 'BNN_L3_505050')
# epochs=4000

# cd /home/ycung/Desktop/DRACO-MLfoy/train_scripts/
# # for name in "${names[@]}"; do
# python train_template_bnn.py -o QT_BNN_BNN_L2_5050_training -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n BNN -p --printroc --log --binary --signal ttH -e "$epochs" -q --layers 50,50
# python train_template_bnn.py -o BNN_L2_5050_training_ -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n BNN -p --printroc --binary --signal ttH -e "$epochs" --layers 50,50
# python train_template_bnn.py -o QT_BNN_BNN_L3_505050_training -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n BNN -p --printroc --log --binary --signal ttH -e "$epochs" -q --layers 50,50,50
# python train_template_bnn.py -o BNN_L3_505050_training_ -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n BNN -p --printroc --binary --signal ttH -e "$epochs" --layers 50,50,50
# # done

