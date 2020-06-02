#!/bin/bash
#TODO: Check whether necessary to delete best_epoch.csv!

#_v2 = n_train_samples = batch_size
#_V3 = early_stopping_percentage = 0.01 und n_train_samples = batch size
#_V4 = early_stopping_percentage = 0.03 und n_train_samples = n_train_samples
#_V5 = early_stopping_percentage = 0.02 und learning_rate = 1e-4 und n_train_samples = n_train_samples
#_V6 = early_stopping_percentage = 0.02 und learning_rate = 1e-3 und n_train_samples = 0.75*n_train_samples 
#_V7 = early_stopping_percentage = 0.02 und learning_rate = 1e-3 und n_train_samples = 0.75*n_train_samples; Korrektur output_activation
#_V8 = early_stopping_percentage = 0.02 und learning_rate = 1e-3 und n_train_samples = 0.75*n_train_samples; Korrektur output_activation; loss in model.compile = neg_log_likelihood + kl mit kl = sum(model.losses)
#_V9 = early_stopping_percentage = 0.02 und learning_rate = 1e-3 und n_train_samples = 0.75*n_train_samples; Korrektur output_activation; loss in model.compile = neg_log_likelihood + kl mit kl = sum(model.losses)/tf.to_float(n_train_samples) und dafuer Skalierung direkt beim Layer entfernt
#_V10 = early_stopping_percentage = 0.02 und learning_rate = 1e-3 und n_train_samples = Anzahl Batches; Korrektur output_activation; loss in model.compile = neg_log_likelihood (ohne kl) und Skalierung beim Layer wieder hinzugefuegt
#_V11 = early_stopping_percentage = 0.02 und learning_rate = 1e-3 und n_train_samples = 0.75*n_train_samples; Korrektur output_activation; loss in model.compile = neg_log_likelihood (ohne kl) und Skalierung beim Layer wieder hinzugefuegt. Changed neg_log_likelihood function to return tf.reduce_mean(dist.log_prob(y_true), axis=-1) 








name=BNN
output1=TEST_Flipout_QT_BNN_training_
output2=TEST_Flipout_BNN_training_

epochs=4000

cd /home/ycung/Desktop/DRACO-MLfoy/train_scripts/
layers=("50") #"100" "200" "50,50")
loss=("without_axis" "softmaxcrossentropy" "log_prob" "axis_one" "binarycross" "sparse")

for i in "${!loss[@]}"; do
    python train_template_bnn_denseflipout.py -o $output1"${loss[$i]}" -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs -q --layers 50 --debugs ${loss[$i]}
    python train_template_bnn_denseflipout.py -o $output2"${loss[$i]}" -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs --layers 50 --debugs ${loss[$i]}
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

