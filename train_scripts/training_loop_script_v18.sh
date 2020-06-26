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
#_V12 = alles wie davor, neg_log_likelihood wieder geandert zu -dist.log(...) und bias_posterior_fn is_singular entfernt und fuer bias_ prior_fn Funktion statt None
#_V13 = alles wie davor bias_posterior_fn und bias_ prior_fn  wieder zurueck geaendert; Testen wo Netz ab welcher Groeße das Training nicht mehr funktioniert
#_V14 = alles wie davor, aber nun loss von dem https://towardsdatascience.com/deep-learning-segmentation-with-uncertainty-via-3d-bayesian-convolutional-neural-networks-6b1c7277b078 benutzt und Skalierung von KL Term aus Layer rausgenommen
#_V15 = alles wie davor, aber nun kl - Term durch Gesamtzahl trainingssample und nicht durch Anzahl batches geteilt
#_V16 = wie v13 aber kernel_posterior_fn = tfp.layers.util.default_mean_field_normal_fn()
#_V17 = BNN_v17 benutzt "kernel_posterior_fn":         tfp.layers.util.default_mean_field_normal_fn(),
                      # "kernel_prior_fn":             default_mean_field_normal_fn(is_singular=True),
                      # "bias_posterior_fn":           tfp.layers.util.default_mean_field_normal_fn(), 
                      # "bias_prior_fn":               default_mean_field_normal_fn(is_singular=True), 

#_V18 = BNN_v18 benutzt "kernel_posterior_fn":         tfp.layers.util.default_mean_field_normal_fn(loc_initializer=tf1.initializers.random_normal(mean=0.,stddev=0.), untransformed_scale_initializer=tf1.initializers.random_normal(mean=0.541324854612918, stddev=0.)), #mean is 0.541324854612918 so that after softplus transformation scale = 0.9999999
                      # "kernel_prior_fn":             default_mean_field_normal_fn(loc_initializer=tf1.initializers.random_normal(mean=0.,stddev=0.), is_singular=True),
                      # "bias_posterior_fn":           tfp.layers.util.default_mean_field_normal_fn(loc_initializer=tf1.initializers.random_normal(mean=0.,stddev=0.), untransformed_scale_initializer=tf1.initializers.random_normal(mean=0.541324854612918, stddev=0.)), 
                      # "bias_prior_fn":               default_mean_field_normal_fn(loc_initializer=tf1.initializers.random_normal(mean=0.,stddev=0.), is_singular=True), 


name=BNN_v18
output1=Flipout_QT_BNN_training_
output2=Flipout_BNN_training_


epochs=4000
iteration=100

cd /home/ycung/Desktop/DRACO-MLfoy/train_scripts/
layers=("300,300" "300,300,300")

for i in "${!layers[@]}"; do
    python train_template_bnn_denseflipout_test.py -o $output1"${layers[$i]}"_v18 -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs -q --layers ${layers[$i]}
    python train_template_bnn_denseflipout_test.py -o $output2"${layers[$i]}"_v18 -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs --layers ${layers[$i]}
done

for ((i=350; i<550; i+=50)); do
    python train_template_bnn_denseflipout_test.py -o $output1"$i","$i"_v18 -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs -q --layers "$i","$i"
    python train_template_bnn_denseflipout_test.py -o $output2"$i","$i"_v18 -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs --layers "$i","$i"
    python train_template_bnn_denseflipout_test.py -o $output1"$i","$i","$i"_v18 -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs -q --layers "$i","$i","$i"
    python train_template_bnn_denseflipout_test.py -o $output2"$i","$i","$i"_v18 -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs --layers "$i","$i","$i"
done

for ((i=550; i<1050; i+=50)); do
    python train_template_bnn_denseflipout_test.py -o $output1"$i","$i"_v18 -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs -q --layers "$i","$i"
    python train_template_bnn_denseflipout_test.py -o $output2"$i","$i"_v18 -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e $epochs --layers "$i","$i"
done