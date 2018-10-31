'''from __future__ import absolute_import, division, print_function

import sys
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.backend.tensorflow_backend import set_session

class NeuralNetworkTrainerAachen(object):

	def __init__(self):

		# Limit gpu usage
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

    def loadData(self,
    			 path_to_training_dataset,
    			 path_to_validation_dataset,
    			 path_to_test_dataset=None):

    	self.training_data_set   = NNFlowDataFrame(path_to_training_data_set)
        self.validation_data_set = NNFlowDataFrame(path_to_validation_data_set)
        self.test_data_set = NNFlowDataFrame(path_to_test_data_set)

        self.number_of_input_neurons  = training_data_set.get_number_of_input_neurons()
        self.number_of_output_neurons = training_data_set.get_number_of_output_neurons()

        # Get weights and labels
        self.train_label, self.train_weights = self.training_data_set.get_labels_event_weights()
        self.vali_label, self.vali_weights= self.validation_data_set.get_labels_event_weights()
        self.test_label, self.test_weights= self.test_data_set.get_labels_event_weights()

        # Get scaled data. data is scaled between 0 and 1
        self.train_data =training_data_set.get_scaled_data()
        self.vali_data =validation_data_set.get_scaled_data()
        self.test_data =test_data_set.get_scaled_data()

    def build_first_model(self,
    			   number_of_input_neurons,
    			   number_of_neurons_per_layer,
    			   activation_function,
    			   dropout,
    			   l2_regularization_beta,
    			   ):

    	self.Inputs = keras,layers.Input(shape=(number_of_input_neurons,))
    	X=self.Inputs 

    	for i, nNeurons in enumerate(number_of_neurons_per_layer):
    		X = keras.layers.Dense(nNeurons,
    							   activation=activation_function,
    							   kernel_regularizer=keras.regularizers.l2(l2_regularization_beta))(X)
   
    	Inputs= keras.layers.Input(shape=(number_of_input_neurons,))
        X= Inputs
        if dropout_keep_probability != 1:
            X = keras.layers.Dropout(dropout_keep_probability)(X)
        for i in range(len(hidden_layers) - 1):
            X =keras.layers.Dense(hidden_layers[i],
                                activation=activation_function_name,
                                kernel_regularizer=keras.regularizers.l2(l2_regularization_beta))(X)
            if dropout_keep_probability != 1:
                X= keras.layers.Dropout(dropout_keep_probability)(X)
        # Build last layer, regression only works with 1 parameter
        X= keras.layers.Dense(1,
                                         activation='sigmoid',
                                         kernel_regularizer=keras.regularizers.l2(l2_regularization_beta))(X)

        class_model = keras.models.Model(inputs=[Inputs], outputs=[X])

        adv_layers = class_model(Inputs)
        adv_layers = keras.layers.Dense(100,
                                activation=activation_function_name,
                                kernel_regularizer=keras.regularizers.l2(l2_regularization_beta))(adv_layers)
        adv_layers = keras.layers.Dense(100,
                                activation=activation_function_name,
                                kernel_regularizer=keras.regularizers.l2(l2_regularization_beta))(adv_layers)
        adv_layers = keras.layers.Dense(1,
                                activation='sigmoid',
                                kernel_regularizer=keras.regularizers.l2(l2_regularization_beta))(adv_layers)
        adv_model = keras.models.Model(inputs=[Inputs], outputs = [adv_layers])

        # Create the optimier. Here the Adamoptimizer is used but can be changed to a different optimizer
        optimizer = tf.train.AdamOptimizer(1e-5)

	


    def build_second_model(self,
    			   number_of_input_neurons,
    			   number_of_neurons_per_layer,
    			   activation_function,
    			   droput,
    			   regularisation,
    			   ):

    def train(self,
    		  nEpochs,
    		  earlyStopIntervall)'''



from __future__ import absolute_import, division, print_function

import sys
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.backend.tensorflow_backend import set_session
import keras.backend 


# Limit gpu usage
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# Parameters
number_of_input_neurons = 1
number_of_neurons_per_layer = [10,10]
activation_function = 'relu'
l2_regularization_beta = 0.0001
dropout = 0.7

#Load Data

test = np.arange(10)
test_label = np.arange(10)

# Define first model
Inputs = keras.layers.Input(shape=(number_of_input_neurons,))
X=Inputs 
layers_list = [X]
for i, nNeurons in enumerate(number_of_neurons_per_layer):
    Dense = keras.layers.Dense(nNeurons, activation=activation_function,kernel_regularizer=keras.regularizers.l2(l2_regularization_beta), name="Dense_"+str(i))(X)
    layers_list.append(Dense)
    if dropout != 1:
            X= keras.layers.Dropout(dropout)(Dense)
X= keras.layers.Dense(1,
                      activation='sigmoid',
                      kernel_regularizer=keras.regularizers.l2(l2_regularization_beta))(X)
layers_list.append(X)

first_model = keras.models.Model(inputs=[Inputs], outputs=[X])
first_model.summary()
first_model.compile(loss='Huber',
              optimizer='adam',
             metrics=['accuracy'])
#Train first model
first_model.fit(test,test_label,nb_epoch=1,verbose=1)


# Make Parameters of first model untrainable
for layer in first_model.layers:
    layer.trainable = False

# Create Input/conc layer for second NN
conc_layer = keras.layers.concatenate(layers_list,axis=-1)

#Define second NN
Y=conc_layer

for i, nNeurons in enumerate(number_of_neurons_per_layer):
    Y = keras.layers.Dense(nNeurons, activation=activation_function,kernel_regularizer=keras.regularizers.l2(l2_regularization_beta), name="Dense_new_"+str(i))(Y)

    if dropout != 1:
            Y= keras.layers.Dropout(dropout)(Y)
Y= keras.layers.Dense(1,
                      activation='sigmoid',
                      kernel_regularizer=keras.regularizers.l2(l2_regularization_beta))(Y)
first_model.trainable = False
second_model = keras.models.Model(inputs=[Inputs], outputs=[Y])
second_model.summary()

#Train second NN