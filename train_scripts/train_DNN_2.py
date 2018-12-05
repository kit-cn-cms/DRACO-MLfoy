# global imports
import rootpy.plotting as rp
import numpy as np
import os
import sys

import keras
import keras.models as models
import keras.layers as layer
from keras import backend as K

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

import DRACO_Frameworks.DNN.DNN as DNN
import DRACO_Frameworks.DNN.variable_info as variable_info

category_vars = {
    "4j_ge3t": variable_info.variables_4j_3b,
    "5j_ge3t": variable_info.variables_5j_3b,
    "ge6j_ge3t": variable_info.variables_6j_3b
    }
categories = {
    "4j_ge3t":   "(N_Jets == 4 and N_BTagsM >= 3)",
    "5j_ge3t":   "(N_Jets == 5 and N_BTagsM >= 3)",
    "ge6j_ge3t": "(N_Jets >= 6 and N_BTagsM >= 3)",
    }

event_classes = ["ttHbb", "ttbb", "tt2b", "ttb", "ttcc", "ttlf"]

workpath = "/ceph/hluedemann/DRACO-MLfoy/workdir"

key = sys.argv[1]

inPath   = workpath + "/train_samples/AachenDNN_files"
savepath = workpath + "/Version2_DNN_"+str(key)+"/"




dnn = DNN.DNN(
    in_path         = inPath,
    save_path       = savepath,
    event_classes   = event_classes,
    event_category  = categories[key],
    train_variables = category_vars[key],
    train_epochs    = 500,
    early_stopping  = 20,
    eval_metrics    = ["acc"])

dropout                     = dnn.architecture["Dropout"]
batchNorm                   = dnn.architecture["batchNorm"]
activation_function         = dnn.architecture["activation_function"]
l2_regularization_beta      = dnn.architecture["L2_Norm"]
number_of_input_neurons = dnn.data.n_input_neurons
number_of_neurons_per_layer = dnn.architecture["prenet_layer"]

# Build model

#K.set_learning_phase(True)

# define model
model = models.Sequential()
# add input layer
model.add(layer.Dense(
            200,
            input_dim = number_of_input_neurons,
            activation = "elu",
            kernel_regularizer = keras.regularizers.l2(l2_regularization_beta)))

# loop over all dens layers
model.add(layer.Dense(
        100,
        activation = "elu",
        kernel_regularizer = keras.regularizers.l2(l2_regularization_beta)))
model.add(layer.Dropout(dropout))

model.add(layer.Dense(
        100,
        activation = "elu",
        kernel_regularizer = keras.regularizers.l2(l2_regularization_beta)))
model.add(layer.Dropout(dropout))


# create output layer
model.add(layer.Dense(dnn.data.n_output_neurons,
            activation = "softmax",
            kernel_regularizer = keras.regularizers.l2(l2_regularization_beta)))


dnn.build_model(model)
dnn.train_model()
dnn.eval_model()
dnn.plot_metrics()
#dnn.plot_input_output_correlation()

dnn.plot_confusion_matrix()
