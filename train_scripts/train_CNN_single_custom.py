# global imports
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import keras.layers as layer
import keras.models as models


# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)
#import DRACO_Frameworks.CNN.data_frame as data_frame
import DRACO_Frameworks.CNN.CNN_single_outputs as CNN


inPath = "/storage/c/vanderlinden/DRACO-MLfoy/workdir/train_samples/base_train_set"

cnn = CNN.CNN(
    in_path         = inPath,
    save_path       = basedir+"/workdir/basic_cnn_single",
    class_label     = "nJets",
    batch_size      = 128,
    train_epochs    = 15,
    optimizer       = "adam",
    loss_function   = "mean_absolute_error",
    eval_metrics    = ["mean_squared_error", "acc"] )

cnn.load_datasets()

#------------------------------------------------
# build model
model = models.Sequential()
#first layer
model.add(
    layer.Conv2D( 32, kernel_size = (4,4), activation = "sigmoid", padding = "same",
    input_shape = cnn.train_data.input_shape ))
model.add(
    layer.AveragePooling2D( pool_size = (4,4), padding = "same" ))
model.add(
    layer.Dropout(0.5))

# second layer
model.add(
    layer.Conv2D( 64, kernel_size = (4,4), activation = "sigmoid", padding = "same"))
model.add(
    layer.AveragePooling2D( pool_size = (4,4), padding = "same" ))
model.add(
    layer.Dropout(0.5))

# first dense layer
model.add(
    layer.Flatten())
model.add(
    layer.Dense( 128, activation = "relu" ))
model.add(
    layer.Dropout(0.5))

#second dense layer
model.add(
    layer.Dense(128, activation = "relu" ))
model.add(
    layer.Dropout(0.5))

#output layer
model.add(
    layer.Dense( cnn.num_classes, activation = "relu" ))
# -----------------------------------------------

cnn.build_model(model) #building custom model
cnn.train_model()
cnn.eval_model()

# evaluate stuff
cnn.print_classification_examples()
cnn.plot_metrics()
cnn.plot_discriminators()
cnn.plot_confusion_matrix()






