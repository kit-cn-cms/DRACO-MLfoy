# global imports
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import sys
import tensorflow.keras.models as models
import tensorflow.keras.layers as layer
import tensorflow.keras.utils as utils
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Activation, Dropout, Flatten, Dense
import numpy as np

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)
sys.path.append(basedir + '/visualization')

# import class for CNN training
import DRACO_Frameworks.CNN.CNN as CNN
import DRACO_Frameworks.CNN.data_frame as df

# import utils for visualization of results
import prepareVisualizationData as vis

# option_handler
import optionHandler_cnn
options = optionHandler_cnn.optionHandler_cnn(sys.argv)
options.initArguments()


#===================================
# prepare samples and training class
#===================================

# load samples
input_samples = df.InputSamples(options.getInputDir(), options.getActivatedSamples(), options.getTestPercentage())

# define all samples
input_samples.addSample(options.getSampleName("ttH"), label = "ttH", normalization_weight = options.getNomWeight())
input_samples.addSample(options.getSampleName("ttbar")  , label = "ttbar"  , normalization_weight = options.getNomWeight())

if options.isBinary():
    input_samples.addBinaryLabel(options.getSignal(), options.getBinaryBkgTarget())

# initializing CNN training class
cnn = CNN.CNN(
    save_path       = options.getOutputDir(),
    input_samples   = input_samples,
    category_name   = options.getCategory(),
    train_variables = options.getTrainVariables(),
    # number of epochs
    train_epochs    = options.getTrainEpochs(),
    # metrics for evaluation (c.f. KERAS metrics)
    eval_metrics    = ["acc"],
    # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
    test_percentage = options.getTestPercentage(),
    # balance samples per epoch such that there amount of samples per category is roughly equal
    balanceSamples  = options.doBalanceSamples(),
    evenSel         = options.doEvenSelection(),
    norm_variables  = options.doNormVariables(),
    # phi_padding factor: a factor of 0.25 adds the upper quarter of the image to the button and vice versa. default 0 leaves the image unchanged
    phi_padding     = 0,
    # set threshold for normed input data to ...%. Every value greater will be set to 1.
    normed_to 	    = 0.95,
    pseudoData      = options.isPseudo())

# prepare visualization
if not options.getModel() == "noConv":
    visualizer = vis.visualizer(options.getInputDir(), options.getOutputDir(), cnn.data.input_shape[0:2], options.getPlotName(), options.getRotationName(), options.getTrainVariables(), options.getFilterNum(), options.getFilterSize(), options.getModel(), 0.95, options.isPseudo())


#================
# define CNN model
#================


# input
input_shape = Input(shape=cnn.data.input_shape)

if options.getModel() == 'noConv':

    merge = Flatten()(input_shape)

    # further layers
    finishing = Dense( 50, activation = "relu")(merge)
    finishing = Dropout(0.5)(finishing)

    # output layer
    out = Dense( cnn.data.n_output_neurons, activation = "sigmoid")(finishing)

else:

    # filter layers
    towers = []
    for column in options.getFilterSize():
        tower = Conv2D(options.getFilterNum(), (column[0], column[0]), padding = 'same', activation = 'linear')(input_shape)
        for i in range(len(column)-1):
            tower = Conv2D(options.getFilterNum(), (column[i+1], column[i+1]), padding = 'same', activation = 'linear')(tower)
        towers.append(tower)
        
    # merging layers
    if len(options.getFilterSize()) > 1:
        merge = Concatenate()(towers)
        merge = Flatten()(merge)
    else:
        merge = Flatten()(towers[0])

    if options.getModel() == 'basic':

        # further layers
        finishing = Dense( 50, activation = "relu")(merge)
        finishing = Dropout(0.5)(finishing)

        # output layer
        out = Dense( cnn.data.n_output_neurons, activation = "sigmoid")(finishing)

    elif options.getModel() == 'reduced':

        # output layer
        out = Dense( cnn.data.n_output_neurons, activation = "sigmoid")(merge)

    elif options.getModel() == 'reduced_untr':

        # output layer
        out = Dense( cnn.data.n_output_neurons, activation = "sigmoid", trainable = False)(merge)

    else:
        print 'model type', options.getModel(), 'is not supported!'

# create model
model = Model(input_shape, out)

print("number of free parameters: "+str(model.count_params()))


#================
# build CNN model
#================

cnn.build_model(model=model) # for now, no cnn-netconfigs 
model.summary()

# get filter outputs before training
if not options.getModel() == "noConv":
    visualizer.readOutFilters(model.get_weights(), 'before')

#print model.get_weights()

# perform the training
cnn.train_model()

#get filter outputs after training
if not options.getModel() == "noConv":
    visualizer.readOutFilters(model.get_weights(), 'after')

#print model.get_weights()

# evalute the trained model
cnn.eval_model()

# plotting
#=========

if not options.getModel() == "noConv":
    visualizer.prepareImageData(), 

if options.doPlots():
    # plot the evaluation metrics
    # cnn.plot_metrics(privateWork = options.isPrivateWork())

    if options.isBinary():
        # plot output node
        bin_range = options.getBinaryBinRange()
        cnn.plot_binaryOutput(
            log          = options.doLogPlots(), 
            privateWork  = options.isPrivateWork(), 
            printROC     = options.doPrintROC(), 
            bin_range    = bin_range, 
            name         = options.getPlotName(),
            rotationMode = options.getRotationName(),
            sigScale     = options.getSignalScale())

