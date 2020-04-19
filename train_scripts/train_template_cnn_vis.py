# global imports
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import sys
import tensorflow.keras.models as models
import tensorflow.keras.layers as layer
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
import prepare_feature_map as vis

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
    normed_to 	    = 0.95)


#================
# define CNN model
#================

model = models.Sequential()

# first layer
if not options.getModel() == "no_Conv2D":
    model.add(
        layer.Conv2D(options.getFilterNum(), kernel_size = options.getFilterSize(), strides=1, activation = "linear", padding = "same",
        input_shape = cnn.data.input_shape ))
    model.add(
        layer.Flatten())
else:
    model.add(
        layer.Flatten(input_shape = cnn.data.input_shape))

# further layers
if not options.getModel() == "only_Conv2D":
    model.add(
        layer.Dense( 50, activation = "relu"))
    #model.add(
     #   layer.Dropout(0.5))

# output layer
if options.getModel() == "only_Conv2D":
    model.add(
        layer.Dense( cnn.data.n_output_neurons, activation = "sigmoid", trainable = False ))
    model.set_weights([model.get_weights()[0], model.get_weights()[1], np.ones((1320, 1)), model.get_weights()[3]])
else:
    model.add(
        layer.Dense( cnn.data.n_output_neurons, activation = "sigmoid" ))

print("number of free parameters: "+str(model.count_params()))


#================
# build CNN model
#================

cnn.build_model(model=model) # for now, no cnn-netconfigs 
model.summary()

# get filter outputs before training
if not options.getModel() == "no_Conv2D":
    vis.readOutFilters(model.get_weights()[0], options.getOutputDir(), 'before', options.getPlotName())

print model.get_weights()

# perform the training
cnn.train_model()

#get filter outputs after training
if not options.getModel() == "no_Conv2D":
    vis.readOutFilters(model.get_weights()[0], options.getOutputDir(), 'after', options.getPlotName())

print model.get_weights()

# evalute the trained model
cnn.eval_model()

#=========
# plotting
#=========

if not options.getModel() == "no_Conv2D":
    vis.prepare_feature_maps(path_to_filter_data = options.getOutputDir(), 
			path_to_image_data = options.getInputDir() + '/' + options.getSampleName('ttH'), 
			filename = options.getPlotName(), 
			channels = options.getTrainVariables(),
			image_index = 5) # doesn't matter

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

