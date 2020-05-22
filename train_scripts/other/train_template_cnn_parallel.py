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



input_shape = Input(shape=cnn.data.input_shape)

# filter layers
tower_1 = Conv2D(1, (1, 1), padding='same', activation='linear')(input_shape)
#tower_1 = Conv2D(1, (2, 2), padding='same', activation='linear')(tower_1)
#tower_1 = MaxPooling2D((1, 11), strides=(1, 1), padding='same')(tower_1)

tower_2 = Conv2D(1, (2, 2), padding='same', activation='linear')(input_shape)
tower_2 = Conv2D(1, (3, 3), padding='same', activation='linear')(tower_2)
#tower_2 = Conv2D(1, (5, 5), padding='same', activation='linear')(tower_2)
#tower_2 = MaxPooling2D((1, 9), strides=(1, 1), padding='same')(tower_2)

tower_3 = Conv2D(1, (4, 4), padding='same', activation='linear')(input_shape)
tower_3 = Conv2D(1, (5, 5), padding='same', activation='linear')(tower_3)
tower_3 = Conv2D(1, (6, 6), padding='same', activation='linear')(tower_3)
#tower_3 = MaxPooling2D((1, 6), strides=(1, 1), padding='same')(tower_3)


# merge layers
merge = Concatenate()([tower_1, tower_2, tower_3])
merge = Flatten()(merge)

# further layers
#finishing = Dense( 50, activation = "relu")(merge)
#finishing = Dropout(0.5)(finishing)

# output layer
out = Dense( cnn.data.n_output_neurons, activation = "sigmoid")(merge)

model = Model(input_shape, out)

#================
# build CNN model
#================

cnn.build_model(model=model) # for now, no cnn-netconfigs 
model.summary()

print model.get_weights()
print np.asarray(model.get_weights()).shape
for i in range(12):
    print np.asarray(model.get_weights()[i]).shape
'''
# get filter outputs before training

vis.readOutFilters(model.get_weights()[0], options.getOutputDir(), '1_before', options.getPlotName())
vis.readOutFilters(model.get_weights()[2], options.getOutputDir(), '4_before', options.getPlotName())
vis.readOutFilters(model.get_weights()[4], options.getOutputDir(), '7_before', options.getPlotName())

#for i in model.get_weights():
 #   print i.shape
#print model.get_weights()

# perform the training
cnn.train_model()

#get filter outputs after training

vis.readOutFilters(model.get_weights()[0], options.getOutputDir(), '1_after', options.getPlotName())
vis.readOutFilters(model.get_weights()[2], options.getOutputDir(), '4_after', options.getPlotName())
vis.readOutFilters(model.get_weights()[4], options.getOutputDir(), '7_after', options.getPlotName())

#print model.get_weights()

# evalute the trained model
cnn.eval_model()


#=========
# plotting
#=========

#vis.prepare_feature_maps(path_to_filter_data = options.getOutputDir(), 
		    #path_to_image_data = options.getInputDir() + '/' + options.getSampleName('ttH'), 
		    #filename = options.getPlotName(), 
		    #channels = options.getTrainVariables(),
		    #image_index = 5) # doesn't matter

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
'''
