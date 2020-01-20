# global imports
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import sys

# option handler
import optionHandler
options = optionHandler.optionHandler(sys.argv)

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

# import class for DNN training
import DRACO_Frameworks.CNN.CNN as CNN
import DRACO_Frameworks.CNN.data_frame as df

import keras.models as models
import keras.layers as layer

options.initArguments()

# load samples
input_samples = df.InputSamples(options.getInputDirectory(), options.getActivatedSamples(), options.getTestPercentage())

# define all samples
input_samples.addSample(options.getDefaultName("ttH"), label = "ttH", normalization_weight = options.getNomWeight())
input_samples.addSample(options.getDefaultName("ttbar")  , label = "ttbar"  , normalization_weight = options.getNomWeight())

if options.isBinary():
    input_samples.addBinaryLabel(options.getSignal(), options.getBinaryBkgTarget())

# initializing DNN training class
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
    phi_padding     = 0.25)

#================
#define CNN model
#================


''' #FROM HAUKE
model = models.Sequential()
#first layer
model.add(
    layer.Conv2D( 32, kernel_size = (4,4), activation = "sigmoid", padding = "same",
    input_shape = cnn.data.input_shape ))
model.add(
    layer.AveragePooling2D( pool_size = (4,4), padding = "same" ))
model.add(
    layer.Dropout(0.2))

# second layer
model.add(
    layer.Conv2D( 64, kernel_size = (4,4), activation = "sigmoid", padding = "same"))
model.add(
    layer.AveragePooling2D( pool_size = (4,4), padding = "same" ))
model.add(
    layer.Dropout(0.2))

# first dense layer
model.add(
    layer.Flatten())
model.add(
    layer.Dense( 128, activation = "sigmoid" ))
model.add(
    layer.Dropout(0.5))

#second dense layer
model.add(
    layer.Dense(128, activation = "sigmoid" ))
model.add(
    layer.Dropout(0.5))

#third dense layer
model.add(
    layer.Dense(128, activation = "sigmoid" ))
model.add(
    layer.Dropout(0.5))

#output layer
model.add(
    layer.Dense( cnn.data.n_output_neurons, activation = "sigmoid" ))
'''

model = models.Sequential()
#first layer
model.add(
    layer.Conv2D( 8, kernel_size = (4,4), activation = "linear", padding = "same",
    input_shape = cnn.data.input_shape ))

# first dense layer
model.add(
    layer.Flatten())
model.add(
    layer.Dense( 50, activation = "relu" ))
model.add(
    layer.Dropout(0.5))

#output layer
model.add(
    layer.Dense( cnn.data.n_output_neurons, activation = "sigmoid" ))

print("number of free parameters: "+str(model.count_params()))

#================
# build CNN model
#================
cnn.build_model(model=model) #for now, no cnn-netconfigs 

model.summary()

# perform the training
cnn.train_model()

# evalute the trained model
cnn.eval_model()

# save information
#cnn.save_model(sys.argv, filedir)

# save and print variable ranking
#cnn.get_input_weights()

# plotting
if options.doPlots():
    # plot the evaluation metrics
    cnn.plot_metrics(privateWork = options.isPrivateWork())

    if options.isBinary():
        # plot output node
        bin_range = options.getBinaryBinRange()
        cnn.plot_binaryOutput(
            log         = options.doLogPlots(), 
            privateWork = options.isPrivateWork(), 
            printROC    = options.doPrintROC(), 
            bin_range   = bin_range, 
            name        = options.getName(),
            sigScale    = options.getSignalScale())
    else:
        # plot the confusion matrix
        cnn.plot_confusionMatrix(
            privateWork = options.isPrivateWork(),
            printROC    = options.doPrintROC())

        # plot the output discriminators
        cnn.plot_discriminators(
            log                 = options.doLogPlots(),
            signal_class        = options.getSignal(),
            privateWork         = options.isPrivateWork(),
            printROC            = options.doPrintROC(),
            sigScale            = options.getSignalScale())

        # plot the output nodes
        cnn.plot_outputNodes(
            log                 = options.doLogPlots(),
            signal_class        = options.getSignal(),
            privateWork         = options.isPrivateWork(),
            printROC            = options.doPrintROC(),
            sigScale            = options.getSignalScale())

        # plot event yields
        cnn.plot_eventYields(
            log                 = options.doLogPlots(),
            signal_class        = options.getSignal(),
            privateWork         = options.isPrivateWork(),
            sigScale            = options.getSignalScale())

        # plot closure test
        cnn.plot_closureTest(
            log                 = options.doLogPlots(),
            signal_class        = options.getSignal(),
            privateWork         = options.isPrivateWork())
