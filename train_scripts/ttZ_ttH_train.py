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
import DRACO_Frameworks.DNN.DNN as DNN
import DRACO_Frameworks.DNN.data_frame as df

# some more imports needed
import pandas as pd
import keras

options.initArguments()

# load samples
input_samples = df.InputSamples(options.getInputDirectory(), options.getActivatedSamples(), options.getTestPercentage())

# define all samples
input_samples.addSample(options.getDefaultName("ttZ"),  label = "ttZ",  normalization_weight = options.getNomWeight())
input_samples.addSample(options.getDefaultName("ttH"),  label = "ttH",  normalization_weight = options.getNomWeight())
#input_samples.addSample(options.getDefaultName("ttbb"), label = "ttbb", normalization_weight = options.getNomWeight())
#input_samples.addSample(options.getDefaultName("tthf"), label = "tthf", normalization_weight = options.getNomWeight())
#input_samples.addSample(options.getDefaultName("ttcc"), label = "ttcc", normalization_weight = options.getNomWeight())
#input_samples.addSample(options.getDefaultName("ttlf"), label = "ttlf", normalization_weight = options.getNomWeight())

def __importVariableSelection(variableSelection):
    if not os.path.isabs(variableSelection):
        sys.path.append(basedir+"/variable_sets/")
        variable_set = __import__(variableSelection)
    elif os.path.exists(options.variableSelection):
        variable_set = __import__(variableSelection)
    else:
        sys.exit("ERROR: Variable Selection File does not exist!")
    return variable_set
    


NoRecoVars = __importVariableSelection("allVars_noReco_S01")
RecoVarsOnly = __importVariableSelection("RecoVarsOnly_S01")





#if options.isBinary():
    #input_samples.addBinaryLabel(options.getSignal(), options.getBinaryBkgTarget())

# initializing DNN training class, only use this for getting the JT string
dnn = DNN.DNN(
    save_path       = options.getOutputDir(),
    input_samples   = input_samples,
    category_name   = options.getCategory(),
    train_variables = options.getTrainVariables(),
    # number of epochs
    train_epochs    = options.getTrainEpochs(),
    # metrics for evaluation (c.f. KERAS metrics)
    eval_metrics    = ["acc","mean_squared_error","mean_absolute_error"],
    # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
    test_percentage = options.getTestPercentage(),
    # balance samples per epoch such that there amount of samples per category is roughly equal
    balanceSamples  = options.doBalanceSamples(),
    evenSel         = options.doEvenSelection(),
    norm_variables  = options.doNormVariables())

#generate that JT string to get the new varibale selection
trainVarsCombined = RecoVarsOnly[dnn.category_label] + NoRecoVars[dnn.category_label]

#now load the DNN again with the variable selection of both sets combined
dnn = DNN.DNN(
    save_path       = options.getOutputDir(),
    input_samples   = input_samples,
    category_name   = options.getCategory(),
    train_variables = trainVarsCombined,
    # number of epochs
    train_epochs    = options.getTrainEpochs(),
    # metrics for evaluation (c.f. KERAS metrics)
    eval_metrics    = ["acc","mean_squared_error","mean_absolute_error"],
    # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
    test_percentage = options.getTestPercentage(),
    # balance samples per epoch such that there amount of samples per category is roughly equal
    balanceSamples  = options.doBalanceSamples(),
    evenSel         = options.doEvenSelection(),
    norm_variables  = options.doNormVariables())

# build DNN model
#dnn.build_model(options.getNetConfig())
def build_branched_model(dnn, NoRecoVars, RecoVarsOnly):
    ''' build default straight forward DNN from architecture dictionary '''
    

    # infer number of input neurons from number of train variables
    NumInputNeuronsNoReco = len(NoRecoVars[dnn.category_label])
    NumInputNeuronsRecoOnly = len(RecoVarsOnly[dnn.category_label])
    
    # get all the architecture settings needed to build model
    number_of_neurons_per_layer = dnn.architecture["layers"]
    dropout                     = dnn.architecture["Dropout"]
    activation_function         = dnn.architecture["activation_function"]
    if activation_function == "leakyrelu":
        activation_function = "linear"
    l2_regularization_beta      = dnn.architecture["L2_Norm"]
    output_activation           = dnn.architecture["output_activation"]

    # define main net named X training without Reco
    NoRecoInputs = keras.layers.Input(
        shape = (NumInputNeuronsNoReco,),
        name  = "NoReco_Input")
    X = noRecoInputs
    #self.layer_list = [X]

    # loop over dense layers
    for iLayer, nNeurons in enumerate(number_of_neurons_per_layer):
        X = keras.layers.Dense(
            units               = nNeurons,
            activation          = activation_function,
            kernel_regularizer  = keras.regularizers.l2(l2_regularization_beta),
            name                = "X_DenseLayer_"+str(iLayer)
            )(X)

        if self.architecture["activation_function"] == "leakyrelu":
            X = keras.layers.LeakyReLU(alpha=0.1)(X)

        # add dropout percentage to layer if activated
        if not dropout == 0:
            X = keras.layers.Dropout(dropout, name = "DropoutLayer_"+str(iLayer))(X)
    
    #repeat the procedure for Y: side net with RecoVars
    RecoVarsInputs = keras.layers.Input(
        shape = (NumInputNeuronsRecoOnly,),
        name  = "RecoOnly_Input")
    Y = RecoVarsInputs
    #self.layer_list = [X]

    # loop over dense layers
    for iLayer, nNeurons in enumerate(number_of_neurons_per_layer):
        Y = keras.layers.Dense(
            units               = nNeurons,
            activation          = activation_function,
            kernel_regularizer  = keras.regularizers.l2(l2_regularization_beta),
            name                = "Y_DenseLayer_"+str(iLayer)
            )(X)

        if self.architecture["activation_function"] == "leakyrelu":
            Y = keras.layers.LeakyReLU(alpha=0.1)(X)

        # add dropout percentage to layer if activated
        if not dropout == 0:
            Y = keras.layers.Dropout(dropout, name = "DropoutLayer_"+str(iLayer))(X)
    
    
    
    # generate side output layer
    sideOutput = keras.layers.Dense(
        units               = dnn.data.n_output_neurons,
        activation          = output_activation.lower(),
        kernel_regularizer  = keras.regularizers.l2(l2_regularization_beta),
        name                = "RecoVars_output"
        )(Y)
        
    #generate main output layer
    mainOutput = keras.layers.Dense(
        units               = self.data.n_output_neurons,
        activation          = output_activation.lower(),
        kernel_regularizer  = keras.regularizers.l2(l2_regularization_beta),
        name                = "NoReco_output"
        )(X)
    
    combinedLayers = keras.layers.Average([mainOutput,sideOutput], 
        name =              = "CombinedOutput"
        )
    
    #CombinedOutput = keras.layers.Dense(
        #units               = self.data.n_output_neurons,
        #activation          = output_activation.lower(),
        #kernel_regularizer  = keras.regularizers.l2(l2_regularization_beta),
        #name                = "main_output"
        #)(combinedLayers)
    
    
    # define model
    model = models.Model(inputs = [noRecoInputs, RecoVarsInputs], outputs = [mainOutput,sideOutput,combinedLayers])
    model.summary()

    return model

BranchedModel = build_branched_model(dnn,NoRecoVars, RecoVarsOnly) 

#now we can again use the DNN class methods to compile
dnn.build_model(model=BranchedModel)

# for training we need another setup than for the DNN class
dnn.trained_model = dnn.model.fit(
    {'NoReco_Input': dnn.data.df_train[ NoRecoVars[dnn.category_label]].values, 
    'RecoVars_Input': dnn.data.df_train[ RecoVarsOnly[dnn.category_label]].values,
    'NoReco_Output': dnn.data.get_train_labels(),
    'RecoVars_Output': dnn.data.get_train_labels(),
    'CombinedOutput': dnn.data.get_train_labels()}
    batch_size          = dnn.architecture["batch_size"],
    epochs              = dnn.train_epochs,
    shuffle             = True,
    callbacks           = callbacks,
    validation_split    = 0.25,
    sample_weight       = dnn.data.get_train_weights())

# evalute the trained model
#dnn.eval_model()
self.model_history = self.trained_model.history


dnn.model_prediction_vector = dnn.model.predict({'NoReco_Input': dnn.data.df_test[ NoRecoVars[dnn.category_label]].values, 
    'RecoVars_Input': dnn.data.df_test[ RecoVarsOnly[dnn.category_label]].values}
    )
print(dnn.model_prediction_vector)

dnn.plot_metrics(privateWork = options.isPrivateWork())


# save information
dnn.save_model(sys.argv, filedir)
'''
# save and print variable ranking
dnn.get_input_weights()

# save and print variable ranking according to all layer weights
dnn.get_propagated_weights()

dnn.get_variations()

# plotting 
if options.doPlots():
    # plot the evaluation metrics
    dnn.plot_metrics(privateWork = options.isPrivateWork())

    if options.isBinary():
        # plot output node
        bin_range = options.getBinaryBinRange()
        dnn.plot_binaryOutput(
            log         = options.doLogPlots(), 
            privateWork = options.isPrivateWork(), 
            printROC    = options.doPrintROC(), 
            bin_range   = bin_range, 
            name        = options.getName())
    else:
        # plot the confusion matrix
        dnn.plot_confusionMatrix(
            privateWork = options.isPrivateWork(), 
            printROC    = options.doPrintROC())

        # plot the output discriminators
        dnn.plot_discriminators(
            log                 = options.doLogPlots(), 
            signal_class        = options.getSignal(),         
            privateWork         = options.isPrivateWork(), 
            printROC            = options.doPrintROC(),    
            sigScale            = options.getSignalScale())

        # plot the output nodes
        dnn.plot_outputNodes(
            log                 = options.doLogPlots(), 
            signal_class        = options.getSignal(), 
            privateWork         = options.isPrivateWork(), 
            printROC            = options.doPrintROC(), 
            sigScale            = options.getSignalScale())

        # plot event yields
        dnn.plot_eventYields(
            log                 = options.doLogPlots(), 
            signal_class        = options.getSignal(), 
            privateWork         = options.isPrivateWork(), 
            sigScale            = options.getSignalScale())

        # plot closure test
        dnn.plot_closureTest(
            log                 = options.doLogPlots(), 
            #signal_class        = options.getSignal(),
            signal_class        = ["ttZ"],
            privateWork         = options.isPrivateWork())
'''

