# global imports
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os, numpy as np
import sys
import json
import pandas as pd

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

#with open("ttZ_flags.py") as config_file:
#    config = json.load(config_file)

options.initArguments()

# load samples
input_samples = df.InputSamples(options.getInputDirectory(), options.getActivatedSamples(), options.getTestPercentage(), options.getAddSampleSuffix())

weight_expr = 'x.Weight_XS * x.Weight_btagSF * x.Weight_GEN_nom * x.lumiWeight'
# define all samples
input_samples.addSample(options.getDefaultName("ttH")  , label = "ttH"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
input_samples.addSample(options.getDefaultName("ttHbb")  , label = "ttHbb"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
input_samples.addSample(options.getDefaultName("ttHnonbb")  , label = "ttHnonbb"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
input_samples.addSample(options.getDefaultName("ttZ")  , label = "ttZ"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
input_samples.addSample(options.getDefaultName("ttZbb")  , label = "ttZbb"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
input_samples.addSample(options.getDefaultName("ttZnonbb")  , label = "ttZnonbb"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
input_samples.addSample(options.getDefaultName("ttX")  , label = "ttX"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
input_samples.addSample(options.getDefaultName("ttXbb")  , label = "ttXbb"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
input_samples.addSample(options.getDefaultName("ttXnonbb")  , label = "ttXnonbb"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )

input_samples.addSample(options.getDefaultName("ttbar") , label = "ttbar" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
input_samples.addSample(options.getDefaultName("ttnonbb") , label = "ttnonbb" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
input_samples.addSample(options.getDefaultName("ttmb") , label = "ttmb" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
input_samples.addSample(options.getDefaultName("ttbb") , label = "ttbb" , normalization_weight = options.getNomWeight()*35.8*0.44, total_weight_expr = weight_expr )
input_samples.addSample(options.getDefaultName("tt2b") , label = "tt2b" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
input_samples.addSample(options.getDefaultName("ttb")  , label = "ttb"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
input_samples.addSample(options.getDefaultName("ttcc") , label = "ttcc" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
input_samples.addSample(options.getDefaultName("ttlf") , label = "ttlf" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )

input_samples.addSample(options.getDefaultName("tHq") , label = "tHq" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
input_samples.addSample(options.getDefaultName("tHW") , label = "tHW" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )

input_samples.addSample(options.getDefaultName("sig") ,  label = "sig" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
input_samples.addSample(options.getDefaultName("bkg") ,  label = "bkg" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )

# additional samples for adversary training
if options.isAdversary():
    input_samples.addSample(options.getAddSampleName("ttmb"), label = "ttmb"+options.getAddSampleSuffix(), normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
    input_samples.addSample(options.getAddSampleName("ttbb"), label = "ttbb"+options.getAddSampleSuffix(), normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
    input_samples.addSample(options.getAddSampleName("tt2b"), label = "tt2b"+options.getAddSampleSuffix(), normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
    input_samples.addSample(options.getAddSampleName("ttb") , label = "ttb"+options.getAddSampleSuffix() , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )

if options.isBinary():
    input_samples.addBinaryLabel(options.getSignal(), options.getBinaryBkgTarget())


pre_var=[]
with open("../variable_sets/flags.py", "r") as flag_list:
    file_content = flag_list.readlines()
    for line in file_content:
        current_place = line[:-1]
        pre_var.append(current_place)

# Pre net
dnn_pre=DNN.DNN(
        save_path       = options.getOutputDir() + "_pre-net", # set extra saving path
        input_samples   = input_samples,
        category_name   = options.getCategory(),                # eg. ge4j_ge4t
        train_variables = options.getTrainVariables(), #pre_var, 
        # number of epochs
        train_epochs    = options.getTrainEpochs(),
        # metrics for evaluation (c.f. KERAS metrics)
        eval_metrics    = ["acc"],
        # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
        test_percentage = options.getTestPercentage(),
        # balance samples per epoch such that there amount of samples per category is roughly equal
        balanceSamples  = options.doBalanceSamples(),
        evenSel         = options.doEvenSelection(),
        norm_variables  = False) #options.doNormVariables())

if not options.isAdversary():
    # initializing DNN training class
    dnn = DNN.DNN(
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
        norm_variables  = False, #options.doNormVariables(),
        shuffle_seed = dnn_pre.data.shuffleSeed)
else:
    import DRACO_Frameworks.DNN.CAN as CAN
    # initializing CAN training class
    dnn = CAN.CAN(
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
        addSampleSuffix = options.getAddSampleSuffix(),
        shuffle_seed = dnn_pre.data.shuffleSeed)


print dnn_pre.data.get_test_data()

# build DNN model
from net_configs import config_dict
dnn_pre.data.n_output_neurons = len(pre_var)        # set output dimension
dnn_pre.build_model(config=config_dict["dnn_ttZ_prenet"], penalty=options.getPenalty())
#print(dnn.data.df_train)
#print( dnn_pre.data.df_train["train_weight"])
#print(dnn_pre.data.get_train_labels())
flags = dnn_pre.data.df_train[pre_var].values
print flags
print dnn_pre.data.df_train["flags_ft_has_z_b"]
# perform the training
dnn_pre.train_model_flags(flags=pre_var)
print(dnn_pre.model.predict(dnn_pre.data.get_train_data())[0:10,:])
print flags[0:10,:]
# evalute the trained model
dnn_pre.eval_model_flags(flags=pre_var)

'''
# get activated samples to name output of first net
activated_samples=options.getActivatedSamples().split(",")
print "activated_samples: ", activated_samples
output_names = []
for i in range(len(activated_samples)):
    output_names.append("output_" + activated_samples[i])
print output_names
'''
output_names=[]
for i in range(len(pre_var)):
    output_names.append(pre_var[i])
print output_names

# do predictions on all samples by first net which are to be passed on
train_predict=dnn_pre.model.predict(dnn_pre.data.get_train_data())
test_predict=dnn_pre.model.predict(dnn_pre.data.get_test_data())
# create pandas data frames of outputs
df_train_predict=pd.DataFrame(train_predict, columns=output_names)
df_test_predict=pd.DataFrame(test_predict, columns=output_names)


print ("Prediction of pre net: ")

print "train: ", np.shape(train_predict)
print "test: ", np.shape(test_predict)

dnn.data.df_train=dnn_pre.data.df_train.copy()
dnn.data.df_test=dnn_pre.data.df_test.copy()
# add variables to variable set of second net
for key in output_names:
    dnn.data.df_train[key]=df_train_predict[key].values
    dnn.data.df_test[key]=df_test_predict[key].values
#dnn.data.df_train['b']=train_predict[:,1]

# add new variables to list in order for get_train_data to work
print np.shape(dnn.data.train_variables)
dnn.data.train_variables.extend(output_names)
print np.shape(dnn.data.train_variables)
print dnn.data.train_variables
#dnn.data.train_variables.append('b')

print np.shape(dnn.data.get_train_data())

# adapt number of input neurons for some reason not automatically done
dnn.data.n_input_neurons = np.shape(dnn.data.get_train_data())[1]
print "number of input neurons: ", dnn.data.n_input_neurons

# build model
dnn.build_model(config=options.getNetConfig(), penalty=options.getPenalty())

# perform the training
dnn.train_model()
# evalute the trained model
dnn.eval_model()

'''
# pre net
# save information
dnn_pre.save_model(sys.argv, filedir, options.getNetConfigName(), get_gradients = options.doGradients())

# save and print variable ranking according to the input layer weights
dnn_pre.get_input_weights()

# save and print variable ranking according to all layer weights
dnn_pre.get_weights()

# variation plots
if options.doVariations():
    dnn_pre.get_variations(options.isBinary())

# plotting
if options.doPlots():
    # plot the evaluation metrics
    dnn_pre.plot_metrics(privateWork = options.isPrivateWork())

    if options.isBinary():
        # plot output node
        bin_range = options.getBinaryBinRange()
        dnn_pre.plot_binaryOutput(
            log         = options.doLogPlots(),
            privateWork = options.isPrivateWork(),
            printROC    = options.doPrintROC(),
            nbins       = 15,
            bin_range   = bin_range,
            name        = options.getName(),
            sigScale    = options.getSignalScale())
        if options.isAdversary():
            dnn_pre.plot_ttbbKS_binary(
                log                 = options.doLogPlots(),
                signal_class        = options.getSignal(),
                privateWork         = options.isPrivateWork())
    else:
        # plot the confusion matrix
        dnn_pre.plot_confusionMatrix(
            privateWork = options.isPrivateWork(),
            printROC    = options.doPrintROC())

        # plot the output discriminators
        dnn_pre.plot_discriminators(
            log                 = options.doLogPlots(),
            signal_class        = options.getSignal(),
            privateWork         = options.isPrivateWork(),
            printROC            = options.doPrintROC(),
            sigScale            = options.getSignalScale())

        # plot the output nodes
        dnn_pre.plot_outputNodes(
            log                 = options.doLogPlots(),
            signal_class        = options.getSignal(),
            privateWork         = options.isPrivateWork(),
            printROC            = options.doPrintROC(),
            sigScale            = options.getSignalScale())

        # plot event yields
        dnn_pre.plot_eventYields(
            log                 = options.doLogPlots(),
            signal_class        = options.getSignal(),
            privateWork         = options.isPrivateWork(),
            sigScale            = options.getSignalScale())

        # plot closure test
        dnn_pre.plot_closureTest(
            log                 = options.doLogPlots(),
            signal_class        = options.getSignal(),
            privateWork         = options.isPrivateWork())

        # plot ttbb KS test
        if options.isAdversary():
            dnn_pre.plot_ttbbKS(
                log                 = options.doLogPlots(),
                signal_class        = options.getSignal(),
                privateWork         = options.isPrivateWork())

if options.doGradients():
    dnn_pre.get_gradients(options.isBinary())

'''
# save information
dnn.save_model(sys.argv, filedir, options.getNetConfigName(), get_gradients = options.doGradients())

# save and print variable ranking according to the input layer weights
dnn.get_input_weights()

# save and print variable ranking according to all layer weights
dnn.get_weights()

# variation plots
if options.doVariations():
    dnn.get_variations(options.isBinary())

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
            nbins       = 15,
            bin_range   = bin_range,
            name        = options.getName(),
            sigScale    = options.getSignalScale())
        if options.isAdversary():
            dnn.plot_ttbbKS_binary(
                log                 = options.doLogPlots(),
                signal_class        = options.getSignal(),
                privateWork         = options.isPrivateWork())
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
            signal_class        = options.getSignal(),
            privateWork         = options.isPrivateWork())

        # plot ttbb KS test
        if options.isAdversary():
            dnn.plot_ttbbKS(
                log                 = options.doLogPlots(),
                signal_class        = options.getSignal(),
                privateWork         = options.isPrivateWork())
if options.doGradients():
    dnn.get_gradients(options.isBinary())
