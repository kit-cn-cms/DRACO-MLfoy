# global imports
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import sys
import optparse

# option handler
import optionHandler
options = optionHandler.optionHandler(sys.argv)

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
print(filedir)
sys.path.append(basedir)

# import class for DNN training
import DRACO_Frameworks.DNN.DNN as DNN
import DRACO_Frameworks.DNN.data_frame as df

options.initArguments()

# load samples
input_samples = df.InputSamples(options.getInputDirectory(), options.getActivatedSamples(), options.getTestPercentage())

Evt_weights = "x.Weight_GEN * x.gen_ft_xsNorm * x.lepSF_ft_muIsoSF * x.lepSF_ft_muTrigSF * x.lepSF_ft_muIDSF * x.bSF_ft_itFitSF_nom * x.gen_ft_pileup"

# during preprocessing half of the ttH sample is discarded (Even/Odd splitting),
# thus, the event yield has to be multiplied by two. This is done with normalization_weight = 2.
input_samples.addSample(options.getDefaultName("Sig_MH1200_hSBB"), label = "Sig_MH1200_hSBB", total_weight_expr=Evt_weights)
input_samples.addSample(options.getDefaultName("Sig_MH1200_hSTauTau"), label = "Sig_MH1200_hSTauTau", total_weight_expr=Evt_weights)
input_samples.addSample(options.getDefaultName("ttbar"), label = "ttbar", total_weight_expr=Evt_weights)
input_samples.addSample(options.getDefaultName("Zll"), label = "Zll", total_weight_expr=Evt_weights)
input_samples.addSample(options.getDefaultName("Wjet"), label = "Wjet", total_weight_expr=Evt_weights)



if options.isBinary():
   input_samples.addBinaryLabel(options.getSignal(), options.getBinaryBkgTarget())


# initializing DNN training class

dnn = DNN.DNN(

    save_path       =  options.getOutputDir(),
    input_samples   = input_samples,

    category_name   = options.getCategory(),

    train_variables = options.getTrainVariables(),

    # number of epochs
    train_epochs    = options.getTrainEpochs(),
    # metrics for evaluation (c.f. KERAS metrics)
    eval_metrics    = ['acc'],
    # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
    test_percentage = options.getTestPercentage(),
    # balance samples per epoch such that there amount of samples per category is roughly equal
    balanceSamples  = options.doBalanceSamples(),
    evenSel         = options.doEvenSelection(),
    norm_variables  = options.doNormVariables()
)



# build DNN model
dnn.build_model(options.getNetConfig())

# perform the training
dnn.train_model()

# evalute the trained model
dnn.eval_model()

# save information
dnn.save_model(sys.argv, filedir, options.getNetConfigName())

# save configurations of variables for plotscript
#dnn.variables_configuration()

# save and print variable ranking according to the input layer weights
dnn.get_input_weights()

# save and print variable ranking according to all layer weights
dnn.get_weights()


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
            nbins       = 20,
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
            signal_class        = options.getSignal(),
            privateWork         = options.isPrivateWork())

dnn.get_gradients(options.isBinary())
