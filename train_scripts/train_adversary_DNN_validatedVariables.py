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

options.initArguments()

# load samples
input_samples = df.InputSamples(options.getInputDirectory(), options.getActivatedSamples(), options.getTestPercentage())
# during preprocessing half of the ttH sample is discarded (Even/Odd splitting),
#       thus, the event yield has to be multiplied by two. This is done with normalization_weight = 2.
input_samples.addSample(options.getDefaultName("ttHbb"),     label = "ttH", normalization_weight = 2.)
input_samples.addSample(options.getDefaultName("ttbb"),      label = "ttbb")
input_samples.addSample(options.getDefaultName("tt2b"),      label = "tt2b")
input_samples.addSample(options.getDefaultName("ttb"),       label = "ttb")
input_samples.addSample(options.getDefaultName("ttcc"),      label = "ttcc")
input_samples.addSample(options.getDefaultName("ttlf"),      label = "ttlf")
input_samples.addSample("ttbb_dnn_OL.h5",   label = "ttbb_OL")
input_samples.addSample("ttb_dnn_OL.h5",    label = "ttb_OL")
input_samples.addSample("tt2b_dnn_OL.h5",   label = "tt2b_OL")
#input_samples.addSample("ttZ"+naming,   label = "ttZ", isTrainSample = False, signalSample = True)

# initializing DNN training class
dnn = DNN.GAN(
    save_path       = options.getOutputDir(),
    input_samples   = input_samples,
    category_name   = options.getCategory(),
    train_variables = options.getTrainVariables(),
    # number of epochs
    train_epochs    = options.getTrainEpochs(), # hard coded in backend
    # metrics for evaluation (c.f. KERAS metrics)
    eval_metrics    = ["acc"],
    # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
    test_percentage = options.getTestPercentage(),
    # balance samples per epoch such that there amount of samples per category is roughly equal
    balanceSamples  = options.doBalanceSamples(),
    evenSel         = options.doEvenSelection(),
    norm_variables  = options.doNormVariables())


dnn.build_model(config=options.getNetConfig() ,penalty=10)

dnn.train_model()

dnn.save_model(sys.argv, filedir, options.getNetConfigName())


# dnn.eval_model()
# # get variable ranking
# dnn.get_input_weights()

# # plotting 
# # plot the evaluation metrics
# dnn.plot_metrics()
# # plot the confusion matrix
# dnn.plot_confusionMatrix(norm_matrix = True)
# # plot the output discriminators
# dnn.plot_discriminators()
# dnn.plot_outputNodes()

# example: python train_adversary_DNN_validatedVariables.py -i /storage/9/jschindler/DNN_adversary_2019/DNN_newJEC/ --naming _dnn_newJEC.h5 -o adversary_test -n adversary_test -c ge6j_ge3t -v newJEC_validated
