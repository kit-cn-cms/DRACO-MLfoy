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

# specify which variable set to use
import variable_sets.newJEC_validated as variable_set


cats = ["4j_ge3t","5j_ge3t","ge6j_ge3t"]


# when executing the script give the jet-tag category as a first argument
# (ge)[nJets]j_(ge)[nTags]t
JTcategory      = cats[2]

# the input variables are loaded from the variable_set file
variables       = variable_set.variables[JTcategory]

# absolute path to folder with input dataframes
inPath   = "/storage/9/jschindler/DNN_adversary_2019/DNN_newJEC/"

# naming for input files
naming = "_dnn_newJEC.h5"

# load samples
input_samples = df.InputSamples(inPath)
# during preprocessing half of the ttH sample is discarded (Even/Odd splitting),
#       thus, the event yield has to be multiplied by two. This is done with normalization_weight = 2.
input_samples.addSample("ttHbb"+naming,     label = "ttH", normalization_weight = 2.)
input_samples.addSample("ttbb"+naming,      label = "ttbb")
input_samples.addSample("tt2b"+naming,      label = "tt2b")
input_samples.addSample("ttb"+naming,       label = "ttb")
input_samples.addSample("ttcc"+naming,      label = "ttcc")
input_samples.addSample("ttlf"+naming,      label = "ttlf")
input_samples.addSample("ttbb_dnn_OL.h5",   label = "ttbb_OL")
input_samples.addSample("ttb_dnn_OL.h5",    label = "ttb_OL")
input_samples.addSample("tt2b_dnn_OL.h5",   label = "tt2b_OL")
#input_samples.addSample("ttZ"+naming,   label = "ttZ", isTrainSample = False, signalSample = True)



# path to output directory (adjust NAMING)
savepath = basedir+"/workdir/masterthesis/lambda_"+str(10)+"_OL_redone_2/test" +str(JTcategory)+"/"

# initializing DNN training class
dnn = DNN.GAN(
    save_path       = savepath,
    input_samples   = input_samples,
    category_name   = JTcategory,
    train_variables = variables,
    # number of epochs
    train_epochs    = 500,
    # metrics for evaluation (c.f. KERAS metrics)
    eval_metrics    = ["acc"],
    # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
    test_percentage = 0.2,
    balanceSamples  = False)


dnn.build_model(config=options.getNetConfig() ,penalty=10)
dnn.train_model()


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
