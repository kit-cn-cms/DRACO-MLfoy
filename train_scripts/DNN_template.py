# global imports
import os
import sys

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

# import class for DNN training
import DRACO_Frameworks.DNN.DNN as DNN
import DRACO_Frameworks.DNN.data_frame as df
# specify which variable set to use
import variable_sets.dnnVariableSet as variable_set

# when executing the script give the jet-tag category as a first argument
# (ge)[nJets]j_(ge)[nTags]t
JTcategory      = sys.argv[1]

# the input variables are loaded from the variable_set file
variables       = variable_set.variables[JTcategory]

# absolute path to folder with input dataframes
inPath   = "/ceph/vanderlinden/MLFoyTrainData/DNN_oldJEC/"

# naming for input files
naming = "_dnn_oldJEC.h5"

# load samples
input_samples = df.InputSamples(inPath)

# define the input samples one by one
input_samples.addSample("ttHbb"+naming, label = "ttHbb", signalSample = True)
input_samples.addSample("ttbb"+naming,  label = "ttbb")
input_samples.addSample("tt2b"+naming,  label = "tt2b")
input_samples.addSample("ttb"+naming,   label = "ttb")
input_samples.addSample("ttcc"+naming,  label = "ttcc")
input_samples.addSample("ttlf"+naming,  label = "ttlf")
# add samples that are not used in training but can be plotted in the discriminators
# set isTrainSample to false -> sample ignored during training
# set signalSample to true   -> sample plotted as additional signal (False for adding it to the background stack)
input_samples.addSample("ttZ"+naming,   label = "ttZ", isTrainSample = False, signalSample = True)


# path to output directory (adjust NAMING)
savepath = basedir+"/workdir/"+"oldJEC_allVariables_"+str(JTcategory)

# initializing DNN training class
dnn = DNN.DNN(
    save_path       = savepath,
    input_samples   = input_samples,
    event_category  = JTcategory,
    train_variables = variables,
    # number of epochs
    train_epochs    = 500,
    # number of epochs without decrease in loss before stopping
    early_stopping  = 20,
    # metrics for evaluation (c.f. KERAS metrics)
    eval_metrics    = ["acc"],
    # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
    test_percentage = 0.2)

#dnn.data.get_non_train_samples()

# build default model
dnn.build_model()
# perform the training
dnn.train_model()
# evalute the trained model
dnn.eval_model()
# get variable ranking
dnn.get_input_weights()

# plotting 
# plot the evaluation metrics
dnn.plot_metrics()
# plot the confusion matrix
dnn.plot_confusionMatrix(norm_matrix = True)
# plot the output discriminators
dnn.plot_discriminators(plot_nonTrainData = True)
