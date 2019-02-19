# global imports
import os
import sys

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

# import class for DNN training
import DRACO_Frameworks.AutoEncoder.AutoEncoder as AE
import DRACO_Frameworks.AutoEncoder.data_frame as df
# specify which variable set to use
import variable_sets.allVariables as variable_set

# when executing the script give the jet-tag category as a first argument
# (ge)[nJets]j_(ge)[nTags]t
JTcategory      = sys.argv[1]

# the input variables are loaded from the variable_set file
variables       = variable_set.variables[JTcategory]

# absolute path to folder with input dataframes
inPath   = "/ceph/vanderlinden/MLFoyTrainData/DNN_newJEC/"

# naming for input files
naming = "_dnn_newJEC.h5"

# load samples
input_samples = df.InputSamples(inPath)
input_samples.addEncoderSample("ttHbb"+naming, label = "ttH", normalization_weight = 2)

input_samples.addEvalSample("ttbb"+naming,  label = "ttbb")
input_samples.addEvalSample("tt2b"+naming,  label = "tt2b")
input_samples.addEvalSample("ttb"+naming,   label = "ttb")
input_samples.addEvalSample("ttcc"+naming,  label = "ttcc")
input_samples.addEvalSample("ttlf"+naming,  label = "ttlf")



# path to output directory (adjust NAMING)
savepath = basedir+"/workdir/"+"autoEncoder_newJEC_v1_"+str(JTcategory)

ae = AE.AutoEncoder(
    save_path       = savepath,
    input_samples   = input_samples,
    event_category  = JTcategory,
    train_variables = variables,
    # number of epochs
    train_epochs    = 200,
    batch_size      = 200,
    # number of epochs without decrease in loss before stopping
    early_stopping  = 20,
    # loss function
    loss_function   = "mean_squared_error",
    # metrics for evaluation (c.f. KERAS metrics)
    eval_metrics    = ["mean_absolute_error"],
    # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
    test_percentage = 0.2)

# build default model
ae.build_model()
# perform the training
ae.train_model()
# evalute the trained model
ae.eval_model()

# plotting 
# plot the evaluation metrics
ae.plot_metrics()
ae.plot_reconstruction()
ae.plot_lossValues()

