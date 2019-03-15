# global imports
import os
import sys
import keras.optimizers as optimizers

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

# import class for DNN training
import DRACO_Frameworks.DNN.DNN as DNN
import DRACO_Frameworks.DNN.data_frame as df

# specify which variable set to use
import variable_sets.newJEC_top20Variables as variable_set
# import file with net configs
import net_configs

# jet-tag category for trainig
# (ge)[nJets]j_(ge)[nTags]t
JTcategory      = sys.argv[1]

# the input variables are loaded from the variable_set file
variables       = variable_set.variables[JTcategory]

# absolute path to folder with input dataframes
inPath   = basedir+"/workdir/InputFeatures/"

# naming of input files
naming = "_dnn.h5"

# load samples
input_samples = df.InputSamples(inPath)

# during preprocessing half of the ttH sample is discarded (Even/Odd splitting),
#       thus, the event yield has to be multiplied by two. This is done with normalization_weight = 2.
input_samples.addSample("ttHbb"+naming, label = "ttHbb", signalSample = True, normalization_weight = 2.)
input_samples.addSample("ttbb"+naming,  label = "ttbb")
input_samples.addSample("tt2b"+naming,  label = "tt2b")
input_samples.addSample("ttb"+naming,   label = "ttb")
input_samples.addSample("ttcc"+naming,  label = "ttcc")
input_samples.addSample("ttlf"+naming,  label = "ttlf")

# path to output directory (adjust NAMING)
name = "test_training"
savepath = basedir+"/workdir/"+name+"_"+str(JTcategory)

# initializing DNN training class
dnn = DNN.DNN(
    save_path       = savepath,
    input_samples   = input_samples,
    event_category  = JTcategory,
    train_variables = variables,
    # number of epochs
    train_epochs    = 1000,
    # number of epochs without decrease in loss before stopping
    early_stopping  = 20,
    # metrics for evaluation (c.f. KERAS metrics)
    eval_metrics    = ["acc"],
    # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
    test_percentage = 0.2)

# build default model
# if given a dictionary as a first argument it uses the specified configs,
# otherwise builds default network defined in DNN class

dpg_config = {
    "layers":                   [200,200],#,300,300,300],
    "loss_function":            "categorical_crossentropy",
    "Dropout":                  0.5,
    "L2_Norm":                  1e-5,
    "batch_size":               5000,
    "optimizer":                optimizers.Adagrad(decay=0.99),
    "activation_function":      "elu",
    "output_activation":        "Softmax",
    "earlystopping_percentage": 0.05,
    "batchNorm":                False
    }


dnn.build_model(dpg_config)

# perform the training
dnn.train_model()

# evalute the trained model
dnn.eval_model()

# get variable ranking
dnn.get_input_weights()

# plotting 
# plot the evaluation metrics
dnn.plot_metrics(privateWork = True)

# plot the confusion matrix
dnn.plot_confusionMatrix(norm_matrix = True, privateWork = True)

# plot the output discriminators
dnn.plot_discriminators(signal_class = "ttHbb")


