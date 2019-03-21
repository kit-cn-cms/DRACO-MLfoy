# global imports
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import sys
import keras.optimizers as optimizers
import optparse

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

# import class for DNN training
import DRACO_Frameworks.DNN.DNN as DNN
import DRACO_Frameworks.DNN.data_frame as df

"""
USE: python train_template.py -o DIR -v FILE -n STR -c STR -e INT -s INT -p -l --privatework --netconfig=STR --signalclass=STR --printroc
"""
usage="usage=%prog [options] \n"
usage+="USE: python train_template.py -o DIR -v FILE -n STR -c STR -e INT -s INT -p -l --privatework --netconfig=STR --signalclass=STR --printroc "

parser = optparse.OptionParser(usage=usage)

parser.add_option("-o", "--outputdirectory", dest="outputDir",default="test_training",
        help="DIR for output", metavar="outputDir")

parser.add_option("-i", "--inputdirectory", dest="inputDir",default="InputFeatures",
        help="DIR for input", metavar="inputDir")

parser.add_option("-n", "--naming", dest="naming",default="_dnn.h5",
        help="file ending for the samples in preprocessing", metavar="naming")

parser.add_option("-c", "--category", dest="category",default="4j_ge3t",
        help="STR name of the category (ge)[nJets]j_(ge)[nTags]t", metavar="category")

parser.add_option("-e", "--trainepochs", dest="train_epochs",default=1000,
        help="INT number of training epochs", metavar="train_epochs")

parser.add_option("-v", "--variableselection", dest="variableSelection",default="example_variables",
        help="FILE for variables used to train DNNs", metavar="variableSelection")

parser.add_option("-p", "--plot", dest="plot", action = "store_true", default=False,
        help="activate to create plots", metavar="plot")

parser.add_option("-l", "--log", dest="log", action = "store_true", default=False,
        help="activate for logarithmic plots", metavar="log")

parser.add_option("--privatework", dest="privateWork", action = "store_true", default=False,
        help="activate to create private work plot label", metavar="privateWork")

parser.add_option("--netconfig", dest="net_config",default=None,
        help="STR of config name in net_config (config in this file will not be used anymore!)", metavar="net_config")

parser.add_option("--signalclass", dest="signal_class", default=None,
        help="STR of signal class for plots", metavar="signal_class")

parser.add_option("--printroc", dest="printROC", action = "store_true", default=False,
        help="activate to print ROC value for confusion matrix", metavar="printROC")


(options, args) = parser.parse_args()

#import Variable Selection
if not os.path.isabs(options.variableSelection):
    sys.path.append(basedir+"/variable_sets/")
    variable_set = __import__(options.variableSelection)
elif os.path.exists(options.variableSelection):
    variable_set = __import__(options.variableSelection)
else:
    sys.exit("ERROR: Variable Selection File does not exist!")

#get input directory path
if not os.path.isabs(options.inputDir):
    inPath = basedir+"/workdir/"+options.inputDir
elif os.path.exists(options.inputDir):
    inPath=options.inputDir
else:
    sys.exit("ERROR: Input Directory does not exist!")

#get output directory path
if not os.path.isabs(options.outputDir):
    outputdir = basedir+"/workdir/"+options.outputDir
elif os.path.exists(options.outputDir):
    outputdir=options.outputDir
else:
    sys.exit("ERROR: Output Directory does not exist!")

#add nJets and nTags to output directory
outputdir += "_"+options.category

# the input variables are loaded from the variable_set file
if options.category in variable_set.variables:
    variables = variable_set.variables[options.category]
else:
    variables = variable_st.all_variables
    print("category {} not specified in variable set {} - using all variables".format(
        options.category, options.variableSelection))


# load samples
input_samples = df.InputSamples(inPath)
naming = options.naming

# during preprocessing half of the ttH sample is discarded (Even/Odd splitting),
#       thus, the event yield has to be multiplied by two. This is done with normalization_weight = 2.
input_samples.addSample("ttHbb"+naming, label = "ttHbb", normalization_weight = 2.)
input_samples.addSample("ttbb"+naming,  label = "ttbb")
input_samples.addSample("tt2b"+naming,  label = "tt2b")
input_samples.addSample("ttb"+naming,   label = "ttb")
input_samples.addSample("ttcc"+naming,  label = "ttcc")
input_samples.addSample("ttlf"+naming,  label = "ttlf")


# initializing DNN training class
dnn = DNN.DNN(
    save_path       = outputdir,
    input_samples   = input_samples,
    event_category  = options.category,
    train_variables = variables,
    # number of epochs
    train_epochs    = int(options.train_epochs),
    # metrics for evaluation (c.f. KERAS metrics)
    eval_metrics    = ["acc"],
    # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
    test_percentage = 0.2)

# config dictionary for DNN architecture
config = {
    "layers":                   [200,200],
    "loss_function":            "categorical_crossentropy",
    "Dropout":                  0.5,
    "L2_Norm":                  1e-5,
    "batch_size":               5000,
    "optimizer":                optimizers.Adagrad(decay=0.99),
    "activation_function":      "elu",
    "output_activation":        "Softmax",
    "earlystopping_percentage": 0.05,
    "earlystopping_epochs":     50,
    }

# import file with net configs if option is used
if options.net_config:
    from net_configs import config_dict
    config=config_dict[options.net_config]

# build DNN model
dnn.build_model(config)

# perform the training
dnn.train_model()

# save information
dnn.save_model(sys.argv, filedir)

# evalute the trained model
dnn.eval_model()

# save and print variable ranking
dnn.get_input_weights()

# plotting 
if options.plot:
    # plot the evaluation metrics
    dnn.plot_metrics(privateWork = options.privateWork)

    # plot the confusion matrix
    dnn.plot_confusionMatrix(privateWork = options.privateWork, printROC = options.printROC)

    # plot the output discriminators
    dnn.plot_discriminators(log = options.log, signal_class = options.signal_class, privateWork = options.privateWork, printROC = options.printROC)

    # plot the output nodes
    dnn.plot_outputNodes(log = options.log, signal_class = options.signal_class, privateWork = options.privateWork, printROC = options.printROC)

    # plot closure test
    dnn.plot_closureTest(log = options.log, signal_class = options.signal_class, privateWork = options.privateWork)
