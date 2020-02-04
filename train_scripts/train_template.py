# global imports
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import sys
import optparse

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

# import class for DNN training
import DRACO_Frameworks.DNN.DNN as DNN
import DRACO_Frameworks.DNN.data_frame as df

import keras.optimizers as optimizers
"""
USE: python train_template.py -o DIR -v FILE -n STR -c STR -e INT -s INT -p -l --privatework --netconfig=STR --signalclass=STR --printroc
"""
usage="usage=%prog [options] \n"
usage+="USE: python train_template.py -o DIR -v FILE -n STR -c STR -e INT -s INT -p -l --privatework --netconfig=STR --signalclass=STR --printroc "

parser = optparse.OptionParser(usage=usage)

parser.add_option("-o", "--outputdirectory", dest="outputDir",default="test_training",
        help="DIR for output (allows relative path to workdir or absolute path)", metavar="outputDir")

parser.add_option("-i", "--inputdirectory", dest="inputDir",default="InputFeatures",
        help="DIR of input h5 files (definition of files to load has to be adjusted in the script itself)", metavar="inputDir")

parser.add_option("-n", "--naming", dest="naming",default="_dnn.h5",
        help="file ending for the samples in input directory (default _dnn.h5)", metavar="naming")

parser.add_option("-c", "--category", dest="category",default="4j_ge3t",
        help="STR name of the category (ge/le)[nJets]j_(ge/le)[nTags]t", metavar="category")

parser.add_option("-e", "--trainepochs", dest="train_epochs",default=1000,
        help="INT number of training epochs (default 1000)", metavar="train_epochs")

parser.add_option("-v", "--variableselection", dest="variableSelection",default="trainZ",
        help="FILE for variables used to train DNNs (allows relative path to variable_sets)", metavar="variableSelection")

parser.add_option("-p", "--plot", dest="plot", action = "store_true", default=False,
        help="activate to create plots", metavar="plot")

parser.add_option("-l", "--log", dest="log", action = "store_true", default=False,
        help="activate for logarithmic plots", metavar="log")

parser.add_option("--privatework", dest="privateWork", action = "store_true", default=False,
        help="activate to create private work plot label", metavar="privateWork")

parser.add_option("--netconfig", dest="net_config",default="ttH_2017",
        help="STR of name of config (in net_configs.py) for building the network architecture ", metavar="net_config")

parser.add_option("--signalclass", dest="signal_class", default=None,
        help="STR of signal class for plots (allows comma separated list)", metavar="signal_class")

parser.add_option("--printroc", dest="printROC", action = "store_true", default=False,
        help="activate to print ROC value for confusion matrix", metavar="printROC")

parser.add_option("--balanceSamples", dest="balanceSamples", action = "store_true", default=False,
        help="activate to balance train samples such that number of events per epoch is roughly equal for all classes. The usual balancing of train weights for all samples is actiaved by default and is not covered with this option.", metavar="balanceSamples")

parser.add_option("--binary", dest="binary", action = "store_true", default=False,
        help="activate to perform binary classification instead of multiclassification. Takes the classes passed to 'signal_class' as signals, all others as backgrounds.")

parser.add_option("-t", "--binaryBkgTarget", dest="binary_bkg_target", default = 0.,
        help="target value for training of background samples (default is 0, signal is always 1)")

parser.add_option("-a", "--activateSamples", dest = "activateSamples", default = None,
        help="give comma separated list of samples to be used. ignore option if all should be used")

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
    outputdir = options.outputDir
elif os.path.exists(os.path.dirname(options.outputDir)):
    outputdir = options.outputDir
else:
    sys.exit("ERROR: Output Directory does not exist!")

#add nJets and nTags to output directory
run_name = outputdir.split("/")[-1]
outputdir += "_"+options.category

# the input variables are loaded from the variable_set file
if options.category in variable_set.variables:
    variables = variable_set.variables[options.category]
else:
    variables = variable_set.all_variables
    print("category {} not specified in variable set {} - using all variables".format(
        options.category, options.variableSelection))

if options.signal_class:
    signal=options.signal_class.split(",")
else:
    signal=None

if options.binary:
    if not signal:
        sys.exit("ERROR: need to specify signal class if binary classification is activated")

# load samples
input_samples = df.InputSamples(inPath, options.activateSamples)
naming = options.naming


# during preprocessing half of the ttH sample is discarded (Even/Odd splitting),
#       thus, the event yield has to be multiplied by two. This is done with normalization_weight = 2.

input_samples.addSample("ttH"+naming,   label = "ttH", normalization_weight = 2.)
input_samples.addSample("bkg"+naming,  label = "bkg")
input_samples.addSample("ttbb"+naming,  label = "ttbb")
input_samples.addSample("tt2b"+naming,  label = "tt2b")
input_samples.addSample("ttb"+naming,   label = "ttb")
input_samples.addSample("ttcc"+naming,  label = "ttcc")
input_samples.addSample("ttlf"+naming,  label = "ttlf")

if options.binary:
    input_samples.addBinaryLabel(signal, options.binary_bkg_target)

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
    test_percentage = 0.2,
    # balance samples per epoch such that there amount of samples per category is roughly equal
    balanceSamples  = options.balanceSamples)

# import file with net configs if option is used
if options.net_config:
    from net_configs import config_dict
    config=config_dict[options.net_config]

# build DNN model
dnn.build_model(config)

# perform the training
dnn.train_model()

# evalute the trained model
dnn.eval_model()

# save information
dnn.save_model(sys.argv, filedir)

# save and print variable ranking
dnn.get_input_weights()

# plotting 
if options.plot:
    # plot the evaluation metrics
    dnn.plot_metrics(privateWork = options.privateWork)
 
    if options.binary:
        # plot output node
        bin_range = [input_samples.bkg_target, 1.]
        dnn.plot_binaryOutput(log = options.log, privateWork = options.privateWork, printROC = options.printROC, bin_range = bin_range, name = run_name)
    else:
        # plot the confusion matrix
        dnn.plot_confusionMatrix(privateWork = options.privateWork, printROC = options.printROC)

        # plot the output discriminators
        dnn.plot_discriminators(log = options.log, signal_class = signal, privateWork = options.privateWork, printROC = options.printROC)

        # plot the output nodes
        dnn.plot_outputNodes(log = options.log, signal_class = signal, privateWork = options.privateWork, printROC = options.printROC)
        
        # plot event yields
        dnn.plot_eventYields(log = options.log, signal_class = signal, privateWork = options.privateWork)

        # plot closure test
        dnn.plot_closureTest(log = options.log, signal_class = signal, privateWork = options.privateWork)

