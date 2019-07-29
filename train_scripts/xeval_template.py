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

# import with ROOT
from pyrootsOfTheCaribbean.evaluationScripts import plottingScripts

"""
USE: python train_template.py -o DIR -v FILE -p -l --privatework --signalclass=STR --printroc
"""
usage="usage=%prog [options] \n"
usage+="USE: python train_template.py -o DIR -v FILE -p -l --privatework --signalclass=STR --printroc "

parser = optparse.OptionParser(usage=usage)

parser.add_option("-i", "--inputdirectory", dest="inputDir",default="test_training_4j_ge3t",
        help="DIR of trained net data", metavar="inputDir")

parser.add_option("-o", "--outputdirectory", dest="outDir",default=None,
        help="DIR of evaluation outputs, if None specified use inputDir", metavar="outDir")

parser.add_option("-v", "--variableselection", dest="variableSelection",default="example_variables",
        help="FILE for variables used to train DNNs (allows relative path to variable_sets)", metavar="variableSelection")

parser.add_option("-p", "--plot", dest="plot", action = "store_true", default=False,
        help="activate to create plots", metavar="plot")

parser.add_option("-l", "--log", dest="log", action = "store_true", default=False,
        help="activate for logarithmic plots", metavar="log")

parser.add_option("--privatework", dest="privateWork", action = "store_true", default=False,
        help="activate to create private work plot label", metavar="privateWork")

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

#get input directory path of dnn_odd
if not os.path.isabs(options.inputDir+"_odd"):
    inPath = basedir+"/workdir/"+options.inputDir
elif os.path.exists(options.inputDir+"_odd"):
    inPath = options.inputDir
else:
    sys.exit("ERROR: Input Directory for odd DNN does not exist!")

#get input directory path of dnn_even
if not os.path.isabs(options.inputDir+"_even"):
    inPath = basedir+"/workdir/"+options.inputDir
elif os.path.exists(options.inputDir+"_even"):
    inPath = options.inputDir
else:
    sys.exit("ERROR: Input Directory for even DNN does not exist!")

if not options.outDir:
    outPath = inPath+"_xEval"
elif not os.path.isabs(options.outDir):
    outPath = basedir+"/workdir/"+options.outDir
else:
    outPath = options.outDir

if options.signal_class:
    signal=options.signal_class.split(",")
else:
    signal=None

#load DNNs
dnn_odd = DNN.loadDNN_crossEval(inPath+"_odd", outPath)
dnn_even = DNN.loadDNN_crossEval(inPath+"_even", outPath)

# plotting 
if options.plot:
    # # plot the confusion matrix
    # dnn_odd.plot_confusionMatrix(privateWork = options.privateWork, printROC = options.printROC)
    # dnn_even.plot_confusionMatrix(privateWork = options.privateWork, printROC = options.printROC)

    # # plot the output discriminators
    # dnn_odd.plot_discriminators(log = options.log, signal_class = options.signal_class, privateWork = options.privateWork, printROC = options.printROC)
    # dnn_even.plot_discriminators(log = options.log, signal_class = options.signal_class, privateWork = options.privateWork, printROC = options.printROC)


    # # plot the output nodes
    # dnn_odd.plot_outputNodes(log = options.log, signal_class = options.signal_class, privateWork = options.privateWork, printROC = options.printROC)
    # dnn_even.plot_outputNodes(log = options.log, signal_class = options.signal_class, privateWork = options.privateWork, printROC = options.printROC)


    # # plot closure test
    # dnn_odd.plot_closureTest(log = options.log, signal_class = options.signal_class, privateWork = options.privateWork)
    # dnn_even.plot_closureTest(log = options.log, signal_class = options.signal_class, privateWork = options.privateWork)


    # plot cross evaluation
    DNN.plot_crossEval(dnn_even = dnn_even, dnn_odd = dnn_odd, plotdir = outPath, log = options.log, signal_class = options.signal_class, privateWork = options.privateWork)


