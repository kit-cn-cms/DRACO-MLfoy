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

parser.add_option("-i", "--inputdirectory", dest="inputDir",default="test_training_ge4j_ge4t",
        help="DIR of trained net data", metavar="inputDir")

parser.add_option("-o", "--outputdirectory", dest="outDir",default=None,
        help="DIR of evaluation outputs, if None specified use inputDir", metavar="outDir")

parser.add_option("-p", "--plot", dest="plot", action = "store_true", default=False,
        help="activate to create plots", metavar="plot")

parser.add_option("-l", "--log", dest="log", action = "store_true", default=False,
        help="activate for logarithmic plots", metavar="log")

parser.add_option("--privatework", dest="privateWork", action = "store_true", default=False,
        help="activate to create private work plot label", metavar="privateWork")

parser.add_option("--signalclass","-S", dest="signal_class", default=None,
        help="STR of signal class for plots", metavar="signal_class")

parser.add_option("--printroc", dest="printROC", action = "store_true", default=False,
        help="activate to print ROC value for confusion matrix", metavar="printROC")

parser.add_option("--binary", dest="binary", action = "store_true", default=False,
        help="activate to perform binary classification instead of multiclassification. Takes the classes passed to 'signal_class' as signals, all others as backgrounds.")

parser.add_option("-t", "--binaryBkgTarget", dest="binary_bkg_target", default = 0.,
        help="target value for training of background samples (default is 0, signal is always 1)")

parser.add_option("--total-weight-expr", dest="total_weight_expr",default="x.Weight_XS * x.Weight_CSV * x.Weight_GEN_nom",
        help="string containing expression of total event weight (use letter \"x\" for event-object; example: \"x.weight\")", metavar="total_weight_expr")

parser.add_option("-d", "--derivatives", dest="derivatives", action = "store_true", default=False,
        help="activate to get first and second order derivatives", metavar="dev")
parser.add_option("-s", "--shap_evaluation", dest="shap_evaluation", action = "store_true", default=False,
        help="activate to get input feature decomposition using SHAP algorithm. Default: False")

parser.add_option("-c", "--category", dest="category",default="4j_ge3t",
                help="STR name of the category (ge/le)[nJets]j_(ge/le)[nTags]t", metavar="CATEGORY")
(options, args) = parser.parse_args()

#get input directory path
if not os.path.isabs(options.inputDir):
    inPath = basedir+"/workdir/"+options.inputDir
elif os.path.exists(options.inputDir):
    inPath=options.inputDir
else:
    sys.exit("ERROR: Input Directory does not exist!")

if not options.outDir:
    outPath = inPath
elif not os.path.isabs(options.outDir):
    outPath = basedir+"/workdir/"+options.outDir
else:
    outPath = options.outDir

if options.signal_class:
    signal=options.signal_class.split(",")
else:
    signal=None
category_cutString_dict = {

    '3j_'+  '2t': '(N_jets == 3) & (N_btags == 2)',
    '3j_'+  '3t': '(N_jets == 3) & (N_btags == 3)',
  'ge4j_'+  '2t': '(N_jets >= 4) & (N_btags == 2)',
  'ge4j_'+  '3t': '(N_jets >= 4) & (N_btags == 3)',
  'ge4j_'+'ge4t': '(N_jets >= 4) & (N_btags >= 4)',

  'ge4j_'+'ge3t': '(N_jets >= 4) & (N_btags >= 3)',
}

category_label_dict = {

    '3j_'+  '2t': 'N_jets = 3, N_btags = 2',
    '3j_'+  '3t': 'N_jets = 3, N_btags = 3',
  'ge4j_'+  '2t': 'N_jets \\geq 4, N_btags = 2',
  'ge4j_'+  '3t': 'N_jets \\geq 4, N_btags = 3',
  'ge4j_'+'ge4t': 'N_jets \\geq 4, N_btags \\geq 4',

  'ge4j_'+'ge3t': 'N_jets \\geq 4, N_btags \\geq 3',
}

if options.binary:
    if not signal:
        sys.exit("ERROR: need to specify signal class if binary classification is activated")

dnn = DNN.loadDNN(inPath, outPath, binary = options.binary, signal = signal, binary_target = options.binary_bkg_target, total_weight_expr=options.total_weight_expr,
                    category_cutString = category_cutString_dict[options.category], category_label = category_label_dict[options.category])

# plotting
if options.plot:
    if options.binary:
        # plot output node
        bin_range = [options.binary_bkg_target, 1.]
        dnn.plot_binaryOutput(log = options.log, privateWork = options.privateWork, printROC = options.printROC, bin_range = bin_range)

    else:
        # plot the confusion matrix
        dnn.plot_confusionMatrix(privateWork = options.privateWork, printROC = options.printROC)

        # plot the output discriminators
        dnn.plot_discriminators(log = options.log, signal_class = options.signal_class, privateWork = options.privateWork, printROC = options.printROC)

        # plot the output nodes
        dnn.plot_outputNodes(log = options.log, signal_class = options.signal_class, privateWork = options.privateWork, printROC = options.printROC)

        # plot closure test
        dnn.plot_closureTest(log = options.log, signal_class = options.signal_class, privateWork = options.privateWork)

if options.derivatives:
    dnn.get_gradients(options.binary)

if options.shap_evaluation:
    dnn.get_shap_evaluation()