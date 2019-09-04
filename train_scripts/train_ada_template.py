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
import DRACO_Frameworks.DNN.AdaBoost as ADA
import DRACO_Frameworks.DNN.data_frame as df

import keras.optimizers as optimizers
import numpy as np
"""
USE: python train_ada_template.py -o DIR -v FILE -n STR -c STR -e INT -s INT -p -l --privatework --netconfig=STR --signalclass=STR --printroc
python train_ada_template.py -o /home/ngolks/Templates/DM_output/ada_first_test/ -i /ceph/swieland/ttH/h5Files/LegacyStrategy/Baseline/ -n _LegacyStrategyStudyBaseline.h5 --trainepochs 100 --netconfig ada_weak1 --adaboost 2 --binary -t -1 --signalclass ttH -c ge6j_ge3t
"""

# python train_ada_template.py -o /home/ngolks/Templates/DM_output/ada_first_test/ -i /home/swieland/ttH/LegacyStrategy/Baseline/ -n _LegacyStrategyStudyBaseline.h5 --trainepochs 100 --netconfig binary_config --adaboost 2 --binary -t -1 --signalclass ttH -c ge6j_ge3t

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

parser.add_option("-v", "--variableselection", dest="variableSelection",default="example_variables",
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

parser.add_option("-b", "--simultan", dest = "simultan", default = None,
        help = "INT number of networks trained simultaneously")

parser.add_option("--adaboost", dest = "adaboost", default = None,
        help = "INT number of epoches Adaboost should perform")

parser.add_option("--m2", dest = "m2_use", default = False,
        help = "Should AdaBoost.M2 algorithm be used")

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
    # signal=None       #signal=None doesn't work due to tHbb gets triggered later on if signal=None
    signal=["ttH", "ttbb", "tt2b", "ttb", "ttcc", "ttlf"]
if options.binary:
    if not signal:
        sys.exit("ERROR: need to specify signal class if binary classification is activated")

#get number of dnns to train
if options.simultan:
    n_simoular = int(options.simultan)
else:
    n_simoular = 1

#get number of epoches adaboost should perform and set default signal
if options.adaboost:
    ada_epochs = int(options.adaboost)
    use_ada = True
    options.binary = True
    if options.signal_class:
        pass
    else:
        signal = "ttH"
else:
    use_ada = False

#Should AdaBoost.m2_use be used
m2_use = options.m2_use
if m2_use:
    options.binary_bkg_target = -1       #algorithm does not work with different range

# load samples
input_samples = df.InputSamples(inPath, options.activateSamples)
naming = options.naming


# during preprocessing half of the ttH sample is discarded (Even/Odd splitting),
#       thus, the event yield has to be multiplied by two. This is done with normalization_weight = 2.


input_samples.addSample("ttH"+naming,   label = "ttH", normalization_weight = 2.)
input_samples.addSample("ttbb"+naming,  label = "ttbb")
input_samples.addSample("tt2b"+naming,  label = "tt2b")
input_samples.addSample("ttb"+naming,   label = "ttb")
input_samples.addSample("ttcc"+naming,  label = "ttcc")
input_samples.addSample("ttlf"+naming,  label = "ttlf")

if options.binary:
    input_samples.addBinaryLabel(signal, options.binary_bkg_target)

# get output path and name
if m2_use:
    path = "/home/ngolks/Projects/boosted_dnn/AdaBoost_M2/"
else:
    path = "/home/ngolks/Projects/boosted_dnn/AdaBoost/"           #needs to be adjusted
name_raw = "b"+str(int(options.train_epochs))+"a"+str(int(ada_epochs))+"_"+str(options.category)+"_"+options.net_config

prediction_list = []

#loop over training to use simultaneously training of nets with same data (data is initialized before)
for i in range(1, n_simoular+1):   #due to naming
    print("\n", "\n")
    print("Loop i: ", i, " of ", n_simoular)
    name = name_raw + '_s' + str(i)

    #initializing AdaBoost training class
    ada = ADA.AdaBoost(
        save_path       = outputdir,
        path            = path,
        name            = name,
        input_samples   = input_samples,        #samples are splitted before training the networks
        event_category  = options.category,
        train_variables = variables,
        #binary_bkg_target
        binary_bkg_target = options.binary_bkg_target,
        # number of epochs
        train_epochs    = int(options.train_epochs),
        eval_metrics    = ["acc"],
        # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
        test_percentage = 0.2,
        # balance samples per epoch such that there amount of samples per category is roughly equal
        balanceSamples  = options.balanceSamples,
        adaboost_epochs = ada_epochs,
        shuffle_seed = 9,   #shuffle_seed is for data
        m2 = m2_use)

    # import file with net configs if option is used
    if options.net_config:
        from net_configs import config_dict
        config=config_dict[options.net_config]

    # check if this DNN was already trained
    save_path = path + "save_model/" + name + "/"
    if os.path.exists(save_path):
        exists = True
    else:
        exists = False
    print("# DEBUG: exists: ", exists)

    if exists:
        # load trained model
        ada.load_trained_model(path)
    else:
        # build DNN model
        ada.build_model(config)

        # perform the training
        ada.train_model()

    print("# DEBUG: eval_model")
    # evalute the trained model
    ada.eval_model()

    print("# DEBUG: plot_binaryOutput")
    # make discriminator plot
    ada.plot_binaryOutput(log = options.log, privateWork = options.privateWork, printROC = options.printROC)

    if not exists:
        # save the trained model
        ada.save_model(signal)

    print("# DEBUG: appending to prediction_list")
    #for comparison of the DNNs
    if n_simoular > 1:
        #store prediction_vector
        prediction_list.append(ada.model_prediction_vector)

print("# DEBUG: starting comparison")

#make comparison plots
prediction_vector = np.asarray(prediction_list)
data_len = prediction_vector.shape[1]

for h in np.arange(0, n_simoular-1):
    for j in np.arange(1+h, n_simoular):
        title = "Compare prediction between two DNNs"# (" + str(h) + "," + str(j) +")"
        out1 = path + "plot/Compare/diff1" + "_" + name_raw + "_" + str(h) + "_" + str(j) + ".pdf"
        out2 = path + "plot/Compare/diff2" + "_" + name_raw + "_" + str(h) + "_" + str(j) + ".pdf"
        c1=ROOT.TCanvas("c1","Data", 200, 10, 700, 500)
        # c1.Divide(2,1)
        c1.cd(1)
        hist = ROOT.TH1D("hist", "", 15,-0.4,0.4)
        for i in np.arange(0, data_len):
            hist.Fill(prediction_vector[h][i] - prediction_vector[j][i])
        hist.SetTitle(title)
        hist.Draw()
        c1.Print(out1)
        # label_roc(h, j, roc_vector[h], roc_vector[j])      #write down the roc output
        c2=ROOT.TCanvas("c1","Data", 200, 10, 700, 500)
        c2.cd(1)
        hist2=ROOT.TH2D("hist", "", 40, 0, 1, 40, 0, 1)
        for i in np.arange(0, data_len):
            hist2.Fill(prediction_vector[h][i], prediction_vector[j][i])
        hist2.Draw("colz")
        # label_correlation(hist2.GetCorrelationFactor())
        c2.Print(out2)

print("Done: ", name)
