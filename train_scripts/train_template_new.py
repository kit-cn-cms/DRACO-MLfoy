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
import numpy as np

print("imports done")
"""
USE: python train_template.py -o DIR -v FILE -n STR -c STR -e INT -s INT -p -l --privatework --netconfig=STR --signalclass=STR --printroc
python train_template_new.py -i /ceph/swieland/ttH/h5Files/LegacyStrategy/Baseline -n _LegacyStrategyStudyBaseline.h5 --trainepochs 300 --netconfig binary_config --binary -t -1 --signalclass ttH -c ge6j_ge3t
#latest one
python train_template_new.py -o /home/ngolks/Templates/DM_output/BinaryNN -i /ceph/swieland/ttH/h5Files/LegacyStrategy/Baseline -n _LegacyStrategyStudyBaseline.h5 --trainepochs 300 --netconfig binary_config2 --binary -t 0 --signalclass ttH -c ge6j_ge3t --simultan 5
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
# parser.add_option("-b", "--boost", dest = "boost", default = None,
#         help = "INT number of networks trained parallel")

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


# Everything gets saved here
path = "/home/ngolks/Projects/boosted_dnn/BinaryNN/"
name_raw = "b"+str(int(options.train_epochs))+"_"+str(options.category)+"_"+options.net_config

#get number of epoches adaboost should perform and set default signal
# if options.adaboost:
#     ada_epochs = int(options.adaboost)
#     use_ada = True
#     options.binary = True
#     if options.signal_class:
#         pass
#     else:
#         signal = "ttH"
# else:
#     use_ada = False
#
# if options.m2:
#     m2 = options.m2
#     options.binary_bkg_target = 0     #otherwise the algorithm does not work
# else:
#     m2 = False

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

prediction_list = []

#loop dnn training to use boosting
for i in range(1, n_simoular+1):   #due to naming
    print("\n", "\n")
    print("Loop i: ", i, " of ", n_simoular)
    name = name_raw + '_s' + str(i)

    # initializing DNN training class
    dnn = DNN.DNN(
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
        # metrics for evaluation (c.f. KERAS metrics)
        eval_metrics    = ["acc"],
        # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
        test_percentage = 0.2,
        # balance samples per epoch such that there amount of samples per category is roughly equal
        balanceSamples  = options.balanceSamples,
        shuffle_seed = 9)   #shuffle_seed is for data not for weights

    # import file with net configs if option is used
    if options.net_config:
        from net_configs import config_dict
        config=config_dict[options.net_config]

    # check if this NN was already trained
    save_path = path + "save_model/" + name + "/"
    if os.path.exists(save_path):
        exists = True
    else:
        exists = False
    print("# DEBUG: exists: ", exists)

    if exists:
        # load trained model
        dnn.load_trained_model(save_path)    #loading the hole model takes like forever
        # ada.load_needed(path)
    else:
        # build DNN model
        dnn.build_model(config)

        # perform the training
        dnn.train_model()

    # evalute the trained model
    dnn.eval_model()

    # save information
    dnn.save_model(signal)

    # plotting
    # dnn.plot_binaryOutput(log = options.log, privateWork = options.privateWork, printROC = options.printROC)
    print("Starting plot metrics")
    dnn.plot_metrics()

    #for comparison of the DNNs
    if n_simoular > 1:
        #store prediction_vector
        prediction_list.append(dnn.model_prediction_vector)

#make comparison plots
prediction_vector = np.asarray(prediction_list)
data_len = prediction_vector.shape[1]

for h in np.arange(0, n_simoular-1):
    for j in np.arange(1+h, n_simoular):
        # title = "Compare prediction between two DNNs"# (" + str(h) + "," + str(j) +")"
        out1 = "/home/ngolks/Projects/boosted_dnn/BinaryNN/plot/Compare/diff1" + "_" + name_raw + "_" + str(h) + "_" + str(j) + ".pdf"
        out2 = "/home/ngolks/Projects/boosted_dnn/BinaryNN/plot/Compare/diff2" + "_" + name_raw + "_" + str(h) + "_" + str(j) + ".pdf"
        c1=ROOT.TCanvas("c1","Data", 200, 10, 700, 500)
        # c1.Divide(2,1)
        c1.SetLeftMargin(0.15)
        c1.cd(1)
        hist = ROOT.TH1D("hist", "", 15,-0.08,0.08)
        for i in np.arange(0, data_len):
            hist.Fill(prediction_vector[h][i] - prediction_vector[j][i])
        # hist.SetTitle(title)
        hist.GetXaxis().SetTitle("Differenz der Ausgabe")
        hist.GetXaxis().SetTitleSize(0.045)
        hist.GetXaxis().SetLabelSize(0.045)
        hist.GetYaxis().SetTitle("Anzahl")
        hist.GetYaxis().SetTitleSize(0.045)
        hist.GetYaxis().SetLabelSize(0.045)
        hist.Draw()

        diff = prediction_vector[h] - prediction_vector[j]
        abs_mean = np.mean(np.absolute(diff))
        print("absolute mean ", h, "_", j, " : ", abs_mean)

        c1.cd(2)
        l = c1.GetLeftMargin()
        t = c1.GetTopMargin()
        r = c1.GetRightMargin()
        b = c1.GetBottomMargin()

        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextColor(ROOT.kBlack)
        latex.SetTextSize(0.04)
        text = "CMS private work"
        latex.DrawLatex(l+0.57,1.-t+0.03, text)

        c1.Print(out1)

        # label_roc(h, j, roc_vector[h], roc_vector[j])      #write down the roc output
        c2=ROOT.TCanvas("c1","Data", 200, 10, 700, 500)
        c2.cd(1)
        hist2=ROOT.TH2D("hist", "", 40, 0, 1, 40, 0, 1)
        for i in np.arange(0, data_len):
            hist2.Fill(prediction_vector[h][i], prediction_vector[j][i])
        # hist.GetXaxis().SetTitle("Vorhersage B")
        # hist.GetYaxis().SetTitle("Vorhersage A")
        hist2.GetXaxis().SetLabelSize(0.045)
        hist2.GetYaxis().SetLabelSize(0.045)
        correlation = hist2.GetCorrelationFactor()
        print("correlation ", h, "_", j, " : ", correlation)
        hist2.SetStats(0)
        hist2.Draw("colz")

        c2.cd(2)
        l = c2.GetLeftMargin()
        t = c2.GetTopMargin()
        r = c2.GetRightMargin()
        b = c2.GetBottomMargin()

        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextColor(ROOT.kBlack)
        latex.SetTextSize(0.04)
        text = "CMS private work"
        latex.DrawLatex(l+0.57,1.-t+0.03, text)

        # label_correlation(hist2.GetCorrelationFactor())
        c2.Print(out2)


# save and print variable ranking
# dnn.get_input_weights()

# plotting
# if options.plot:
#     # plot the evaluation metrics
#     dnn.plot_metrics(privateWork = options.privateWork)
#
#     if options.binary:
#         # plot output node
#         bin_range = [input_samples.bkg_target, 1.]
#         dnn.plot_binaryOutput(log = options.log, privateWork = options.privateWork, printROC = options.printROC, bin_range = bin_range, name = run_name)
#     else:
#         # plot the confusion matrix
#         dnn.plot_confusionMatrix(privateWork = options.privateWork, printROC = options.printROC)
#
#         # plot the output discriminators
#         dnn.plot_discriminators(log = options.log, signal_class = signal, privateWork = options.privateWork, printROC = options.printROC)
#
#         # plot the output nodes
#         dnn.plot_outputNodes(log = options.log, signal_class = signal, privateWork = options.privateWork, printROC = options.printROC)
#
#         # plot event yields
#         dnn.plot_eventYields(log = options.log, signal_class = signal, privateWork = options.privateWork)
#
#         # plot closure test
#         # dnn.plot_closureTest(log = options.log, signal_class = signal, privateWork = options.privateWork)     #doesn't work so far
