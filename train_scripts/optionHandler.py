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


usage = ""
parser = optparse.OptionParser(usage=usage)

parser.add_option("-o", "--outputdirectory", dest="outputDir",default="test_training", 
        help="DIR for output (allows relative path to workdir or absolute path)", metavar="outputDir")

sampleopts = optparse.OptionGroup(parser, "Sample Settings")
sampleopts.add_option("-i", "--inputdirectory", dest="inputDir",default="InputFeatures",
        help="DIR of input h5 files (definition of files to load has to be adjusted in the script itself)", metavar="INPUTDIR")
sampleopts.add_option("--naming", dest="naming",default="_dnn.h5",
        help="file ending for the samples in input directory (default _dnn.h5)", metavar="SAMPLENAMING")
sampleopts.add_option("-c", "--category", dest="category",default="4j_ge3t",
        help="STR name of the category (ge/le)[nJets]j_(ge/le)[nTags]t", metavar="CATEGORY")
sampleopts.add_option("-a", "--activateSamples", dest = "activateSamples", default = None,
        help="give comma separated list of samples to be used. ignore option if all should be used")
sampleopts.add_option("--even",dest="even_sel",action="store_true",default=None,
        help="only select events with Evt_Odd==0")
sampleopts.add_option("--odd",dest="even_sel",action="store_false",default=None,
        help="only select events with Evt_Odd==1")
parser.add_option_group(sampleopts)

trainopts = optparse.OptionGroup(parser, "Train Configurations")
trainopts.add_option("-v", "--variableselection", dest="variableSelection",default="example_variables",
        help="FILE for variables used to train DNNs (allows relative path to variable_sets)", metavar="VARIABLESET")
trainopts.add_option("-n", "--netconfig", dest="net_config",default="ttH_2017",
        help="STR of name of config (in net_configs.py) for building the network architecture ", metavar="NETCONFIG")
trainopts.add_option("--layers", dest="layer_config", default=None,
        help="STR neurons_layer1, neurons_layer2, ... for dynamically changing the layer setting in net_configs.py") #me
trainopts.add_option("-e", "--epochs", dest="train_epochs",default=1000,
        help="INT number of training epochs (default 1000)", metavar="TRAINEPOCHS")
trainopts.add_option("-f", "--testfraction", dest="test_percentage",default=0.2,type=float,
        help="set fraction of events used for testing, rest is used for training", metavar="TESTFRACTION")
trainopts.add_option("--balanceSamples", dest="balanceSamples", action = "store_true", default=False,
        help="activate to balance train samples such that number of events per epoch is roughly equal for all classes. The usual balancing of train weights for all samples is actiaved by default and is not covered with this option.")
trainopts.add_option("-u", "--unnormed", dest = "norm_variables", action = "store_false", default = True,
        help = "activate to NOT perform a normalization of input features to mean zero and std deviation one.If using -q then the input features are automatically normed!") #me
trainopts.add_option("-q", "--quantile", dest = "qt_transformed_variables", action = "store_true", default = False,
        help = "activate to perform a quantile transformation on the input features") #me
trainopts.add_option("--restorefitdir", dest = "restore_fit_dir", default = None,
        help = "activate to restore the fit information from a quantile transformation. Only takes an ABSOLUTE path!") #me
trainopts.add_option("--debugs", dest = "debugs", default = None,
        help = "activate to restore the fit information from a quantile transformation. Only takes an ABSOLUTE path!") #me DEBUG

parser.add_option_group(trainopts)

plotopts = optparse.OptionGroup(parser, "Plotting Options")
plotopts.add_option("-p", "--plot", dest="plot", action = "store_true", default=False,
        help="activate to create plots")
plotopts.add_option("-s", "--sigScale", dest="sigScale", default = -1, type = float, metavar = "SIGSCALE",
        help="scale of signal histograms in output plots. -1 scales to background integral")
plotopts.add_option("-L", "--log", dest="log", action = "store_true", default=False,
        help="activate for logarithmic plots")
plotopts.add_option("-P", "--privatework", dest="privateWork", action = "store_true", default=False,
        help="activate to create private work plot label")
plotopts.add_option("-R", "--printroc", dest="printROC", action = "store_true", default=False, 
        help="activate to print ROC value for confusion matrix")
plotopts.add_option("-S","--signalclass", dest="signal_class", default=None, metavar="SIGNALCLASS",
        help="STR of signal class for plots (allows comma separated list) (same as --binarySignal)")
plotopts.add_option("-G","--gradients", dest="gradients", default=False, action="store_true",
        help="activate gradient calculation with tensorflow")
plotopts.add_option("-V","--variations", dest="variations", default=False, action="store_true",
        help="activate variation plots")
parser.add_option_group(plotopts)

binaryOptions = optparse.OptionGroup(parser, "Binary Options",
    "settings for binary DNN training")
binaryOptions.add_option("--binary", dest="binary", action = "store_true", default=False, 
        help="activate to perform binary classification instead of multiclassification. Takes the classes passed to 'signal_class' as signals, all others as backgrounds.")
binaryOptions.add_option("-t", "--binaryBkgTarget", dest="binary_bkg_target", default = 0., metavar="BKGTARGET",
        help="target value for training of background samples (default is 0, signal is always 1)")
binaryOptions.add_option("--signal", dest="signal_class", default=None, metavar="SIGNALCLASS", 
        help="STR of signal class for binary classification (allows comma separated list) (same as --signalclass)")
parser.add_option_group(binaryOptions)

adversaryopts = optparse.OptionGroup(parser, "Adversary Settings")
adversaryopts.add_option("--adversary", dest="adversary", action = "store_true", default=False,
        help="activate to train a classifying adversarial network")
adversaryopts.add_option("--penalty", dest="penalty", default=1,
        help="FLOAT number of penalty in loss function for adversary training (default 1)", metavar="PENALTY")
adversaryopts.add_option("--addsamplenaming", dest="AddSampleNaming", default="_dnn_OL.h5",
        help="file ending for the samples in input directory (default _dnn.h5)", metavar="SAMPLENAMING")


class optionHandler:
    def __init__(self, argv):
        (options, args) = parser.parse_args(argv[1:])
        self.__options  = options
        self.__args     = args


    def initArguments(self):
        self.__importVariableSelection()
        self.__setPaths()
        self.__setNomWeight()
        self.__loadVariables()
        self.__setSignalClass()
        self.__setNetConfig()
        self.__setAdversary()

    # setters

    def __importVariableSelection(self):
        if not os.path.isabs(self.__options.variableSelection):
            sys.path.append(basedir+"/variable_sets/")
            self.__variable_set = __import__(self.__options.variableSelection)
        elif os.path.exists(options.variableSelection):
            self.__variable_set = __import__(self.__options.variableSelection)
        else:
            sys.exit("ERROR: Variable Selection File does not exist!")

    def __setPaths(self):
        #get input directory path
        if not os.path.isabs(self.__options.inputDir):
            self.__inPath = basedir+"/workdir/"+self.__options.inputDir
        elif os.path.exists(self.__options.inputDir):
            self.__inPath = self.__options.inputDir
        else:
            sys.exit("ERROR: Input Directory does not exist!")

        #get output directory path
        if not os.path.isabs(self.__options.outputDir):
            self.__outputdir = basedir+"/workdir/"+self.__options.outputDir
        elif os.path.exists(self.__options.outputDir):
            self.__outputdir = self.__options.outputDir
        elif os.path.exists(os.path.dirname(self.__options.outputDir)):
            self.__outputdir = self.__options.outputDir
        else:
            sys.exit("ERROR: Output Directory does not exist!")

        #add nJets and nTags to output directory
        self.__run_name = self.__outputdir.split("/")[-1]
        self.__outputdir += "_"+self.__options.category

    def __setNomWeight(self):
        # handle even odd selection
        self.__nom_weight = 1.
        if self.__options.even_sel is None:
            return

        if self.__options.even_sel:
            self.__outputdir+="_even"
            self.__nom_weight = 2.
        elif not self.__options.even_sel:
            self.__outputdir+="_odd"
            self.__nom_weight = 2.

    def __loadVariables(self):
        # the input variables are loaded from the variable_set file
        if self.__options.category in self.__variable_set.variables:
            self.__variables = self.__variable_set.variables[self.__options.category]
        else:
            self.__variables = self.__variable_set.all_variables
            print("category {} not specified in variable set {} - using all variables".format(
                self.__options.category, self.__options.variableSelection))

    def __setSignalClass(self):
        if self.__options.signal_class:
            self.__signal = self.__options.signal_class.split(",")
        else:
            self.__signal = None

        if self.__options.binary:
            if not self.__signal:
                sys.exit("ERROR: need to specify signal class if binary classification is activated")


    def __setNetConfig(self):
        from net_configs import config_dict
        if self.__options.net_config:
            self.__config = config_dict[self.__options.net_config]
        
        else:
            self.__config = config_dict["example_config"]
            print("no net config was specified - using 'example_config'")
        
        if  self.__options.net_config and self.__options.layer_config:
            self.__config["layers"] = map(int, self.__options.layer_config.split(",")) 
            print self.__config

    def __setAdversary(self):
        if self.__options.adversary:
            if not self.__options.binary:
                print("WARNING: Weighting for multiclass adversary training is incorrect!")
            if self.__options.AddSampleNaming == "_dnn_OL.h5":
                print("no additional sample name was specified - using '_dnn_OL.h5'")
            if self.__options.naming == self.__options.AddSampleNaming:
                sys.exit("ERROR: need to use samples from different generators")



    # getters

    def getInputDirectory(self):
        return self.__inPath

    def getActivatedSamples(self):
        return self.__options.activateSamples

    def getTestPercentage(self):
        return self.__options.test_percentage

    def getDefaultName(self, sample):
        return sample+self.__options.naming

    def getNomWeight(self):
        return self.__nom_weight

    def isBinary(self):
        return self.__options.binary

    def getSignal(self):
        return self.__signal

    def getBinaryBkgTarget(self):
        return self.__options.binary_bkg_target

    def getOutputDir(self):
        return self.__outputdir

    def getCategory(self):
        return self.__options.category

    def getTrainVariables(self):
        return self.__variables

    def getTrainEpochs(self):
        return int(self.__options.train_epochs)

    def doBalanceSamples(self):
        return self.__options.balanceSamples

    def doEvenSelection(self):
        return self.__options.even_sel

    def doNormVariables(self):
        return self.__options.norm_variables
    
    def doQTNormVariables(self):
        return self.__options.qt_transformed_variables
    
    def getRestoreFitDir(self):
        return self.__options.restore_fit_dir
    
    def getDebug(self):
        return self.__options.debugs

    def getNetConfig(self):
        return self.__config

    def getNetConfigName(self):
        return self.__options.net_config

    def doPlots(self):
        return self.__options.plot

    def isPrivateWork(self):
        return self.__options.privateWork

    def getBinaryBinRange(self):
        return [float(self.getBinaryBkgTarget()), 1.]

    def doLogPlots(self):
        return self.__options.log

    def doPrintROC(self):
        return self.__options.printROC

    def getName(self):
        return self.__run_name

    def getSignalScale(self):
        return self.__options.sigScale

    def doVariations(self):
        return self.__options.variations

    def doGradients(self):
        return self.__options.gradients

    def isAdversary(self):
        return self.__options.adversary

    def getPenalty(self):
        return float(self.__options.penalty)

    def getAddSampleName(self, sample):
        return sample+self.__options.AddSampleNaming

    def getAddSampleSuffix(self):
        return self.__options.AddSampleNaming.replace("_dnn", "", 1)[:-3]
