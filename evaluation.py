# global imports
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import sys
import keras.optimizers as optimizers
import optparse
import json


# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

# import class for DNN training
import DNNClass as DNN

"""
USE: python evaluation.py -o DIR -v FILE -n STR -c STR -e INT -s INT -p -l --privatework --netconfig=STR --signalclass=STR --printroc
"""
usage="usage=%prog [options] \n"
usage+="USE: python train_template.py -o DIR -v FILE -n STR -c STR -e INT -s INT -p -l --privatework --netconfig=STR --signalclass=STR --printroc "

parser = optparse.OptionParser(usage=usage)

parser.add_option("-i", "--inputdirectory", dest="inputDir",default="test_training_ge4j_ge4t",
        help="DIR of trained net data", metavar="inputDir")

parser.add_option("-m", "--modelcheckpointsdirectory", dest="modcheckDir",default="test_training_ge4j_ge4t",
        help="DIR of trained net data", metavar="modcheckDir")

parser.add_option("-o", "--outputdirectory", dest="outDir",default=None,
        help="DIR of evaluation outputs, if None specified use inputDir", metavar="outDir")

parser.add_option("-p", "--plot", dest="plot", action = "store_true", default=False,
        help="activate to create plots", metavar="plot")

parser.add_option("-l", "--log", dest="log", action = "store_true", default=False,
        help="activate for logarithmic plots", metavar="log")

parser.add_option("--privatework", dest="privateWork", action = "store_true", default=False,
        help="activate to create private work plot label", metavar="privateWork")

(options, args) = parser.parse_args()

print(filedir)
print(basedir)



#get input directory path
if not os.path.isabs(options.inputDir):
    inPath = filedir+"/workdir/"+options.inputDir
elif os.path.exists(options.inputDir):
    inPath=options.inputDir
else:
    sys.exit("ERROR: Input Directory does not exist!")

print(inPath)

if not os.path.isabs(options.modcheckDir):
    modelDir = filedir+"/workdir/"+options.modcheckDir
elif os.path.exists(options.modcheckDir):
    modelDir=options.modcheckDir
else:
    sys.exit("ERROR: Model checkpoints Directory does not exist!")

print(modelDir)

if not options.outDir:
    outPath = inPath
elif not os.path.isabs(options.outDir):
    outPath = filedir+"/workdir/"+options.outDir
else:
    outPath = options.outDir

configFile = modelDir+"/checkpoints/net_config.json"
if not os.path.exists(configFile):
    sys.exit("config needed to load trained DNN not found\n{}".format(configFile))

with open(configFile) as f:
    config = f.read()
config = json.loads(config)

# load samples
input_samples = DNN.InputSamples(inPath)
naming = "_dnn.h5"

input_samples.addSample("ttHbb"+naming, label = "ttHbb")
input_samples.addSample("ttbb"+naming,  label = "ttbb")
input_samples.addSample("tt2b"+naming,  label = "tt2b")
input_samples.addSample("ttb"+naming,   label = "ttb")
input_samples.addSample("ttcc"+naming,  label = "ttcc")
input_samples.addSample("ttlf"+naming,  label = "ttlf")


# init DNN class
dnn = DNN.DNN(
    save_path       = outPath,
    input_samples   = input_samples,
    event_category  = config["JetTagCategory"],
    variables = config["trainVariables"]
    )

# load the trained model
dnn.load_trained_model(modelDir)

#dnn.predict_event_query()

dnn.plot_outputNodes(log = options.log, privateWork = options.privateWork)
