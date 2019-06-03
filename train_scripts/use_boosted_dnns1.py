# global imports
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import sys
import optparse
import numpy

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

# import class for DNN training
import DRACO_Frameworks.DNN.DNN as DNN
import DRACO_Frameworks.DNN.data_frame as df
import keras.optimizers as optimizers

"""
USE: python use_boosted_dnns1.py -i DIR -o DIR --printroc
"""
usage="usage=%prog [options] \n"
usage+="USE: python use_boosted_dnns1.py -i DIR -o DIR --printroc"

parser = optparse.OptionParser(usage=usage)

parser.add_option("-i", "--inputdirectory", dest="inputDir",default="test_training_4j_ge3t",
        help="DIR of train data", metavar="inputDir")

parser.add_option("--netdirectory", dest="netDir",
        help="DIR of DIRs of trained net data", metavar="netDir")

parser.add_option("-o", "--outputdirectory", dest="outDir",default=None,
        help="DIR of evaluation outputs, if None specified use inputDir", metavar="outDir")

parser.add_option("-p", "--plot", dest="plot", action = "store_true", default=False,
        help="activate to create plots", metavar="plot")

parser.add_option("-l", "--log", dest="log", action = "store_true", default=False,
        help="activate for logarithmic plots", metavar="log")

parser.add_option("--printroc", dest="printROC", action = "store_true", default=False,
        help="activate to print ROC value for confusion matrix", metavar="printROC")

(options, args) = parser.parse_args()

#get input directory path
if not os.path.isabs(options.inputDir):
    inPath = basedir+"/workdir/"+options.inputDir
elif os.path.exists(options.inputDir):
    inPath=options.inputDir
else:
    sys.exit("ERROR: Input Directory does not exist!")

if os.path.exists(options.netDir):
    netPath=options.netDir
else:
    sys.exit("ERROR: Net Directory does not exist!")

if not options.outDir:
    outPath = inPath
elif not os.path.isabs(options.outDir):
    outPath = basedir+"/workdir/"+options.outDir
else:
    outPath = options.outDir

#get all subdirectories
netPaths = [dI for dI in os.listdir(netPath) if os.path.isdir(os.path.join(netPath,dI))]
print("Found dirs: ", netPaths)

#load data to evaluate
# for now only the evaluation part of the train data will be compared


#load dnns and evalueate them
prediction_list = []
for path in netPaths:
    path = netPath + path
    print("Path: ", path)
    dnn = DNN.loadDNN(path, outPath)
    prediction_list.append(dnn.model_prediction_vector)

print(type(prediction_list[0]))
prediction_vector = prediction_list[0]
print(type(prediction_vector))
#
# for i in range(1, len(prediction_list)):
#     prediction_vector = numpy.append(prediction_vector, prediction_list[i], axis = 0)

prediction_vector = numpy.asarray(prediction_list)
print(type(prediction_vector))
print(prediction_vector.shape)

#plot histogram over deviation from mean for each event
