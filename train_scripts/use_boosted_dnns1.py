# global imports
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import sys
import optparse

import numpy
import matplotlib.pyplot as plt

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

parser.add_option("--load_old_predictions", dest="load_old_predictions", default=False,
        help='set True if old prediction should be loaded')

(options, args) = parser.parse_args()

load_old_predictions = options.load_old_predictions
if(load_old_predictions):
    prediction_vector = numpy.load("/home/ngolks/Projects/boosted_dnn/prediction_array/array1.npy")
else:
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

    prediction_vector = numpy.asarray(prediction_list)
    numpy.save("/home/ngolks/Projects/boosted_dnn/prediction_array/array1.npy", prediction_vector)

# print("Shape of prediction_vector: ", prediction_vector.shape)
# print(prediction_vector[0])
# print("Minimum: ", numpy.amin(prediction_vector))
# print("Maximum: ", numpy.amax(prediction_vector))

prediction_vector_eventmean = numpy.mean(prediction_vector, axis = 0)
nnets = prediction_vector.shape[0]
data_len = prediction_vector.shape[1]

#plot histogram over deviation from mean for each event
# for i in range(0, prediction_vector.shape[0]):
#     delta = prediction_vector[i] - prediction_vector_eventmean
#     plt.hist(delta, alpha=0.5)
# plt.show()

#plot histogram over output deviation between two nets for each event
for h in numpy.arange(0, nnets):
    for j in numpy.arange(1+i, nnets):
        title = "Compare prediction between two DNNs (" + str(h) + "," + str(j) +")"
        out = "/home/ngolks/Projects/boosted_dnn/plotts/difference/diff" + str(h) + "_" + str(j) + ".png"
        c1=ROOT.TCanvas("c1","Data", 200, 10, 700, 500)
        c1.Divide(1,1)
        c1.cd(1)
        print("# DEBUG: h, j: ", h, j)
        hist = ROOT.TH1D("hist", "", 25,-0.3,0.3)
        for i in numpy.arange(0, data_len):
            hist.Fill(prediction_vector[h][i] - prediction_vector[j][i]
        hist.SetTitle(title)
        hist.Draw()
        c1.Print(out)
