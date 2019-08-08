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
from sklearn.metrics import roc_auc_score

def label_roc(num1, num2, roc1, roc2):
    nums = [num1, num2]
    rocs = [roc1, roc2]
    i = 0
    high = 0.89
    while i < 2:
        text = "ROC of network " + str(nums[i]) + ": " + str(rocs[i])
        label_roc = ROOT.TText()
        label_roc.SetNDC()
        label_roc.SetTextFont(1)
        label_roc.SetTextColor(1)
        label_roc.SetTextSize(0.03)
        label_roc.SetTextAlign(2)
        label_roc.SetTextAngle(0)
        label_roc.DrawText(0.14, high, text)
        high -= 0.02
        i += 1

def label_correlation(corr):
    text = "Correlation between networks: " + str(corr)
    label_corr = ROOT.TText()
    label_corr.SetNDC()
    label_corr.SetTextFont(1)
    label_corr.SetTextColor(1)
    label_corr.SetTextSize(0.03)
    label_corr.SetTextAlign(2)
    label_corr.SetTextAngle(0)
    label_corr.DrawText(0.14, 0.88, text)

def Get_roc_auc(data, prediction_vector):
    return roc_auc_score(data.get_test_labels(), prediction_vector)


"""
USE: python use_boosted_dnns1.py -o DIR --netdirectory DIR -c category --load_old_predictions True/False
"""
usage="usage=%prog [options] \n"
usage+="USE: python use_boosted_dnns1.py -i DIR -o DIR --printroc"

parser = optparse.OptionParser(usage=usage)

# parser.add_option("-i", "--inputdirectory", dest="inputDir",default="test_training_4j_ge3t",
#         help="DIR of train data", metavar="inputDir")

parser.add_option("--netdirectory", dest="netDir",
        help="DIR of DIRs of trained net data", metavar="netDir")

parser.add_option("-o", "--outputdirectory", dest="outDir",default=None,
        help="DIR of evaluation outputs, if None specified use inputDir", metavar="outDir")

parser.add_option("-c", "--category", dest="category",default="4j_ge3t",
        help="STR name of the category (ge/le)[nJets]j_(ge/le)[nTags]t", metavar="category")

# parser.add_option("-p", "--plot", dest="plot", action = "store_true", default=False,
        # help="activate to create plots", metavar="plot")

# parser.add_option("-l", "--log", dest="log", action = "store_true", default=False,
        # help="activate for logarithmic plots", metavar="log")

# parser.add_option("--printroc", dest="printROC", action = "store_true", default=False,
        # help="activate to print ROC value for confusion matrix", metavar="printROC")

parser.add_option("--load_old_predictions", dest="load_old_predictions", default=False,
        help='set True if old prediction should be loaded')

(options, args) = parser.parse_args()

# #get input directory path of data -> doesn't needed any longer because data is loaded from dnn
# if not os.path.isabs(options.inputDir):
#     inPath = basedir+"/workdir/"+options.inputDir
# elif os.path.exists(options.inputDir):
#     inPath=options.inputDir
# else:
#     sys.exit("ERROR: Input Directory does not exist!")
#load data to evaluate
# validation_labels = Get_test_labels(inPath)


#get prediction_vector and roc_vector
load_old_predictions = options.load_old_predictions
if(load_old_predictions):
    prediction_vector = numpy.load("/home/ngolks/Projects/boosted_dnn/prediction_array/prediction_vector.npy")
    roc_vector = numpy.load("/home/ngolks/Projects/boosted_dnn/prediction_array/roc_vector.npy")
else:
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
    netPaths = []
    netPaths_all = [dI for dI in os.listdir(netPath) if os.path.isdir(os.path.join(netPath,dI))]    #containing all directorys
    category = options.category         #wanted categorie (number of wanted dnns should also be here)
    for dirname in netPaths_all:
        if category in dirname:
            # print("Used directory: ", dirname)
            netPaths.append(dirname)
    print("Found dirs: ", netPaths)

    #load dnns and evalueate them
    prediction_list = []
    dnn_roc = []
    for path in netPaths:
        path = netPath + path
        print("Path: ", path)
        dnn = DNN.loadDNN(path, outPath)
        prediction_list.append(dnn.model_prediction_vector)
        dnn_roc.append(dnn.roc_auc_score)
    data = dnn.data #get data class

    prediction_vector = numpy.asarray(prediction_list)
    roc_vector = numpy.asarray(dnn_roc)
    numpy.save("/home/ngolks/Projects/boosted_dnn/prediction_array/prediction_vector.npy", prediction_vector)

#some information needed
nnets = prediction_vector.shape[0]
data_len = prediction_vector.shape[1]

#new prediction_vector merged from all dnns
prediction_vector_eventmean = numpy.mean(prediction_vector, axis = 0)   #mean of all nets for each event
#new prediction_vector merged from all dnns weighted with roc
prediction_vector_eventrocmean = numpy.average(prediction_vector, axis = 0, weights = roc_vector[0:nnets])
if(load_old_predictions):
    roc_formean = roc_vector[-2]
    roc_forweightedmean = roc_vector[-1]
else:
    roc_formean = Get_roc_auc(data, prediction_vector_eventmean)
    dnn_roc.append(roc_formean)
    roc_forweightedmean = Get_roc_auc(data, prediction_vector_eventrocmean)
    dnn_roc.append(roc_forweightedmean)
    roc_vector = numpy.asarray(dnn_roc)
    numpy.save("/home/ngolks/Projects/boosted_dnn/prediction_array/roc_vector.npy", roc_vector)

#plot histogram over output deviation between two nets for each event
for h in numpy.arange(0, nnets-1):
    for j in numpy.arange(1+h, nnets):
        title = "Compare prediction between two DNNs (" + str(h) + "," + str(j) +")"
        out = "/home/ngolks/Projects/boosted_dnn/plotts/difference/diff" + str(h) + "_" + str(j) + ".pdf"
        c1=ROOT.TCanvas("c1","Data", 200, 10, 700, 500)
        c1.Divide(2,1)
        c1.cd(1)
        hist = ROOT.TH1D("hist", "", 25,-0.3,0.3)
        for i in numpy.arange(0, data_len):
            hist.Fill(prediction_vector[h][i] - prediction_vector[j][i])
        hist.SetTitle(title)
        hist.Draw()
        label_roc(h, j, roc_vector[h], roc_vector[j])      #write down the roc output
        c1.cd(2)
        hist2=ROOT.TH2D("hist", "", 40, -1, 1, 40, -1, 1)
        for i in numpy.arange(0, data_len):
            hist2.Fill(prediction_vector[h][i], prediction_vector[j][i])
        hist2.Draw("colz")
        label_correlation(hist2.GetCorrelationFactor())
        c1.Print(out)

#plot histogram over deviation from output and mean for each event
for j in numpy.arange(0, nnets):
    title = "Compare prediction and mean"
    out = "/home/ngolks/Projects/boosted_dnn/plotts/difference/diff_tomean" + str(j) + ".pdf"
    c1=ROOT.TCanvas("c1","Data", 200, 10, 700, 500)
    c1.Divide(2,1)
    c1.cd(1)
    hist = ROOT.TH1D("hist", "", 25,-0.3,0.3)
    for i in numpy.arange(0, data_len):
        hist.Fill(prediction_vector[j][i] - prediction_vector_eventmean[i])
    hist.SetTitle(title)
    hist.Draw()
    label_roc(j, "mean", roc_vector[j], roc_formean)
    c1.cd(2)
    hist2=ROOT.TH2D("hist", "", 40, -1, 1, 40, -1, 1)
    for i in numpy.arange(0, data_len):
        hist2.Fill(prediction_vector[j][i], prediction_vector_eventmean[i])
    hist2.Draw("colz")
    label_correlation(hist2.GetCorrelationFactor())
    c1.Print(out)

#plot histogram over deviation from output and weighted (-> roc) mean for each event
for j in numpy.arange(0, nnets):
    title = "Compare prediction and mean"
    out = "/home/ngolks/Projects/boosted_dnn/plotts/difference/diff_torocmean" + str(j) + ".pdf"
    c1=ROOT.TCanvas("c1","Data", 200, 10, 700, 500)
    c1.Divide(2,1)
    c1.cd(1)
    # print("# DEBUG: h, j: ", h, j)
    hist = ROOT.TH1D("hist", "", 25,-0.3,0.3)
    for i in numpy.arange(0, data_len):
        hist.Fill(prediction_vector[j][i] - prediction_vector_eventrocmean[i])
    hist.SetTitle(title)
    hist.Draw()
    label_roc(j, "weighted_mean", roc_vector[j], roc_forweightedmean)
    c1.cd(2)
    # print(prediction_vector[j].reshape(data_len,)[0:10])
    hist2=ROOT.TH2D("hist", "", 40, -1, 1, 40, -1, 1)
    for i in numpy.arange(0, data_len):
        hist2.Fill(prediction_vector[j][i], prediction_vector_eventrocmean[i])
    hist2.Draw("colz")
    label_correlation(hist2.GetCorrelationFactor())
    c1.Print(out)
