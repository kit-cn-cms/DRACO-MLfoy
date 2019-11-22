# global imports
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import sys
import json


# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

# import class for DNN training
import DRACO_Frameworks.DNN.DNN as DNN 
from pyrootsOfTheCaribbean.evaluationScripts import plottingScripts
from utils.DNNconfig import dnnmodels, outputDirectory, inputData, shuffleSeed, signal_class

Histograms = {}
for model in dnnmodels:
	for dataera in dnnmodels[model]["data_era"]:
		Histograms[dataera] = {}
for model in dnnmodels:
	for dataera in dnnmodels[model]["data_era"]:
		Histograms[dataera][model] = {}

print "-"*40
print Histograms
print "-"*40


for model in dnnmodels:
	for dataera in dnnmodels[model]["data_era"]:
		dnns={}
		for n,dnn in enumerate(dnnmodels[model]["DNNs"]):
			inputDirectory = dnnmodels[model]["DNNs"][n]
			dnnName="dnn"+str(n)
			dnns[dnnName] = DNN.loadDNN(inputDirectory= inputDirectory, outputDirectory=outputDirectory, inputData=inputData, data_era=[dataera], shuffleSeed=shuffleSeed)
			#binning = dnnmodels[model]["binning"]
			binning = None
			histname=dnnName+str(model)+str(dataera)
			dnns[dnnName].get_discriminators(signal_class=signal_class, tag=histname, binning=binning, binflag=binning)

		print dnns[dnnName].Histograms

		
		for node in dnns[dnnName].Histograms:
			print "Node: {}".format(node)
			Histograms[dataera][model][node]={}
			for process in dnns[dnnName].Histograms[node]:
				print "Process: {}".format(process)
				Histograms[dataera][model][node][process]={}
				for dnn in dnns:
					print "DNN number: {}".format(dnn)
					Histograms[dataera][model][node][process][dnn]=dnns[dnn].Histograms[node][process]


for model in Histograms["2016"]:
	print "-"*40
	print model
	print Histograms["2016"][model]

plotpath=basedir+"/plotdir/ge4j_ge4t/meanandstd/"
meanandstd = plottingScripts.meanAndMeanError(Histograms, plotpath, dnns[dnnName].category_label, model)
meanandstd.calculate()


