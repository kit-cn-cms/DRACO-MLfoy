import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
from math import sin, cos, log

import os
import sys
import optparse
import numpy as np
import pandas as pd
import json
import keras
import copy
import math
import matplotlib.pyplot as plt
import scipy.special
from sklearn.metrics import roc_auc_score
# import class for DNN training
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)
import DRACO_Frameworks.DNN.DNN as DNN
import DRACO_Frameworks.DNN.data_frame as data_frame
import plot_configs.setupPlots as setup

from pyrootsOfTheCaribbean.evaluationScripts import plottingScripts


"""
USE: python preprocessing.py --outputdirectory=DIR --variableSelection=FILE --maxentries=INT --MEM=BOOL
"""
usage="usage=%prog [options] \n"
usage+="USE: python preprocessing.py --outputdirectory=DIR --variableselection=FILE --maxentries=INT --MEM=BOOL --name=STR\n"
usage+="OR: python preprocessing.py -o DIR -v FILE -e INT -m BOOL -n STR"

parser = optparse.OptionParser(usage=usage)

parser.add_option("-v", "--variableselection", dest="variableSelection",default="trainHiggs",
		help="FILE for variables used to train DNNs (allows relative path to variable_sets)", metavar="variableSelection")

parser.add_option("-c", "--category", dest="category",default="ge4j_ge3t",
		help="STR name of the category (ge/le)[nJets]j_(ge/le)[nTags]t", metavar="category")

parser.add_option("-i", "--inputdirectory", dest="inputDir",default="InputFeatures",
		help="DIR of trained dnn (definition of files to load has to be adjusted in the script itself)", metavar="inputDir")

parser.add_option("-p", "--percentage", dest="percentage", default="100",
		help="Type 1 for around 1%, 10 for 10 and 100 for 100", metavar="percentage")

parser.add_option("-e", "--events", dest="events", default=10000000,
		help="maximum number of events (default 10M)", metavar="events")

parser.add_option("-s", "--schalter", dest="schalter", default="0",
		help="number of plots of reconstruction you want to create", metavar="schalter")

#parser.add_option("-d", "--dataframe", dest="datafr",default="test",
#		help="DIR of h5 files", metavar="datafr")

(options, args) = parser.parse_args()
#get input directory path
if not os.path.isabs(options.inputDir):
	inPath = basedir+"/workdir/DNNs/"+options.inputDir + "_" + options.category
elif os.path.exists(options.inputDir):
	inPath=options.inputDir
else:
	sys.exit("ERROR: Input Directory does not exist!")
#get df directory path
#if not os.path.isabs(options.datafr):
#	dfPath = basedir+"/workdir/"+options.datafr
#elif os.path.exists(options.datafr):
#	dfPath=options.datafr
#else:
#	sys.exit("ERROR: DataFrame Directory does not exist!")
#import Variable Selection
if not os.path.isabs(options.variableSelection):
	sys.path.append(basedir+"/variable_sets/")
	variable_set = __import__(options.variableSelection)
elif os.path.exists(options.variableSelection):
	variable_set = __import__(options.variableSelection)
else:
	sys.exit("ERROR: Variable Selection File does not exist!")
	# the input variables are loaded from the variable_set file
if options.category in variable_set.variables:
	variables = variable_set.variables[options.category]
else:
	variables = variable_set.all_variables
	print("category {} not specified in variable set {} - using all variables".format(
		options.category, options.variableSelection))

if options.percentage=="1":
	xx="*00"
elif options.percentage=="10":
	xx="*0"
elif options.percentage=="100":
	xx="*"
else:
	print("ERROR: Please enter 1, 10 or 100 as percentage of files you want to evaluate")

if int(options.events):
	EVENTS = int(options.events)
else:
	print("ERROR: Please enter number bigger than 0")

schalter = int(options.schalter)



# initialize list with columns to be written into dataframe
dataframe_columns = copy.deepcopy(variables)

#create df for event
eval_df = pd.DataFrame(columns = dataframe_columns)
df = pd.read_hdf(basedir+"/workdir/eval_dataframes/eval_df_10k3/eval_allCombs_dnn.h5") 
nevents = len(np.unique(df.index.get_level_values(2)))
df = df.reset_index(drop=True)

eval_df_ttbar = pd.DataFrame(columns = dataframe_columns)
df_ttbar = pd.read_hdf(basedir+"/workdir/eval_dataframes/ttbar_evalDf/eval_allCombs_dnn.h5") 
nevents_ttbar = len(np.unique(df_ttbar.index.get_level_values(2)))
df_ttbar = df_ttbar.reset_index(drop=True)


print "\n  done part 1  \n", variables

#############################
def loadDNN(inputDirectory, outputDirectory, binary = True, signal = "ttH", binary_target = 0., total_weight_expr = "1", category_cutString = None,
category_label= None):

	# get net config json
	configFile = inputDirectory+"/checkpoints/net_config.json"
	if not os.path.exists(configFile):
		sys.exit("config needed to load trained DNN not found\n{}".format(configFile))

	with open(configFile) as f:
		config = f.read()
	config = json.loads(config)

	# load samples
	input_samples = data_frame.InputSamples(config["inputData"])

	for sample in config["eventClasses"]:
		input_samples.addBinaryLabel(signal,binary_target)

	print("shuffle seed: {}".format(config["shuffleSeed"]))
	# init DNN class
	dnn = DNN.DNN(
		save_path	   = outputDirectory,
		input_samples   = input_samples,
		event_category  = config["JetTagCategory"],
		train_variables = config["trainVariables"],
		shuffle_seed	= config["shuffleSeed"]
		)

	#print(dnn.data.values)
	checkpoint_path = inputDirectory+"/checkpoints/trained_model.h5py"

	# get the model
	dnn.model = keras.models.load_model(checkpoint_path)
	dnn.model.summary()

	return dnn.model


def findHiggs(dataframe,df, model):
	model_predict = model.predict(dataframe.values, verbose=1)

	#plt.hist(model_predict,bins = 100,range=(0,1))
	#plt.show()

	best_index = np.zeros(nevents_ttbar)
	predictionVal = np.zeros(nevents_ttbar)
	imax = -10
	files = 0
	event_nr = 0
	perm = 0
	N_permutation = -1
	nJets = -1

	for iEvt in df.index:

		if iEvt%10000 == 0:
			print "Event",event_nr,"von",nevents_ttbar
		event = df.loc[iEvt]
		if nJets != int(min(event["N_Jets"], 10)) and perm != N_permutation and iEvt != 0:
			print "! Probably wrong Permutations !"

		nJets = int(min(event["N_Jets"], 10))

		if perm == N_permutation:
			imax = -10
			event_nr += 1
			if event_nr == nevents_ttbar:
				break
			perm = 0
		perm += 1
		N_permutation = scipy.special.binom(nJets,2)

		if model_predict[iEvt] > imax:
			imax = model_predict[iEvt]
			best_index[event_nr] = int(iEvt)
			predictionVal[event_nr] = imax

		if(imax<-1): print "error in model prediction!!"


	#print best_index

	return best_index, predictionVal


def normalize(df,inputdir):
	unnormed_df = df

	df_norms = pd.read_csv(inputdir+"/checkpoints/variable_norm.csv", index_col=0).transpose()
	for ind in df.columns:
		df[ind] = (unnormed_df[ind] - df_norms[ind][0])/df_norms[ind][1]
	return df

def PhiDiff(phi1,phi2):
	if abs(phi1-phi2) > np.pi:
		return 2*np.pi - abs(phi1-phi2)
	else:
		return abs(phi1-phi2)
def GenHiggsPhi(phi,event):
	return correct_phi(phi - event["GenHiggs_Phi"])
	
def getDeltaR(event, genVar,bjet):
	return np.sqrt(
		(event["Reco_Higgs_B"+ bjet + "_Eta"] - event[genVar+"_Eta"])**2 + (PhiDiff(event["Reco_Higgs_B"+ bjet +"_Phi"],event[genVar+"_Phi"]))**2
		)

def correct_phi(phi):
	if(phi  <=  -np.pi):
		phi += 2*np.pi
	if(phi  >	np.pi):
		phi -= 2*np.pi
	return phi

def plotBinary(predictions,valids, ratio = False, printROC = False, privateWork = False, name = "binary discriminator"):

	sig_values = [ predictions[k] for k in range(len(predictions)) \
		if valids[k] == 1 ]
	sig_weights = np.ones(nevents)
	sig_hist = setup.setupHistogram(
		values      = sig_values,
		weights     = sig_weights,
		nbins       = 30,
		bin_range   = [0.,1.],
		color       = ROOT.kCyan,
		xtitle      = "ttH Event",
		ytitle      = "Events expected",
		filled      = False)  
	sig_hist.SetLineWidth(3)

	bkg_values = [ predictions[k] for k in range(len(predictions)) \
		if not valids[k] == 1 ]
	bkg_weights = np.ones(nevents)
	bkg_hist = setup.setupHistogram(
		values      = bkg_values,
		weights     = bkg_weights,
		nbins       = 30,
		bin_range   = [0.,1.],
		color       = ROOT.kOrange,
		xtitle      = "ttbar Event",
		ytitle      = "Events expected",
		filled      = True)  

	scaleFactor = sum(bkg_weights)/(sum(sig_weights)+1e-9)
	sig_hist.Scale(scaleFactor)

	plotOptions = {
		"ratio":      ratio,
		"ratioTitle": "#frac{scaled Signal}{Background}",
		"logscale":   False}

	# initialize canvas
	canvas = setup.drawHistsOnCanvas(
		sig_hist, bkg_hist, plotOptions, 
		canvasName = name)

	# setup legend
	legend = setup.getLegend()

	# add signal entry
	legend.AddEntry(sig_hist, "ttH sample x {:4.0f}".format(scaleFactor), "L")
        
	# add background entries
	legend.AddEntry(bkg_hist, "ttbar sample", "F")

	# draw legend
	legend.Draw("same")

#	# add ROC score if activated
#	if self.printROCScore:
#		setup.printROCScore(canvas, roc, plotOptions["ratio"])
        roc = roc_auc_score(valids,predictions)
        print("ROC: {}".format(roc))
	setup.printROCScore(canvas, roc, plotOptions["ratio"])

	#private work label
	setup.printPrivateWork(canvas, plotOptions["ratio"], nodePlot = True)

	# add category label
	setup.printCategoryLabel(canvas, options.category, ratio = plotOptions["ratio"])

	out_path = basedir +"/workdir/Vergleich/binaryDiscriminator.pdf"
	setup.saveCanvas(canvas, out_path)

###################################################################################################################################


model = loadDNN(inPath, "output")

eval_df = normalize(df[variables],inPath)
BestIndex, predVal = findHiggs(eval_df,df, model)

eval_df_ttbar = normalize(df_ttbar[variables],inPath)
BestIndex_ttbar, predVal_ttbar = findHiggs(eval_df_ttbar,df_ttbar, model)

PtRap_b1 = ROOT.TH2F("PtRap_Higgs_B1", " ; #eta(Higgs B1); p_{T}(Higgs B1) in GeV", 150, -5, 5, 200, 0, 600)
pt_eff = ROOT.TEfficiency("pt_eff", " ;p_{T}) in GeV; Effizienz", 60,0,600)
pt_eff_2 = ROOT.TH1F("pt_eff_2", " ;Transversalimpuls p_{T} in GeV; Effizienz", 60,0,600)
reco_Higgs_M = ROOT.TH1F("reco_M", " ;Reco Higgs Masse in GeV; Events", 60,0,600)
reco_Higgs_M_ttbar = ROOT.TH1F("reco_M", " ;Masse des Rekonstruierten Higgs-Bosons in GeV; Events", 60,0,600)
gen_M = ROOT.TH1F("reco_M", " ;Masse des Rekonstruierten Higgs-Bosons in GeV; Events", 60,0,600)
n = 0
valids = np.zeros(2*nevents_ttbar)
predVal12 = np.zeros(2*nevents_ttbar)





for iEvt in BestIndex:

	event = df.loc[iEvt]

	gen_M.Fill(event["GenHiggs_M"])

	valids[n] = 1
	predVal12[n] = predVal[n]

	nJets = int(min(event["N_Jets"], 10))

	if n%10000 == 0:
		print "Event",n#,"minR1", minR1,"valid events",valid_events

	n+=1
	reco_Higgs_M.Fill(event["Reco_Higgs_M"])

n=0
for iEvt in BestIndex_ttbar:

	predVal12[n+nevents_ttbar] = predVal_ttbar[n]
	event = df_ttbar.loc[iEvt]
	nJets = int(min(event["N_Jets"], 10))

	if n%10000 == 0:
		print "Event",n #,"minR1", minR1,"valid events",valid_events

	n+=1
	reco_Higgs_M_ttbar.Fill(event["Reco_Higgs_M"])


#Binary Output Plot
plotBinary(predVal12,valids)


#Higgs M Plot
c4 = ROOT.TCanvas("c2", "quality of reconstruction", 700,600)
c4.SetRightMargin(0.15)
c4.SetLeftMargin(0.15)
c4.SetBottomMargin(0.15)
c4.SetTopMargin(0.15)

reco_Higgs_M.SetStats(0)
reco_Higgs_M.SetTitleSize(0.05,"xy")
reco_Higgs_M.SetLineColor(ROOT.kRed)
reco_Higgs_M.Draw("C")

reco_Higgs_M_ttbar.SetStats(0)
reco_Higgs_M_ttbar.SetTitleSize(0.05,"xy")
reco_Higgs_M_ttbar.Draw("SAME C")

gen_M.SetStats(0)
gen_M.SetLineColor(ROOT.kBlack)
gen_M.SetTitleSize(0.05,"xy")
gen_M.Draw("SAME C")

#Higgs_M = ROOT.TLine(125,0,125,8000)
#Higgs_M.Draw("SAME")

legend = ROOT.TLegend(0.75,0.55, 1., 0.85)
legend.AddEntry(reco_Higgs_M, "ttH sample", "L")
legend.AddEntry(reco_Higgs_M_ttbar, "ttbar sample", "L")
legend.AddEntry(gen_M, "Gen Higgs Masse", "L")
legend.Draw("SAME")

c4.SaveAs(basedir +"/workdir/Vergleich/Higgs_M.pdf")

#print valid_events,"ttH events,", valid_events/float(nevents)*100,"%"

#print b1b2counter,"mal b1=b2, von",nevents,"events;\t",b1b2counter/float(nevents)*100,"%"




