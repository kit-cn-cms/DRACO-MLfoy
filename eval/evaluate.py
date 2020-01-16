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
df = pd.read_hdf(basedir+"/workdir/eval_dataframes/eval_df_10k/eval_allCombs_dnn.h5") 
nevents = len(np.unique(df.index.get_level_values(2)))
df = df.reset_index(drop=True)


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

	best_index = np.zeros(nevents)
	predictionVal = np.zeros(nevents)
	imax = -10
	files = 0
	event_nr = 0
	perm = 0
	N_permutation = -1
	nJets = -1

	for iEvt in df.index:
		if iEvt%10000 == 0:
			print "Event",event_nr,"von",nevents
		event = df.loc[iEvt]
		if nJets != int(min(event["N_Jets"], 10)) and perm != N_permutation and iEvt != 0:
			print "! Probably wrong Permutations !"

		nJets = int(min(event["N_Jets"], 10))

		if perm == N_permutation:
			imax = -10
			event_nr += 1
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
	sig_weights = np.full(nevents,2)
	sig_hist = setup.setupHistogram(
		values      = sig_values,
		weights     = sig_weights,
		nbins       = 30,
		bin_range   = [0.,1.],
		color       = ROOT.kCyan,
		xtitle      = "signal",
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
		xtitle      = "background",
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
	legend.AddEntry(sig_hist, "signal x {:4.0f}".format(scaleFactor), "L")
        
	# add background entries
	legend.AddEntry(bkg_hist, "background", "F")

	# draw legend
	legend.Draw("same")

#	# add ROC score if activated
#	if self.printROCScore:
#		setup.printROCScore(canvas, roc, plotOptions["ratio"])

	# add category label
	setup.printCategoryLabel(canvas, options.category, ratio = plotOptions["ratio"])

	out_path = inPath + "/Eval_Plots/binaryDiscriminator.pdf"
	setup.saveCanvas(canvas, out_path)

###################################################################################################################################


model = loadDNN(inPath, "output")

eval_df = normalize(df[variables],inPath)
BestIndex, predVal = findHiggs(eval_df,df, model)

PtRap_b1 = ROOT.TH2F("PtRap_Higgs_B1", " ; #eta(Higgs B1); p_{T}(Higgs B1) in GeV", 150, -5, 5, 200, 0, 600)
pt_eff = ROOT.TEfficiency("pt_eff", " ;p_{T}) in GeV; Effizienz", 60,0,600)
pt_eff_2 = ROOT.TH1F("pt_eff_2", " ;Transversalimpuls p_{T} in GeV; Effizienz", 60,0,600)
reco_Higgs_M = ROOT.TH1F("reco_M", " ;Masse des Rekonstruierten Higgs-Bosons in GeV; Events", 60,0,600)
#gen_M = ROOT.TH1F("reco_M", " ;Masse des Rekonstruierten Higgs-Bosons in GeV; Events", 60,0,600)
n = 0
valids = np.zeros(nevents)
b1b2counter = 0
valid_events = 0

for iEvt in BestIndex:

	minR1 = 10000
	minR2 = 10000
	event = df.loc[iEvt]
	nJets = int(min(event["N_Jets"], 10))

	if(nJets>10 or nJets<4):
		print "ok cool, next one"
		continue

	for j in [1,2]:
		deltaR1 = getDeltaR(event, "GenHiggs_B1",str(j))
		if deltaR1 < minR1:
			minR1 = deltaR1
			higgs1 = j

		deltaR2 = getDeltaR(event, "GenHiggs_B2",str(j))
		if deltaR2 < minR2:
			minR2 = deltaR2
			higgs2 = j

	if minR1 <= 0.4 and minR2 <= 0.4: 
		valid_events+=1
		valids[n] = 1


	if higgs1 == higgs2:
		#print "shit, B1 = B2"
		b1b2counter+=1
		if (minR1 < minR2 and higgs1 == 1) or (minR1 > minR2 and higgs1 == 2):
			higgs1 = 1
			higgs2 = 2
		else:
			higgs1 = 2
			higgs2 = 1
	if n%1000 == 0:
		print "Event",n,"minR1", minR1,"valid events",valid_events


	n+=1

	pt_eff.Fill((minR1<=0.4 and minR2 <=0.4), event["Reco_Higgs_Pt"])

	if minR1 < 0.4:
		PtRap_b1.Fill(event["Reco_Higgs_B1_Eta"],event["Reco_Higgs_B1_Pt"])
	reco_Higgs_M.Fill(event["Reco_Higgs_M"])
	#gen_M.Fill(event["GenHiggs_B1_M"]+event["GenHiggs_B2_M"])


#Binary Output Plot
plotBinary(predVal,valids)

#Efficiency Plot
for i in range(valid_events):
    pt_eff_2.SetBinContent(i, pt_eff.GetEfficiency(i))
    pt_eff_2.SetBinError(i, pt_eff.GetEfficiencyErrorLow(i))

c2 = ROOT.TCanvas("c2", "quality of reconstruction", 700,600)
c2.SetRightMargin(0.15)
c2.SetLeftMargin(0.15)
c2.SetBottomMargin(0.15)
c2.SetTopMargin(0.15)

pt_eff_2.SetFillColor(ROOT.kBlue)
pt_eff_2.SetStats(0)
pt_eff_2.SetTitleSize(0.05,"xy")
pt_eff_2.Draw("E3")

c2.SaveAs(inPath + "/Eval_Plots/Efficiency.pdf")

# Pt ueber Eta Plot
c3 = ROOT.TCanvas("c3", "quality of reconstruction", 700,600)
c3.DrawFrame(-2.5,0,2.5,600)
c3.SetRightMargin(0.15)
c3.SetLeftMargin(0.15)
c3.SetBottomMargin(0.15)
c3.SetTopMargin(0.15)

PtRap_b1.SetStats(0)
PtRap_b1.SetTitleSize(.05, "xy")
PtRap_b1.Draw("COLZ")


c3.SaveAs(inPath + "/Eval_Plots/PtRap.pdf")

#Higgs M Plot
c4 = ROOT.TCanvas("c2", "quality of reconstruction", 700,600)
c4.SetRightMargin(0.15)
c4.SetLeftMargin(0.15)
c4.SetBottomMargin(0.15)
c4.SetTopMargin(0.15)

reco_Higgs_M.SetStats(0)
reco_Higgs_M.SetTitleSize(0.05,"xy")
reco_Higgs_M.Draw("C")

#gen_M.SetStats(0)
#gen_M.SetTitleSize(0.05,"xy")
#gen_M.Draw("C")

c4.SaveAs(inPath + "/Eval_Plots/Higgs_M.pdf")

print valid_events,"valid events,", valid_events/float(nevents)*100,"%"

print b1b2counter,"mal b1=b2, von",nevents,"events;\t",b1b2counter/float(nevents)*100,"%"




