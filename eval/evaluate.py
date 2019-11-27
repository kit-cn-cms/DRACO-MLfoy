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
##########################


# initialize list with columns to be written into dataframe
dataframe_columns = copy.deepcopy(variables)

#create df for event
eval_df = pd.DataFrame(columns = dataframe_columns)
df = pd.read_hdf(basedir+"/workdir/latest2017ttH/ttH_dnn.h5")	#test2_eval/All_Combs_dnn.h5")
df = df.reset_index(drop=True)

eval_DNN = DNN.DNN(
    save_path       = "outputdir",
    input_samples   = basedir+"/workdir/latest2017ttH/ttH_dnn.h5",
    event_category  = options.category,
    train_variables = options.variableSelection,
    # number of epochs
    #train_epochs    = int(options.train_epochs),
    # metrics for evaluation (c.f. KERAS metrics)
    eval_metrics    = ["acc"],
    # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
    test_percentage = 0.2,
    # balance samples per epoch such that there amount of samples per category is roughly equal
    balanceSamples  = False
)


#print df

for ind in variables:
	eval_df[ind] = df[ind].values

print eval_df[:20]

print "\n  done part 1  \n", variables

#############################
def loadDNN(inputDirectory, outputDirectory, binary = True, signal = "ttH", binary_target = 1., total_weight_expr = "1", category_cutString = None,
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


def findHiggs(dataframe,df, model,NN):
	model_predict = model.predict(dataframe.values, verbose=1)

	#model_predict = model.predict(NN.data.get_test_data (as_matrix = True), verbose=1)

	plt.hist(model_predict,bins = 100,range=(0,1))
	plt.show()


	#print model_predict, len(model_predict)
	best_index = np.zeros(11000)
	imax = -10
	ind = 0
	files = 0
	event_nr = 0
	perm = 0

	for iEvt in df.index:
		event = df.loc[iEvt]
		nJets = int(min(event["N_Jets"], 10))
		N_permutation = scipy.special.binom(nJets,2)
		if perm == N_permutation:
			imax = -10
			event_nr += 1
			perm = 0
		perm += 1

		if model_predict[ind] > imax:
			imax = model_predict[ind]
			if imax > 0.1:
				print model_predict[ind]
			best_index[event_nr] = int(iEvt)
		ind += 1
		if(imax<-1): print "error in model prediction!!"

	print best_index
	return best_index


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


###################################################################################################################################


model = loadDNN(inPath, "output")


BestIndex = findHiggs(eval_df,df, model, eval_DNN)

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
	#print "r1 :" ,minR1,"r2 " ,minR2,j
	if higgs1 == higgs2:
#		print "shit, B1 = B2"
		if (minR1 < minR2 and higgs1 == 1) or (minR1 > minR2 and higgs1 == 2):
			higgs1 = 1
			higgs2 = 2
		else:
			higgs1 = 2
			higgs2 = 1

	# Rekonstruktionsveranschaulichung
	if iEvt == BestIndex[0]:
		c0 = ROOT.TCanvas("c0", "richtige Jets in #eta - #phi - Ebene", 800,600)
		c0.SetTopMargin(0.15)
		c0.SetRightMargin(0.15)
		c0.SetBottomMargin(0.15)
		c0.SetLeftMargin(0.15)
	
		circleGen = ROOT.TEllipse(event["GenHiggs_Eta"], event["GenHiggs_Phi"], 0.4)
		circleGen.Draw()
	
		#for ind in range(nJets):
			#Jet = ROOT.TGraph(1,	event["Jet_Eta[{}]".format(ind)], event["Jet_Phi[{}]".format(ind)])
		#	Jet.SetMarkerstyle(41)
		#	Jet.Draw("SAME")


		circleGen1 = ROOT.TEllipse(event["GenHiggs_B1_Eta"], event["GenHiggs_B1_Phi"], 0.4)
		circleGen2 = ROOT.TEllipse(event["GenHiggs_B2_Eta"], event["GenHiggs_B2_Phi"], 0.4)
		circleGen1.Draw("SAME")
		circleGen2.Draw("SAME")

	#PtRap_thad = ROOT.TH2F("PtRap_Higgs_B1", " ; #eta(Higgs B1); p_{T}(Higgs B1) in GeV", 150, -5, 5, 200, 0, 600)
   	#PtRap_thad.Fill(event["Reco_Higgs_B1_Eta"],event["Reco_Higgs_B1_Pt"])




