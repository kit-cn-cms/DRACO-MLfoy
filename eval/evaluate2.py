import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
from math import sin, cos, log

import os
import sys
import optparse
import numpy as np
import pandas as pd
import json
from tensorflow import keras
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

parser.add_option("-v", "--variableselection", dest="variableSelection",default="trainX",
		help="FILE for variables used to train DNNs (allows relative path to variable_sets)", metavar="variableSelection")

parser.add_option("-c", "--category", dest="category",default="ge4j_ge3t",
		help="STR name of the category (ge/le)[nJets]j_(ge/le)[nTags]t", metavar="category")

parser.add_option("-i", "--inputdirectory", dest="inputDir",default="InputFeatures",
		help="DIR of trained dnn (definition of files to load has to be adjusted in the script itself)", metavar="inputDir")

parser.add_option("-d", "--dataframe", dest="dataframe", default=None,
		help="File of dataframe to evaluate", metavar="dataframe")

parser.add_option("--Reco", dest="Reco", default=None,
        help="activate reconstruction for Z or Higgs", metavar="Reco")

parser.add_option("--node", dest="node", default=0,
        help="DNN node to evaluate", metavar="node")

parser.add_option("--ratio", dest="ratio", action = "store_true", default=False,
        help="evaluate nodes of multiclass DNN with nodeXY/Othernode ratio", metavar="ratio")


#check for ttZ or ttH evaluation
(options, args) = parser.parse_args()
if options.Reco == "Higgs":
	particle = "Higgs"
	process = "ttH"
elif options.Reco == "Z":
	particle = "Z"
	process = "ttZ"
else:
	sys.exit("Select Higgs/Z with --Reco Z/Higgs")

node = int(options.node)

#get input directory path
if not os.path.isabs(options.inputDir):
	inPath = basedir+"/workdir/ttZ_DNNs/"+options.inputDir + "_" + options.category
elif os.path.exists(options.inputDir):
	inPath=options.inputDir
else:
	sys.exit("ERROR: Input Directory does not exist!")

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






# initialize list with columns to be written into dataframe
dataframe_columns = copy.deepcopy(variables)

#create df for event
if not os.path.exists(basedir+"/workdir/eval_dataframe/{}_eval/{}/eval_allCombs_dnn.h5".format(process, options.dataframe)):
	sys.exit("dataframe don't exists")

eval_df = pd.DataFrame(columns = dataframe_columns)
df = pd.read_hdf(basedir+"/workdir/eval_dataframe/{}_eval/{}/eval_allCombs_dnn.h5".format(process, options.dataframe))
if options.Reco == "Z":
	df = df[df["GenEvt_I_TTZ"]==1]


df = df.reset_index(drop=True)

print "\n  done part 1  \n", variables

#############################
def loadDNN(inputDirectory, outputDirectory, binary = True, signal = process, binary_target = 0., total_weight_expr = "1", category_cutString = None,
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
		shuffle_seed	= config["shuffleSeed"],
		evaluate_py		= True,
		)

	#print(dnn.data.values)
	checkpoint_path = inputDirectory+"/checkpoints/trained_model.h5py"

	# get the model
	dnn.model = keras.models.load_model(checkpoint_path)
	dnn.model.summary()

	return dnn.model


def findparticle(dataframe,df, model):
	model_predict = model.predict(dataframe.values, verbose=1)

	#plt.hist(model_predict,bins = 100,range=(0,1))
	#plt.show()

	best_index = [0]
	predictionVal = [0]
	null = [0]
	imax = -10
	event_nr = 0
	perm = 0
	N_permutation = -1
	nJets = -1
	node_ratio_sum = 0

	for iEvt in df.index:
	
		if iEvt%10000 == 0:
			print "Permutation",iEvt,"von",len(df.index)
		event = df.loc[iEvt]
		if nJets != int(min(event["N_Jets"], 10)) and perm != N_permutation and iEvt != 0:
			print "! Probably wrong Permutations !"

		nJets = int(min(event["N_Jets"], 10))

		if perm == N_permutation:

			if options.ratio :
				node_ratio_sum += model_predict[best_index[event_nr]][node]/float(np.sum(model_predict[best_index[event_nr]], dtype = float)-model_predict[best_index[event_nr]][node])

			imax = -10
			event_nr += 1
			best_index.extend(null)
			predictionVal.extend(null)
			perm = 0
		perm += 1
		N_permutation = scipy.special.binom(nJets,2)


		if model_predict[iEvt][node] > imax:
			imax = model_predict[iEvt][node]
			best_index[event_nr] = int(iEvt)
			predictionVal[event_nr] = imax


		if(imax<-1): print "error in model prediction!!"


	return best_index, predictionVal, node_ratio_sum


def normalize(df,inputdir):
	unnormed_df = df

	df_norms = pd.read_csv(inputdir+"/checkpoints/variable_norm.csv", index_col=0).transpose()
	print df.columns
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
		(event["Reco_"+particle+"_B"+ bjet + "_Eta"] - event[genVar+"_Eta"])**2 + (PhiDiff(event["Reco_"+particle+"_B"+ bjet +"_Phi"],event[genVar+"_Phi"]))**2
		)
def getDeltaR2(event, genVar,bjet):
	return np.sqrt(
		(event["Reco_Particle_B"+ bjet + "_Eta"] - event[genVar+"_Eta"])**2 + (PhiDiff(event["Reco_Particle_B"+ bjet +"_Phi"],event[genVar+"_Phi"]))**2
		)

def correct_phi(phi):
	if(phi  <=  -np.pi):
		phi += 2*np.pi
	if(phi  >	np.pi):
		phi -= 2*np.pi
	return phi

def plotBinary(predictions,valids, nevents, ratio = False, printROC = False, privateWork = False, name = "binary discriminator"):

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

	scaleFactor = (nevents-(sum(valids)))/(sum(valids)+1e-9)
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
	roc = roc_auc_score(valids, predictions)
	print("ROC-AUC = {}".format(roc))
	setup.printROCScore(canvas, roc, plotOptions["ratio"])

	# add category label
	setup.printCategoryLabel(canvas, options.category, ratio = plotOptions["ratio"])
	if not os.path.exists("{}/Eval_Plots".format(inPath)):
		os.mkdir("{}/Eval_Plots".format(inPath))
	out_path = inPath + "/Eval_Plots/binaryDiscriminator_{}.pdf".format(options.dataframe)
	setup.saveCanvas(canvas, out_path)

	


###################################################################################################################################

model = loadDNN(inPath, "output", signal= process)
eval_df = normalize(df[variables],inPath)


BestIndex, predVal, node_ratio_sum = findparticle(eval_df,df, model)

node_ratio = node_ratio_sum/len(BestIndex)


n = 0
valids = np.zeros(len(BestIndex))
b1b2counter = 0
valid_events = 0

valid_y = np.zeros(len(BestIndex)//1000+1)
valid_x = np.zeros(len(BestIndex)//1000+1)

for iEvt in BestIndex:

	# minR1 = 10000
	# minR2 = 10000
	deltaR = np.zeros(shape=(2,2))
	event = df.loc[iEvt]
	nJets = int(min(event["N_Jets"], 10))

	if n%1000 == 0:
		print "Event",n,"valid events",valid_events
		valid_y[n/1000] = valid_events
		valid_x[n/1000] = n
		


	if(nJets>10 or nJets<4):
		print "ok cool, next one"
		continue

	for Genj in [0,1]:
		for Recoj in [0,1]:
			deltaR[Genj][Recoj] = getDeltaR2(event, "Gen{}_B{}".format(particle,Genj+1),str(Recoj+1))


	if (deltaR[0][0] <= 0.4 and deltaR[1][1] <= 0.4) or (deltaR[1][0] <= 0.4 and deltaR[0][1] <= 0.4):
		valid_events +=1
		valids[n] = 1


	# for j in [1,2]:
	# 	deltaR1 = getDeltaR2(event, "Gen{}_B1".format(particle),str(j))
	# 	if deltaR1 < minR1:
	# 		minR1 = deltaR1
	# 		particle1 = j

	# 	deltaR2 = getDeltaR2(event, "Gen{}_B2".format(particle),str(j))
	# 	if deltaR2 < minR2:
	# 		minR2 = deltaR2
	# 		particle2 = j


	# if minR1 <= 0.4 and minR2 <= 0.4: 
	# 	valid_events+=1
	# 	valids[n] = 1



	n+=1



result, uncertainty = np.polyfit(valid_x, valid_y, 1 , cov=True)
#Binary Output Plot
plotBinary(predVal,valids,len(BestIndex), name="{}-Boson: {}% +/- {}%".format(particle,round(result[0]*100,2),round(np.sqrt(uncertainty[0][0])*100,2)))

resultfile = open(inPath + "/Eval_Plots/Evaluation_{}.txt".format(options.Reco),"w")
resultfile.write("Efficiency: {} % +/- {} %".format(result[0]*100, np.sqrt(uncertainty[0][0])*100))
resultfile.write("\nttZ/other node = {}".format(node_ratio))
resultfile.close()

summaryfile = open(basedir+"/workdir/"+process+"_Evaluation_Summary.txt","a")
summaryfile.write("{}:\nEfficiency: {} % +/- {} %".format(options.inputDir, result[0]*100, np.sqrt(uncertainty[0][0])*100))
summaryfile.write("\nttZ/other node = {}".format(node_ratio))
summaryfile.write("\n\n")
summaryfile.close()

print valid_events,"valid events,", result[0]*100,"% +/-",np.sqrt(uncertainty[0][0])*100,"%"
print "ttZ/other node", node_ratio
