import ROOT
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

(options, args) = parser.parse_args()
if options.variableSelection == "trainHiggs":
	particle = "Higgs"
	process = "ttH"
	particle_M = 116.3
	particle_dR= 1.7 #geraten
	sigma_M  = 13.5
	sigma_dR = 5
elif options.variableSelection == "trainZ":
	particle = "Z"
	process = "ttZ"
	particle_M = 86.8
	particle_dR= 1.2 #geraten
	sigma_M  = 13.5
	sigma_dR = 5
else:
	sys.exit("Select -v trainHiggs/trainZ")




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
def GenPhi(phi,event):
	return correct_phi(phi - event["Gen"+particle+"_Phi"])
	
def getDeltaR(event, genVar,bjet):
	return np.sqrt(
		(event["Reco_"+particle+"_B"+ bjet + "_Eta"] - event[genVar+"_Eta"])**2 + (PhiDiff(event["Reco_"+particle+"_B"+ bjet +"_Phi"],event[genVar+"_Phi"]))**2
		)

def correct_phi(phi):
	if(phi  <=  -np.pi):
		phi += 2*np.pi
	if(phi  >	np.pi):
		phi -= 2*np.pi
	return phi

###############################
# initialize list with columns to be written into dataframe
dataframe_columns = copy.deepcopy(variables)

#create df for event
eval_df = pd.DataFrame(columns = dataframe_columns)
df = pd.read_hdf(basedir+"/workdir/eval_dataframe/{}_eval/all/eval_allCombs_dnn.h5".format(process)) 
df = df.reset_index(drop=True)

BestIndex = [0]
predVal = [0]
null = [0]
perm = 0
N_permutation = -1
event_nr = 0
MinDelta = 1000

for iEvt in df.index:
	if iEvt%10000==0:
		print "Permutation",iEvt,"of",len(df.index)
	event = df.loc[iEvt]
	nJets = int(min(event["N_Jets"], 10))

	if perm == N_permutation:
		MinDelta = 1000
		event_nr += 1
		BestIndex.extend(null)
		predVal.extend(null)
		perm = 0
	perm += 1
	N_permutation = scipy.special.binom(nJets,2)
	
	chi2 = ((event["Reco_"+particle+"_M"]-particle_M)/sigma_M)**2#+(event["Reco_"+particle+"_B1_CSV"]-1)**2+(event["Reco_"+particle+"_B2_CSV"]-1)**2+((event["Reco_"+particle+"_Delta_R"]-particle_dR)/sigma_dR)**2
	if chi2 < MinDelta:
		MinDelta = chi2
		BestIndex[event_nr] = int(iEvt)
		predVal[event_nr] = MinDelta


n = 0
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
		deltaR1 = getDeltaR(event, "Gen"+particle+"_B1",str(j))
		if deltaR1 < minR1:
			minR1 = deltaR1
			p1 = j

		deltaR2 = getDeltaR(event, "Gen"+particle+"_B2",str(j))
		if deltaR2 < minR2:
			minR2 = deltaR2
			p2 = j

	if p1 == p2:
		#print "shit, B1 = B2"
		if (minR1 < minR2 and p1 == 1) or (minR1 > minR2 and p1 == 2):
			p1 = 1
			p2 = 2
		else:
			p1 = 2
			p2 = 1

	if minR1 <= 0.4 and minR2 <= 0.4: 
		valid_events+=1


	if n%1000 == 0:
		print "Event",n,"minR1", minR1,"valid events",valid_events


	n+=1



print valid_events,"valid events,", valid_events/float(len(BestIndex))*100,"%"


