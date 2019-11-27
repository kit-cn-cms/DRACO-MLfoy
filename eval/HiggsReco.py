import sys
import os
from math import sin, cos, log
import numpy as np
import pandas as pd
import ROOT
import copy

# temporary hard coded bullshit
csvWP = 0.277
dRCut = 0.4
nWrongAssignments = 1

################################################
def eval(df, additionalVariables):
	nevents = df.shape[0]

	# required information
	assignedJets	= ["Higgs_B1","Higgs_B2"]
	recoParticles   = ["Higgs"]

	jetVars			= ["Pt", "Eta", "Phi", "E", "CSV"]
	particleVars	= ["Pt", "Eta", "Phi", "M", "logM","E","logE","logPt"]
	boostVars		= ["Pt", "Eta", "logPt"]

	# initialize list with columns to be written into dataframe
	dataframe_columns = copy.deepcopy(additionalVariables)

	for j in assignedJets:
		for v in jetVars:
			dataframe_columns.append("Reco_"+j+"_"+v)
		dataframe_columns.append("Reco_"+j+"_logE")
		dataframe_columns.append("Reco_"+j+"_logPt")

	for p in recoParticles:
		for v in particleVars:
			dataframe_columns.append("Reco_"+p+"_"+v)

	dataframe_columns.append("Event_Nr")

	# define some variables to be calculated later on
	dataframe_columns.append("Delta_Eta")
	dataframe_columns.append("Delta_Phi")
	dataframe_columns.append("Delta_R")
	dataframe_columns.append("Delta_R3D")
	dataframe_columns.append("Angle")

	for v in boostVars:
		dataframe_columns.append("Boosted1_"+v)
		dataframe_columns.append("Boosted2_"+v)

	new_df = pd.DataFrame(columns = dataframe_columns)

	# event loop
	n = 0
	for iEvt in df.index:
		# get number of jets (max 10)
		if n==1000:
			break
		event = df.loc[iEvt]
		nJets = int(min(event["N_Jets"], 10))

		#quick printouts
		if n%1000 == 0:
			print "Event",n
		n += 1
		assignments = []

		# generate random wrong assignments
		for j in range(nJets):
			for k in range(j):
				if j == k: continue
				comb = [j,k]
				assignments.append(comb)

		# fill assignments
		for idx, ass in enumerate(assignments):
			# fill variables
			entry = {v: None for v in dataframe_columns}
			entry["Event_Nr"] = n
			for v in additionalVariables:
				entry[v] = event[v]

			# fill assigned jets
			# DANGERZONE: make sure the order of entries in assignedJets list is the same as in the assignments list
			for it, j in enumerate(assignedJets):
				for v in jetVars:
					entry["Reco_"+j+"_"+v] = event["Jet_{}[{}]".format(v, ass[it])]

				entry["Reco_"+j+"_logE"]  = log(event["Jet_E[{}]".format(ass[it])])
				entry["Reco_"+j+"_logPt"] = log(event["Jet_Pt[{}]".format(ass[it])])

			# calculate ttH system and write variables
			ttHSystem = reconstruct_ttH(entry, assignedJets)
			for p in recoParticles:
				for v in particleVars:
					entry["Reco_"+p+"_"+v] = ttHSystem.get(p,v)


			#Boost in GenHiggs Richtung
			boost1 	   	  = ROOT.TLorentzVector()
			boost2 		  = ROOT.TLorentzVector()
			RecoHiggs4vec = ROOT.TLorentzVector()

			RecoHiggs4vec.SetPtEtaPhiE(float(entry["Reco_Higgs_Pt"]),float(entry["Reco_Higgs_Eta"]),float(entry["Reco_Higgs_Phi"]),float(entry["Reco_Higgs_E"]))

			boost1.SetPtEtaPhiE(float(event["Jet_Pt[{}]".format(ass[0])]),float(event["Jet_Eta[{}]".format(ass[0])]),float(event["Jet_Phi[{}]".format(ass[0])]),float(event["Jet_E[{}]".format(ass[0])]))
			boost2.SetPtEtaPhiE(float(event["Jet_Pt[{}]".format(ass[1])]),float(event["Jet_Eta[{}]".format(ass[1])]),float(event["Jet_Phi[{}]".format(ass[1])]),float(event["Jet_E[{}]".format(ass[1])]))

			#angle before boost
			entry["Angle"] = boost1.Angle(boost2.Vect())

			boost1.Boost(-RecoHiggs4vec.BoostVector())
			boost2.Boost(-RecoHiggs4vec.BoostVector())

			for var in boostVars:
				entry["Boosted1_"+var] = Get(boost1,var)
				entry["Boosted2_"+var] = Get(boost2,var)

			# add special variables
			entry["Delta_Phi"] = PhiDiff(entry["Reco_Higgs_B1_Phi"], entry["Reco_Higgs_B2_Phi"])
			entry["Delta_Eta"] = abs(entry["Reco_Higgs_B1_Eta"]- entry["Reco_Higgs_B2_Eta"])
			entry["Delta_R"] = np.sqrt(entry["Delta_Eta"]**2 + entry["Delta_Phi"]**2)
			entry["Delta_R3D"] = np.sqrt((entry["Delta_Eta"]/5)**2 + (entry["Delta_Phi"]/(2*np.pi))**2 + (abs(entry["Reco_Higgs_B1_Pt"]- entry["Reco_Higgs_B2_Pt"])/1000)**2)

			new_df = new_df.append(entry, ignore_index = True)
			
			del ttHSystem
			del entry

	return new_df

##########################

#function to correct a difference of two angulars phi which is in [-2pi,2pi] to the correct interval [0,2pi]
def correctPhi(phi):
	while(phi  <=  -np.pi):
		phi += 2*np.pi
	while(phi  >	np.pi):
		phi -= 2*np.pi
	return phi

def PhiDiff(phi1,phi2):
	if abs(phi1-phi2) > np.pi:
		return 2*np.pi - abs(phi1-phi2)
	else:
		return abs(phi1-phi2)

def EtaDiff(eta1,eta2):
	return abs(eta1 + eta2)


class reconstruct_ttH:
	def __init__(self, entry, jets):
		vectors = {}

		for j in jets:
			vectors[j] = ROOT.TLorentzVector()
			vectors[j].SetPtEtaPhiE( entry["Reco_"+j+"_Pt"], entry["Reco_"+j+"_Eta"], entry["Reco_"+j+"_Phi"], entry["Reco_"+j+"_E"] )
			
		vectors["Higgs"] = vectors["Higgs_B1"] + vectors["Higgs_B2"]

		self.vectors = vectors

	def get(self, particle, variable):
		if variable == "Phi":
			return self.vectors[particle].Phi()
		elif variable == "Eta":
			return self.vectors[particle].Eta()
		elif variable == "Pt":
			return self.vectors[particle].Pt()
		elif variable == "M":
			return self.vectors[particle].M()
		elif variable == "logM":
			return log(self.vectors[particle].M())
		elif variable == "E":
			return self.vectors[particle].E()
		elif variable == "logE":
			return log(self.vectors[particle].E())
		elif variable == "logPt":
			return log(self.vectors[particle].Pt())
		else:
			exit("error in get reco var")
def Get(vec, variable):
	if variable == "Eta":
		return vec.Eta()
	elif variable == "Pt":
		return vec.Pt()
	elif variable == "logPt":
		return log(vec.Pt())
	else:
		exit("error in get reco var")
