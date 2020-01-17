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

def findbestbkg(df, additionalVariables):
	nevents = df.shape[0]
	print nevents,"Events in file."

	# required information
	assignedJets	= ["Higgs_B1","Higgs_B2"]
	recoParticles   = ["Higgs"]

	jetVars		 = ["Pt", "Eta", "Phi", "E", "CSV"]
	particleVars	= ["Pt", "Eta", "Phi", "M", "logM","E"]

	# initialize list with columns to be written into dataframe
	dataframe_columns = copy.deepcopy(additionalVariables)

	for j in assignedJets:
		for v in jetVars:
			dataframe_columns.append("Reco_"+j+"_"+v)
	for p in recoParticles:
		for v in particleVars:
			dataframe_columns.append("Reco_"+p+"_"+v)

	dataframe_columns.append("is_Higgs")

	# define some variables to be calculated later on
	dataframe_columns.append("Delta_Eta")
	dataframe_columns.append("Delta_Phi")
	dataframe_columns.append("Delta_R")

	dataframe_columns.append("Boosted1_Phi")
	dataframe_columns.append("Boosted1_Eta")
	dataframe_columns.append("Boosted1_Pt")
	dataframe_columns.append("Boosted1_E")
	dataframe_columns.append("Boosted1_M")
	dataframe_columns.append("Boosted1_logM")
	dataframe_columns.append("Boosted2_Eta")
	dataframe_columns.append("Boosted2_Pt")
	dataframe_columns.append("Boosted2_E")

	new_df = pd.DataFrame(columns = dataframe_columns)

	# event loop
	valid_events = 0
	n = -1
	for iEvt in df.index:
		# get number of jets (max 10)
		event = df.loc[iEvt]
		n += 1
		nJets = int(min(event["N_Jets"], 10))

		#quick printouts
		if n%500 == 0:
			print "Event",n

		# generate random wrong assignments
		for iw1 in range(nJets):
			for iw2 in range(iw1):
				if iw1 == iw2: continue
				ass = [iw1,iw2]

				# fill variables
				entry = {v: None for v in dataframe_columns}
				for v in additionalVariables:
					entry[v] = event[v]

				entry["is_Higgs"] = 0

				# fill assigned jets
				# DANGERZONE: make sure the order of entries in assignedJets list is the same as in the assignments list
				for it, j in enumerate(assignedJets):
					for v in jetVars:
						entry["Reco_"+j+"_"+v] = event["Jet_{}[{}]".format(v, ass[it])]

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

				boost1.Boost(-RecoHiggs4vec.BoostVector())
				boost2.Boost(-RecoHiggs4vec.BoostVector())

				for var in particleVars:
					entry["Boosted1_"+var] = Get(boost1, var)
					entry["Boosted2_"+var] = Get(boost2, var)


				# add special variables
				entry["Delta_Phi"] = PhiDiff(entry["Reco_Higgs_B1_Phi"], entry["Reco_Higgs_B2_Phi"])
				entry["Delta_Eta"] = EtaDiff(entry["Reco_Higgs_B1_Eta"], entry["Reco_Higgs_B2_Eta"])
				entry["Delta_R"] = np.sqrt((entry["Reco_Higgs_B1_Eta"]-entry["Reco_Higgs_B2_Eta"])**2 + entry["Delta_Phi"]**2)


				new_df = new_df.append(entry, ignore_index = True)
				
				del ttHSystem
				del entry

	print("added {}/{} events".format(valid_events, nevents))
	return new_df



def getDeltaR(event, genVar, jetIndex):
	return np.sqrt(
		(event["Jet_Eta[{}]".format(jetIndex)] - event[genVar+"_Eta"])**2 + (PhiDiff(event["Jet_Phi[{}]".format(jetIndex)],event[genVar+"_Phi"]))**2
		)

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
		else:
			exit("error in get reco var")

def Get(vec, variable):
	if variable == "Phi":
		return vec.Phi()
	elif variable == "Eta":
		return vec.Eta()
	elif variable == "Pt":
		return vec.Pt()
	elif variable == "M":
		return vec.M()
	elif variable == "logM":
		return log(vec.M())
	elif variable == "E":
		return vec.E()
	else:
		exit("error in Get reco var")

