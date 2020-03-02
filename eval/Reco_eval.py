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
def Eval(df, additionalVariables, particle):
	nevents = df.shape[0]

	# required information
	recoParticles   = [particle]
	assignedJets	= [recoParticles[0]+"_B1",recoParticles[0]+"_B2"]


	jetVars			= ["Pt", "Eta", "Phi", "E", "CSV"]
	particleVars		= ["Pt", "Eta", "Phi", "M", "logM","E","logE","logPt"]
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
	dataframe_columns.append("Reco_"+recoParticles[0]+"_Delta_Eta")
	dataframe_columns.append("Reco_"+recoParticles[0]+"_Delta_Phi")
	dataframe_columns.append("Reco_"+recoParticles[0]+"_Delta_R")
	dataframe_columns.append("Reco_"+recoParticles[0]+"_Delta_R3D")
	dataframe_columns.append("Reco_"+recoParticles[0]+"_Angle")


	for v in boostVars:
		dataframe_columns.append("Reco_"+recoParticles[0]+"_Boosted1_"+v)
		dataframe_columns.append("Reco_"+recoParticles[0]+"_Boosted2_"+v)

	new_df = pd.DataFrame(columns = dataframe_columns)

	# event loop
	n = 0
	valids = 0
	for iEvt in df.index:

		if n>10000:

			return new_df
		# get number of jets (max 10)
		event = df.loc[iEvt]
		nJets = int(min(event["N_Jets"], 10))
		n += 1

		if 1 == 0:# Richtige Zuordnung ueberhaupt moeglich?
			bestDeltaR1 = 10000.
			bestDeltaR2 = 10000.
			bestIndices = np.full(2,int(1))

			for i1 in range(nJets):
				if event["Jet_CSV[{}]".format(i1)] < csvWP: continue
				deltaR_B1 = getDeltaR(event, "Gen"+recoParticles[0]+"_B1", i1)
				if deltaR_B1 < bestDeltaR1:
					bestDeltaR1 = deltaR_B1
					bestIndices[0] = i1

			for i2 in range(nJets):
				if event["Jet_CSV[{}]".format(i2)] < csvWP: continue
				deltaR_B2 = getDeltaR(event, "Gen"+recoParticles[0]+"_B2", i2)
				if deltaR_B2 < bestDeltaR2:
					bestDeltaR2 = deltaR_B2
					bestIndices[1] = i2
			
			if bestDeltaR1 > dRCut or bestDeltaR2 > dRCut or bestIndices[0] == bestIndices[1]:
				continue
			valids+=1

		#quick printouts
		if n%1000 == 0:
			print "Event",n

		assignments = []

		# generate permutations
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
			for it, j in enumerate(assignedJets):
				for v in jetVars:
					entry["Reco_"+j+"_"+v] = event["Jet_{}[{}]".format(v, ass[it])]

				entry["Reco_"+j+"_logE"]  = log(event["Jet_E[{}]".format(ass[it])])
				entry["Reco_"+j+"_logPt"] = log(event["Jet_Pt[{}]".format(ass[it])])

		# calculate ttZorH system and write variables
			ttZorHSystem = reconstruct(entry, assignedJets, particle)
			for p in recoParticles:
				for v in particleVars:
					entry["Reco_"+p+"_"+v] = ttZorHSystem.get(p,v)




			#Boost in RecoZorH Richtung
			boost1 	   	  = ROOT.TLorentzVector()
			boost2 		  = ROOT.TLorentzVector()
			Reco4vec	  = ROOT.TLorentzVector()

			Reco4vec.SetPtEtaPhiE(float(entry["Reco_"+recoParticles[0]+"_Pt"]),float(entry["Reco_"+recoParticles[0]+"_Eta"]),float(entry["Reco_"+recoParticles[0]+"_Phi"]),float(entry["Reco_"+recoParticles[0]+"_E"]))

			boost1.SetPtEtaPhiE(float(event["Jet_Pt[{}]".format(ass[0])]),float(event["Jet_Eta[{}]".format(ass[0])]),float(event["Jet_Phi[{}]".format(ass[0])]),float(event["Jet_E[{}]".format(ass[0])]))
			boost2.SetPtEtaPhiE(float(event["Jet_Pt[{}]".format(ass[1])]),float(event["Jet_Eta[{}]".format(ass[1])]),float(event["Jet_Phi[{}]".format(ass[1])]),float(event["Jet_E[{}]".format(ass[1])]))
			#angle before boost
			entry["Reco_"+recoParticles[0]+"_Angle"] = boost1.Angle(boost2.Vect())

			boost1.Boost(-Reco4vec.BoostVector())
			boost2.Boost(-Reco4vec.BoostVector())

			for var in boostVars:
				entry["Reco_"+recoParticles[0]+"_Boosted1_"+var] = Get(boost1,var)
				entry["Reco_"+recoParticles[0]+"_Boosted2_"+var] = Get(boost2,var)

			# add special variables
			entry["Reco_"+recoParticles[0]+"_Delta_Phi"] = PhiDiff(entry["Reco_"+recoParticles[0]+"_B1_Phi"], entry["Reco_"+recoParticles[0]+"_B2_Phi"])
			entry["Reco_"+recoParticles[0]+"_Delta_Eta"] = abs(entry["Reco_"+recoParticles[0]+"_B1_Eta"]- entry["Reco_"+recoParticles[0]+"_B2_Eta"])
			entry["Reco_"+recoParticles[0]+"_Delta_R"] = np.sqrt(entry["Reco_"+recoParticles[0]+"_Delta_Eta"]**2 + entry["Reco_"+recoParticles[0]+"_Delta_Phi"]**2)
			entry["Reco_"+recoParticles[0]+"_Delta_R3D"] = np.sqrt((entry["Reco_"+recoParticles[0]+"_Delta_Eta"]/5)**2 + (entry["Reco_"+recoParticles[0]+"_Delta_Phi"]/(2*np.pi))**2 + (abs(entry["Reco_"+recoParticles[0]+"_B1_Pt"]- entry["Reco_"+recoParticles[0]+"_B2_Pt"])/1000)**2)

			new_df = new_df.append(entry, ignore_index = True)
			
			del ttZorHSystem
			del entry

	return new_df



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


def getDeltaR(event, genVar, jetIndex):
	return np.sqrt(
		(event["Jet_Eta[{}]".format(jetIndex)] - event[genVar+"_Eta"])**2 + (PhiDiff(event["Jet_Phi[{}]".format(jetIndex)],event[genVar+"_Phi"]))**2
		)


class reconstruct:
	def __init__(self, entry, jets, recoParticles):
		vectors = {}

		for j in jets:
			vectors[j] = ROOT.TLorentzVector()
			vectors[j].SetPtEtaPhiE( entry["Reco_"+j+"_Pt"], entry["Reco_"+j+"_Eta"], entry["Reco_"+j+"_Phi"], entry["Reco_"+j+"_E"] )
			
		vectors[recoParticles] = vectors[recoParticles+"_B1"] + vectors[recoParticles+"_B2"]

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
