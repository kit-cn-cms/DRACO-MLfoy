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

def findbestZ(df, additionalVariables):
	nevents = df.shape[0]

	# required information
	assignedJets	= ["Z_B1","Z_B2"]
	recoParticles   = ["Z"]

	jetVars			= ["Pt", "Eta", "Phi", "E", "CSV"]
	particleVars	= ["Pt", "Eta", "Phi", "M", "logM","E","logE","logPt"]
	boostVars		= ["Pt", "Eta","logPt"]

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

	dataframe_columns.append("is_Z")

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
	valid_events = 0
	n = 0
	for iEvt in df.index:
		# get number of jets (max 10)
		event = df.loc[iEvt]
		nJets = int(min(event["N_Jets"], 10))

		# best deltaR
		bestDeltaR1 = 10000.
		bestDeltaR2 = 10000.
		foundComb   = False
		bestIndices = np.full(2,int(1))

		for iZ1 in range(nJets):
			if event["Jet_CSV[{}]".format(iZ1)] < csvWP: continue
			deltaR_B1 = getDeltaR(event, "GenZ_B1", iZ1)
			if deltaR_B1 < bestDeltaR1:
				bestDeltaR1 = deltaR_B1
				bestIndices[0] = iZ1

		for iZ2 in range(nJets):
			if event["Jet_CSV[{}]".format(iZ2)] < csvWP: continue
			deltaR_B2 = getDeltaR(event, "GenZ_B2", iZ2)
			if deltaR_B2 < bestDeltaR2:
				bestDeltaR2 = deltaR_B2
				bestIndices[1] = iZ2

		if bestDeltaR1 < dRCut and bestDeltaR2 < dRCut and bestIndices[0] != bestIndices[1]:
			foundComb  = True

		#quick printouts
		if n%1000 == 0:
			print "Event",n,"Delta r1:",bestDeltaR1,"Delta r2:", bestDeltaR2
		n += 1

		# if no valid combination was found at the end of the loop, continue to next event
		if not foundComb: continue
		valid_events += 1

		assignments = []
		assignments.append(np.array(bestIndices))

#		generate random wrong assignments
		for iWrong in range(nWrongAssignments):
			foundNew = False
			while not foundNew:
				wrong = np.random.permutation(nJets)[:2]
				foundNew = True
				for p in bestIndices:
					if (p == wrong[0] or p == wrong[1]):
						foundNew = False
			assignments.append(wrong)

	#Generate all bkg
#		for ihiggs1 in range(nJets):
#			if ihiggs1 == bestIndices[0] or ihiggs1 == bestIndices[1]: continue
#			for ihiggs2 in range(iZ1):
#				if ihiggs2 == bestIndices[0] or ihiggs2 == bestIndices[1] or ihiggs1 == ihiggs2: continue
#				wrong = [ihiggs1,ihiggs2]
#				assignments.append(wrong)

	#Generate less bkg
#		for iWrong in range(nWrongAssignments):
#			wrong = np.random.permutation(nJets)[:2]
#			foundNew = True
#			for p in bestIndices:
#				if (p == wrong[0] or p == wrong[1]):
#					foundNew = False
#			if foundNew == True:
#				assignments.append(wrong)

	#Generate bkg with 1 right jet
		#foundNew = False
		#while foundNew == False:
		##for iWrong in range(nWrongAssignments):
			#wrong = np.random.permutation(nJets)[:2]
			#foundNew = True
			#if (bestIndices[0] != wrong[0] and bestIndices[0] != wrong[1] and bestIndices[1] != wrong[0] and bestIndices[1] != wrong[1]):
				#foundNew = False
			#elif (bestIndices[0] == wrong[0] and bestIndices[1] == wrong[1]) or (bestIndices[0] == wrong[1] and bestIndices[1] == wrong[0]):
				#foundNew = False
			#if foundNew == True:
				#assignments.append(wrong)

		# fill assignments
		for idx, ass in enumerate(assignments):
			# fill variables
			entry = {v: None for v in dataframe_columns}
			for v in additionalVariables:
				entry[v] = event[v]
			if idx == 0:
				entry["is_Z"] = 1
			else:
				entry["is_Z"] = 0

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


			#Boost in GenZ Richtung
			boost1 	   	  = ROOT.TLorentzVector()
			boost2 		  = ROOT.TLorentzVector()
			RecoZ4vec = ROOT.TLorentzVector()

			RecoZ4vec.SetPtEtaPhiE(float(entry["Reco_Z_Pt"]),float(entry["Reco_Z_Eta"]),float(entry["Reco_Z_Phi"]),float(entry["Reco_Z_E"]))

			boost1.SetPtEtaPhiE(float(event["Jet_Pt[{}]".format(ass[0])]),float(event["Jet_Eta[{}]".format(ass[0])]),float(event["Jet_Phi[{}]".format(ass[0])]),float(event["Jet_E[{}]".format(ass[0])]))
			boost2.SetPtEtaPhiE(float(event["Jet_Pt[{}]".format(ass[1])]),float(event["Jet_Eta[{}]".format(ass[1])]),float(event["Jet_Phi[{}]".format(ass[1])]),float(event["Jet_E[{}]".format(ass[1])]))

			#angle before boost
			entry["Angle"] = boost1.Angle(boost2.Vect())

			boost1.Boost(-RecoZ4vec.BoostVector())
			boost2.Boost(-RecoZ4vec.BoostVector())

			for var in boostVars:
				entry["Boosted1_"+var] = Get(boost1,var)
				entry["Boosted2_"+var] = Get(boost2,var)

			# add special variables
			entry["Delta_Phi"] = PhiDiff(entry["Reco_Z_B1_Phi"], entry["Reco_Z_B2_Phi"])
			entry["Delta_Eta"] = abs(entry["Reco_Z_B1_Eta"]- entry["Reco_Z_B2_Eta"])
			entry["Delta_R"] = np.sqrt(entry["Delta_Eta"]**2 + entry["Delta_Phi"]**2)
			entry["Delta_R3D"] = np.sqrt((entry["Delta_Eta"]/5)**2 + (entry["Delta_Phi"]/(2*np.pi))**2 + (abs(entry["Reco_Z_B1_Pt"]- entry["Reco_Z_B2_Pt"])/1000)**2)

			new_df = new_df.append(entry, ignore_index = True)
			
			del ttHSystem
			del entry

	print("added {}/{} events".format(valid_events, nevents))
	return new_df

##########################
def findbestTopBs(df, additionalVariables):
	nevents = df.shape[0]

	# required information
	assignedJets	= ["Z_B1","Z_B2"]
	recoParticles   = ["Z"]

	jetVars		 	= ["Pt", "Eta", "Phi", "E", "CSV"]
	particleVars	= ["Pt", "Eta", "Phi", "M", "logM","E","logE","logPt"]
	boostVars		= ["Pt","Eta","logPt"]

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

	dataframe_columns.append("is_Z")

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
	valid_events = 0
	n = 0
	for iEvt in df.index:
		# get number of jets (max 10)
		event = df.loc[iEvt]
		nJets = int(min(event["N_Jets"], 10))

		# best deltaR
		bestDeltaR1 = 10000.
		bestDeltaR2 = 10000.
		foundComb   = False
		bestIndices = np.full(2,int(1))

		for iZ1 in range(nJets):
			if event["Jet_CSV[{}]".format(iZ1)] < csvWP: continue
			deltaR_B1 = getDeltaR(event, "GenTopHad_B", iZ1)
			if deltaR_B1 < bestDeltaR1:
				bestDeltaR1 = deltaR_B1
				bestIndices[0] = iZ1

		for iZ2 in range(nJets):
			if event["Jet_CSV[{}]".format(iZ2)] < csvWP: continue
			deltaR_B2 = getDeltaR(event, "GenTopLep_B", iZ2)
			if deltaR_B2 < bestDeltaR2:
				bestDeltaR2 = deltaR_B2
				bestIndices[1] = iZ2

		if bestDeltaR1 < dRCut and bestDeltaR2 < dRCut and bestIndices[0] != bestIndices[1]:
			foundComb  = True

		#quick printouts
		if n%1000 == 0:
			print "Event",n,"Delta r1:",bestDeltaR1,"Delta r2:", bestDeltaR2
		n += 1

		# if no valid combination was found at the end of the loop, continue to next event
		if not foundComb: continue
		valid_events += 1

		assignments = []
		assignments.append(np.array(bestIndices))

		# fill assignments
		for idx, ass in enumerate(assignments):
			# fill variables
			entry = {v: None for v in dataframe_columns}
			for v in additionalVariables:
				entry[v] = event[v]
			entry["is_Z"] = 0

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

			#Boost in GenZ Richtung
			boost1 	   	  = ROOT.TLorentzVector()
			boost2 		  = ROOT.TLorentzVector()
			RecoZ4vec = ROOT.TLorentzVector()

			RecoZ4vec.SetPtEtaPhiE(float(entry["Reco_Z_Pt"]),float(entry["Reco_Z_Eta"]),float(entry["Reco_Z_Phi"]),float(entry["Reco_Z_E"]))

			boost1.SetPtEtaPhiE(float(event["Jet_Pt[{}]".format(ass[0])]),float(event["Jet_Eta[{}]".format(ass[0])]),float(event["Jet_Phi[{}]".format(ass[0])]),float(event["Jet_E[{}]".format(ass[0])]))
			boost2.SetPtEtaPhiE(float(event["Jet_Pt[{}]".format(ass[1])]),float(event["Jet_Eta[{}]".format(ass[1])]),float(event["Jet_Phi[{}]".format(ass[1])]),float(event["Jet_E[{}]".format(ass[1])]))

			#angle before boost
			entry["Angle"] = boost1.Angle(boost2.Vect())

			boost1.Boost(-RecoZ4vec.BoostVector())
			boost2.Boost(-RecoZ4vec.BoostVector())

			for var in boostVars:
				entry["Boosted1_"+var] = Get(boost1,var)
				entry["Boosted2_"+var] = Get(boost2,var)

			# add special variables
			entry["Delta_Phi"] = PhiDiff(entry["Reco_Z_B1_Phi"], entry["Reco_Z_B2_Phi"])
			entry["Delta_Eta"] = abs(entry["Reco_Z_B1_Eta"]- entry["Reco_Z_B2_Eta"])
			entry["Delta_R"] = np.sqrt(entry["Delta_Eta"]**2 + entry["Delta_Phi"]**2)
			entry["Delta_R3D"] = np.sqrt((entry["Delta_Eta"]/5)**2 + (entry["Delta_Phi"]/(2*np.pi))**2 + (abs(entry["Reco_Z_B1_Pt"]- entry["Reco_Z_B2_Pt"])/1000)**2)

			new_df = new_df.append(entry, ignore_index = True)
			
			del ttHSystem
			del entry

	print("added {}/{} events".format(valid_events, nevents))
	return new_df

################################################

def getDeltaR(event, genVar, jetIndex):
	return np.sqrt(
		(event["Jet_Eta[{}]".format(jetIndex)] - event[genVar+"_Eta"])**2 + (PhiDiff(event["Jet_Phi[{}]".format(jetIndex)],event[genVar+"_Phi"]))**2
		)

def getDeltaR3(event, genVar, jetIndex):
	return np.sqrt(
		(event["Jet_Eta[{}]".format(jetIndex)]/5 - event[genVar+"_Eta"]/5)**2 + (PhiDiff(event["Jet_Phi[{}]".format(jetIndex)],event[genVar+"_Phi"])/(2*np.pi))**2 + (event["Jet_Pt[{}]".format(jetIndex)]/1000 - event[genVar+"_Pt"]/1000)**2
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
			
		vectors["Z"] = vectors["Z_B1"] + vectors["Z_B2"]

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
