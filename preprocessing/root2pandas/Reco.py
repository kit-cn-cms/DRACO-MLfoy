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

def findbestZorH(df, additionalVariables, ZorH):
	nevents = df.shape[0]

	# required information
	recoParticles   = [ZorH]
	assignedJets	= [recoParticles[0]+"_B1",recoParticles[0]+"_B2"]

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

	dataframe_columns.append("is_"+recoParticles[0])

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
	valid_events = 0
	n = 0
	for iEvt in df.index:
		# get number of jets (max 10)
		event = df.loc[iEvt]
		nJets = int(min(event["N_Jets"], 10))

		# skip events without bb from Lennarts TTZToQQ nTuples
		if ZorH == "Z":
			if event["GenEvt_I_TTZ"]!=1: continue

		# best deltaR
		bestDeltaR1 = 10000.
		bestDeltaR2 = 10000.
		foundComb   = False
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
#		for ih1 in range(nJets):
#			if i1 == bestIndices[0] or i1 == bestIndices[1]: continue
#			for i2 in range(i1):
#				if i2 == bestIndices[0] or i2 == bestIndices[1] or i1 == i2: continue
#				wrong = [i1,i2]
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
				entry["is_"+recoParticles[0]] = 1
			else:
				entry["is_"+recoParticles[0]] = 0

			# fill assigned jets
			# DANGERZONE: make sure the order of entries in assignedJets list is the same as in the assignments list
			for it, j in enumerate(assignedJets):
				for v in jetVars:
					entry["Reco_"+j+"_"+v] = event["Jet_{}[{}]".format(v, ass[it])]
				entry["Reco_"+j+"_logE"]  = log(event["Jet_E[{}]".format(ass[it])])
				entry["Reco_"+j+"_logPt"] = log(event["Jet_Pt[{}]".format(ass[it])])

			# calculate ttZorH system and write variables
			ttZorHSystem = reconstruct_ttZorH(entry, assignedJets, ZorH)

			for p in recoParticles:
				for v in particleVars:
					entry["Reco_"+p+"_"+v] = ttZorHSystem.get(p,v)


			#Boost in GenZorH Richtung
			boost1 	   	  = ROOT.TLorentzVector()
			boost2 		  = ROOT.TLorentzVector()
			Reco4vec = ROOT.TLorentzVector()

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

	print("added {}/{} events".format(valid_events, nevents))
	return new_df

##########################
def findbestTopBs(df, additionalVariables, ZorH):
	nevents = df.shape[0]

	# required information
	assignedJets	= [recoParticles[0]+"_B1",recoParticles[0]+"_B2"]
	recoParticles   = [ZorH]

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

	dataframe_columns.append("is_"+recoParticles[0])

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

		for i1 in range(nJets):
			if event["Jet_CSV[{}]".format(i1)] < csvWP: continue
			deltaR_B1 = getDeltaR(event, "GenTopHad_B", i1)
			if deltaR_B1 < bestDeltaR1:
				bestDeltaR1 = deltaR_B1
				bestIndices[0] = i1

		for i2 in range(nJets):
			if event["Jet_CSV[{}]".format(i2)] < csvWP: continue
			deltaR_B2 = getDeltaR(event, "GenTopLep_B", i2)
			if deltaR_B2 < bestDeltaR2:
				bestDeltaR2 = deltaR_B2
				bestIndices[1] = i2

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
			entry["is_"+recoParticles[0]] = 0

			# fill assigned jets
			# DANGERZONE: make sure the order of entries in assignedJets list is the same as in the assignments list
			for it, j in enumerate(assignedJets):
				for v in jetVars:
					entry["Reco_"+j+"_"+v] = event["Jet_{}[{}]".format(v, ass[it])]
				entry["Reco_"+j+"_logE"]  = log(event["Jet_E[{}]".format(ass[it])])
				entry["Reco_"+j+"_logPt"] = log(event["Jet_Pt[{}]".format(ass[it])])

			# calculate ttZorH system and write variables
			ttZorHSystem = reconstruct_ttZorH(entry, assignedJets, ZorH)
			for p in recoParticles:
				for v in particleVars:
					entry["Reco_"+p+"_"+v] = ttZorHSystem.get(p,v)

			#Boost in GenZorH Richtung
			boost1 	   	  = ROOT.TLorentzVector()
			boost2 		  = ROOT.TLorentzVector()
			Reco4vec = ROOT.TLorentzVector()

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


class reconstruct_ttZorH:
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
