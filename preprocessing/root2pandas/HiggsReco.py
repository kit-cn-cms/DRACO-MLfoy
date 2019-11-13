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

def findbestHiggs(df, additionalVariables):
	nevents = df.shape[0]


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
#	dataframe_columns.append("Boosted_Delta_R")
	dataframe_columns.append("Boosted1_Pt")
	dataframe_columns.append("Boosted1_E")
	dataframe_columns.append("Boosted1_M")
	dataframe_columns.append("Boosted1_logM")
	dataframe_columns.append("Boosted2_Eta")
	dataframe_columns.append("Boosted2_Pt")
	dataframe_columns.append("Boosted2_E")

#	dataframe_columns.append("Boosted_Delta_Phi_Gen")
#	dataframe_columns.append("Boosted_Delta_Eta_Gen")
#	dataframe_columns.append("Boosted_Delta_R_Gen")


	new_df = pd.DataFrame(columns = dataframe_columns)

	# event loop
	valid_events = 0
	n = -1
	for iEvt in df.index:
		# get number of jets (max 10)
		event = df.loc[iEvt]
		n += 1
		nJets = int(min(event["N_Jets"], 10))

		# best deltaR
		bestDeltaR1 = 10000.
		bestDeltaR2 = 10000.
		foundComb   = False
		bestIndices = np.full(2,int(1))

		for iHiggs1 in range(nJets):
			deltaR_B1 = getDeltaR(event, "GenHiggs_B1", iHiggs1)
			if deltaR_B1 < bestDeltaR1:
				bestDeltaR1 = deltaR_B1
				bestIndices[0] = iHiggs1

		for iHiggs2 in range(nJets):
			deltaR_B2 = getDeltaR(event, "GenHiggs_B2", iHiggs2)
			if deltaR_B2 < bestDeltaR2:
				bestDeltaR2 = deltaR_B2
				bestIndices[1] = iHiggs2

		if bestDeltaR1 < dRCut and bestDeltaR2 < dRCut and bestIndices[0] != bestIndices[1]:
			foundComb  = True

		#quick printouts
		if n%500 == 0:
			print "Event",n,"Delta r1:",bestDeltaR1,"Delta r2:", bestDeltaR2

		# if no valid combination was found at the end of the loop, continue to next event
		if not foundComb: continue
		valid_events += 1

		assignments = []
		assignments.append(np.array(bestIndices))

		# generate random wrong assignments
		for iWrong in range(nWrongAssignments):
			foundNew = False
			while not foundNew:
				wrong = np.random.permutation(nJets)[:2]
				foundNew = True
				for p in bestIndices:
					if (p == wrong[0] or p == wrong[1]):
						foundNew = False
			assignments.append(wrong)


		# fill assignments
		for idx, ass in enumerate(assignments):
			# fill variables
			entry = {v: None for v in dataframe_columns}
			for v in additionalVariables:
				entry[v] = event[v]
			if idx == 0:
				entry["is_Higgs"] = 1
			else:
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
			GenHiggs4vec  = ROOT.TLorentzVector()
			RecoHiggs4vec = ROOT.TLorentzVector()
#			GenHiggs14vec = ROOT.TLorentzVector()
#			GenHiggs24vec = ROOT.TLorentzVector()
#			GenHiggs4vec.SetPtEtaPhiE(float(event["GenHiggs_Pt"]),float(event["GenHiggs_Eta"]),float(event["GenHiggs_Phi"]),float(event["GenHiggs_E"]))
			RecoHiggs4vec.SetPtEtaPhiE(float(entry["Reco_Higgs_Pt"]),float(entry["Reco_Higgs_Eta"]),float(entry["Reco_Higgs_Phi"]),float(entry["Reco_Higgs_E"]))
#			GenHiggs14vec.SetPtEtaPhiE(float(event["GenHiggs_B1_Pt"]),float(event["GenHiggs_B1_Eta"]),float(event["GenHiggs_B1_Phi"]),float(event["GenHiggs_B1_E"]))
#			GenHiggs24vec.SetPtEtaPhiE(float(event["GenHiggs_B2_Pt"]),float(event["GenHiggs_B2_Eta"]),float(event["GenHiggs_B2_Phi"]),float(event["GenHiggs_B2_E"]))
			boost1.SetPtEtaPhiE(float(event["Jet_Pt[{}]".format(ass[0])]),float(event["Jet_Eta[{}]".format(ass[0])]),float(event["Jet_Phi[{}]".format(ass[0])]),float(event["Jet_E[{}]".format(ass[0])]))
			boost2.SetPtEtaPhiE(float(event["Jet_Pt[{}]".format(ass[1])]),float(event["Jet_Eta[{}]".format(ass[1])]),float(event["Jet_Phi[{}]".format(ass[1])]),float(event["Jet_E[{}]".format(ass[1])]))
			
			boost1.Boost(-boost2.BoostVector())
			boost2.Boost(-RecoHiggs4vec.BoostVector())

#			GenHiggs14vec.Boost(-GenHiggs4vec.BoostVector())
#			GenHiggs24vec.Boost(-GenHiggs4vec.BoostVector())

#			entry["Boosted_Delta_Phi_Gen"]=PhiDiff(GenHiggs14vec.Phi(),GenHiggs24vec.Phi())
#			entry["Boosted_Delta_Eta_Gen"]=EtaDiff(GenHiggs14vec.Eta(),GenHiggs24vec.Eta())
#			entry["Boosted_Delta_R_Gen"]=np.sqrt(entry["Boosted_Delta_Eta_Gen"]**2 + entry["Boosted_Delta_Phi_Gen"]**2)




			entry["Boosted1_Phi"]=boost1.Phi()#PhiDiff(boost1.Phi(),boost2.Phi())
			entry["Boosted1_Eta"]=boost1.Eta()#EtaDiff(boost1.Eta(),boost2.Eta())
#			entry["Boosted_Delta_R"]=np.sqrt(entry["Boosted_Delta_Eta"]**2 + entry["Boosted_Delta_Phi"]**2)	
			entry["Boosted1_Pt"]=boost1.Pt()#abs(boost1.Pt()-boost2.Pt())
			entry["Boosted1_E"]=boost1.E()#abs(boost1.E()-boost2.E())
			entry["Boosted1_M"]=boost1.M()#abs(boost1.M()-boost2.M())
			entry["Boosted1_logM"]=log(boost1.M())#abs(log(boost1.M()) - log(boost2.M()))
			entry["Boosted2_Eta"]=boost2.Eta()#EtaDiff(boost1.Eta(),boost2.Eta())
			entry["Boosted2_Pt"]=boost2.Pt()#abs(boost1.Pt()-boost2.Pt())
			entry["Boosted2_E"]=boost2.E()#abs(boost1.E()-boost2.E())



			# add special variables
			entry["Delta_Phi"] = PhiDiff(entry["Reco_Higgs_B1_Phi"], entry["Reco_Higgs_B2_Phi"])
			entry["Delta_Eta"] = EtaDiff(entry["Reco_Higgs_B1_Eta"], entry["Reco_Higgs_B2_Eta"])
			entry["Delta_R"] = np.sqrt((entry["Reco_Higgs_B1_Eta"]-entry["Reco_Higgs_B2_Eta"])**2 + entry["Delta_Phi"]**2)


			new_df = new_df.append(entry, ignore_index = True)
			
			del ttHSystem
			del entry

	print("added {}/{} events".format(valid_events, nevents))
	return new_df


def findbestHiggsdPhi(df, additionalVariables):
	nevents = df.shape[0]
	print nevents, "Events found."


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


#	for j in range(10):
#		dataframe_columns.append("BoostedJet1"+str(j))
#		dataframe_columns.append("BoostedJet2"+str(j))


	# define some variables to be calculated later on
	dataframe_columns.append("Delta_Eta")
	dataframe_columns.append("Delta_Phi")
	dataframe_columns.append("Delta_R")
	dataframe_columns.append("Delta_R1")
	dataframe_columns.append("Delta_R2")

	dataframe_columns.append("Boosted1_Phi")
	dataframe_columns.append("Boosted1_Eta")
#	dataframe_columns.append("Boosted_Delta_R")
	dataframe_columns.append("Boosted1_Pt")
	dataframe_columns.append("Boosted1_E")
	dataframe_columns.append("Boosted1_M")
	dataframe_columns.append("Boosted1_logM")
	dataframe_columns.append("Boosted2_Eta")
	dataframe_columns.append("Boosted2_Pt")
	dataframe_columns.append("Boosted2_E")
	dataframe_columns.append("Boosted_Delta_Phi")



#	dataframe_columns.append("Boosted_Delta_Phi_Gen")
#	dataframe_columns.append("Boosted_Delta_Eta_Gen")
#	dataframe_columns.append("Boosted_Delta_R_Gen")


	new_df = pd.DataFrame(columns = dataframe_columns)

	# event loop
	valid_events = 0
	n = -1
	for iEvt in df.index:
		# get number of jets (max 10)
		event = df.loc[iEvt]
		n += 1
		nJets = int(min(event["N_Jets"], 10))

		# best deltaR
		bestdPhi = 10000.
		foundComb   = False
		bestIndices = np.full(2,int(1))

		GenHiggs4vec  = ROOT.TLorentzVector()
		GenHiggs4vec.SetPtEtaPhiE(float(event["GenHiggs_Pt"]),float(event["GenHiggs_Eta"]),float(event["GenHiggs_Phi"]),float(event["GenHiggs_E"]))

		for iHiggs1 in range(nJets):
			B1 = ROOT.TLorentzVector()
			B1.SetPtEtaPhiE(float(event["Jet_Pt[{}]".format(iHiggs1)]),float(event["Jet_Eta[{}]".format(iHiggs1)]),float(event["Jet_Phi[{}]".format(iHiggs1)]),float(event["Jet_E[{}]".format(iHiggs1)]))
			B1.Boost(-GenHiggs4vec.BoostVector())
			for iHiggs2 in range(nJets):
				if iHiggs1 == iHiggs2: continue
				B2 = ROOT.TLorentzVector()
				B2.SetPtEtaPhiE(float(event["Jet_Pt[{}]".format(iHiggs2)]),float(event["Jet_Eta[{}]".format(iHiggs2)]),float(event["Jet_Phi[{}]".format(iHiggs2)]),float(event["Jet_E[{}]".format(iHiggs2)]))
				B2.Boost(-GenHiggs4vec.BoostVector())
				dPhi = abs(PhiDiff(B1.Phi(),B2.Phi())-np.pi)
				if dPhi < bestdPhi:
					bestdPhi = dPhi
					bestIndices[0] = iHiggs1
					bestIndices[1] = iHiggs2

		#quick printouts
		if n%500 == 0:
			print "Event",n,"Delta Phi:",bestdPhi

		# if no valid combination was found at the end of the loop, continue to next event
		if bestdPhi > 0.2: continue
		#if event["Jet_CSV[{}]".format(bestIndices[0])] < csvWP or event["Jet_CSV[{}]".format(bestIndices[1])] < csvWP:
		#	continue

		valid_events += 1

		assignments = []
		assignments.append(np.array(bestIndices))

		# generate random wrong assignments
		for iWrong in range(nWrongAssignments):
			foundNew = False
			while not foundNew:
				wrong = np.random.permutation(nJets)[:2]
				foundNew = True
				for p in bestIndices:
					if (p == wrong[0] or p == wrong[1]):
						foundNew = False
			assignments.append(wrong)


		# fill assignments
		for idx, ass in enumerate(assignments):
			# fill variables
			entry = {v: None for v in dataframe_columns}
			for v in additionalVariables:
				entry[v] = event[v]
			if idx == 0:
				entry["is_Higgs"] = 1
			else:
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


			entry["Delta_R1"] = getDeltaR(event, "GenHiggs_B1", bestIndices[0])
			entry["Delta_R2"] = getDeltaR(event, "GenHiggs_B2", bestIndices[1])

			BoostedJet1 = ROOT.TLorentzVector()
			BoostedJet1.SetPtEtaPhiE(float(event["Jet_Pt[{}]".format(ass[0])]),float(event["Jet_Eta[{}]".format(ass[0])]),float(event["Jet_Phi[{}]".format(ass[0])]),float(event["Jet_E[{}]".format(ass[0])]))
			BoostedJet1.Boost(-GenHiggs4vec.BoostVector())
			BoostedJet2 = ROOT.TLorentzVector()
			BoostedJet2.SetPtEtaPhiE(float(event["Jet_Pt[{}]".format(ass[1])]),float(event["Jet_Eta[{}]".format(ass[1])]),float(event["Jet_Phi[{}]".format(ass[1])]),float(event["Jet_E[{}]".format(ass[1])]))
			BoostedJet2.Boost(-GenHiggs4vec.BoostVector())

			entry["Boosted_best_Delta_Phi"]= bestdPhi
#			entry["Boosted_Delta_Eta_Gen"]=EtaDiff(GenHiggs14vec.Eta(),GenHiggs24vec.Eta())
#			entry["Boosted_Delta_R_Gen"]=np.sqrt(entry["Boosted_Delta_Eta_Gen"]**2 + entry["Boosted_Delta_Phi_Gen"]**2)



			entry["Boosted1_Phi"]=BoostedJet1.Phi()
			entry["Boosted1_Eta"]=BoostedJet1.Eta()
#			entry["Boosted_Delta_R"]=np.sqrt(entry["Boosted_Delta_Eta"]**2 + entry["Boosted_Delta_Phi"]**2)	
			entry["Boosted1_Pt"]=BoostedJet1.Pt()#abs(boost1.Pt()-boost2.Pt())
			entry["Boosted1_E"]=BoostedJet1.E()#abs(boost1.E()-boost2.E())
			entry["Boosted1_M"]=BoostedJet1.M()#abs(boost1.M()-boost2.M())
			entry["Boosted1_logM"]=log(BoostedJet1.M())#abs(log(boost1.M()) - log(boost2.M()))
			entry["Boosted2_Eta"]=BoostedJet2.Eta()#EtaDiff(boost1.Eta(),boost2.Eta())
			entry["Boosted2_Pt"]=BoostedJet2.Pt()#abs(boost1.Pt()-boost2.Pt())
			entry["Boosted2_E"]=BoostedJet2.E()#abs(boost1.E()-boost2.E())
			entry["Boosted_Delta_Phi"]=PhiDiff(BoostedJet1.Phi(),BoostedJet2.Phi())


			# add special variables
			entry["Delta_Phi"] = PhiDiff(entry["Reco_Higgs_B1_Phi"], entry["Reco_Higgs_B2_Phi"])
			entry["Delta_Eta"] = EtaDiff(entry["Reco_Higgs_B1_Eta"], entry["Reco_Higgs_B2_Eta"])
			entry["Delta_R"] = np.sqrt((entry["Reco_Higgs_B1_Eta"]-entry["Reco_Higgs_B2_Eta"])**2 + entry["Delta_Phi"]**2)


			new_df = new_df.append(entry, ignore_index = True)
			
			del ttHSystem
			del entry

	print("added {}/{} events".format(valid_events, nevents))
	return new_df
def findbestTopBs(df, additionalVariables):
	nevents = df.shape[0]
	print nevents,"Events found."

	# required information
	assignedJets	= ["Higgs_B1","Higgs_B2"]
	recoParticles   = ["Higgs"]

	jetVars		 = ["Pt", "Eta", "Phi", "E", "CSV"]
	particleVars	= ["Pt", "Eta", "Phi", "M", "logM"]

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


	new_df = pd.DataFrame(columns = dataframe_columns)

	# event loop
	valid_events = 0
	for iEvt in df.index:
		# get number of jets (max 10)
		event = df.loc[iEvt]
		nJets = int(min(event["N_Jets"], 10))

		#quick outprints
		if iEvt%1000 == 0:
			print "Event Nr",iEvt

		# best deltaR
		bestDeltaR1 = 10000.
		bestDeltaR2 = 10000.
		foundComb   = False
		bestIndices = np.full(2,int(1))

		for iHiggs1 in range(nJets):
			if event["Jet_CSV[{}]".format(iHiggs1)] < csvWP: continue
			deltaR_B1 = getDeltaR(event, "GenTopHad_B", iHiggs1)
			if deltaR_B1 < bestDeltaR1:
				bestDeltaR1 = deltaR_B1
				bestIndices[0] = iHiggs1

		for iHiggs2 in range(nJets):
			if iHiggs2 == bestIndices[0]: continue
			if event["Jet_CSV[{}]".format(iHiggs2)] < csvWP: continue
			deltaR_B2 = getDeltaR(event, "GenTopLep_B", iHiggs2)
			if deltaR_B2 < bestDeltaR2:
				bestDeltaR2 = deltaR_B2
				bestIndices[1] = iHiggs2
			
		if bestDeltaR1 < dRCut and bestDeltaR2 < dRCut:
			foundComb  = True

	
		# if no valid combination was found at the end of the loop, continue to next event
		if not foundComb: continue
		valid_events += 1

		assignments = []
		assignments.append(np.array(bestIndices))

 
		# fill assignments
		for ass in assignments:
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

			# add special variables
			entry["Delta_Phi"] = correctPhi( entry["Reco_Higgs_B1_Phi"] - entry["Reco_Higgs_B2_Phi"] )
			if abs(entry["Reco_Higgs_B1_Eta"]*entry["Reco_Higgs_B2_Eta"]) == entry["Reco_Higgs_B1_Eta"]*entry["Reco_Higgs_B2_Eta"]:
				entry["Delta_Eta"] = abs(entry["Reco_Higgs_B1_Eta"] - entry["Reco_Higgs_B2_Eta"])
			else:
				entry["Delta_Eta"] = abs(entry["Reco_Higgs_B1_Eta"] + entry["Reco_Higgs_B2_Eta"])
			entry["Delta_R"] = np.sqrt((entry["Reco_Higgs_B1_Eta"] - entry["Reco_Higgs_B2_Eta"])**2 + (correctPhi(entry["Reco_Higgs_B1_Phi"] - entry["Reco_Higgs_B2_Phi"]))**2
		)

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

