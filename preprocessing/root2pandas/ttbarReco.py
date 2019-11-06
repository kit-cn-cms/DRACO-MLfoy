import sys
import os
from math import sin, cos, log
import numpy as np
import pandas as pd
import ROOT
import copy

# temporary hard coded bullshit
csvWP = 0.277
dRCut = 0.2
nWrongAssignments = 1

def findbest(df, additionalVariables):
    nevents = df.shape[0]

    # required information
    assignedJets    = ["TopHad_B", "TopLep_B", "TopHad_Q1", "TopHad_Q2"]
    recoParticles   = ["TopHad", "TopLep", "WHad", "WLep"]

    jetVars         = ["Pt", "Eta", "Phi", "E", "CSV"]
    particleVars    = ["Pt", "Eta", "Phi", "M", "logM"]

    # initialize list with columns to be written into dataframe
    dataframe_columns = copy.deepcopy(additionalVariables)

    for j in assignedJets:
        for v in jetVars:
            dataframe_columns.append("RecoTT_"+j+"_"+v)
    for p in recoParticles:
        for v in particleVars:
            dataframe_columns.append("RecoTT_"+p+"_"+v)

    dataframe_columns.append("is_ttbar")

    # define some variables to be calculated later on
    dataframe_columns.append("RecoTT_ttbar_Phi")
    dataframe_columns.append("RecoTT_ttbar_energy_fraction")

    new_df = pd.DataFrame(columns = dataframe_columns)

    # event loop
    valid_events = 0
    for iEvt in df.index:
        # get number of jets (max 10)
        event = df.loc[iEvt]
        nJets = int(min(event["N_Jets"], 10))

        # best deltaR
        bestDeltaR  = 10000.
        foundComb   = False
        bestIndices = []

        for iHadB in range(nJets):
            if event["Jet_CSV[{}]".format(iHadB)] < csvWP: continue
            deltaR_bHad = getDeltaR(event, "GenTopHad_B", iHadB)
            if deltaR_bHad > dRCut: continue

            for iLepB in range(nJets):
                if iLepB == iHadB: continue
                if event["Jet_CSV[{}]".format(iLepB)] < csvWP: continue
                deltaR_bLep = getDeltaR(event, "GenTopLep_B", iLepB)
                if deltaR_bLep > dRCut: continue

                for iHadQ1 in range(nJets):
                    if iHadQ1 == iLepB or iHadQ1 == iHadB: continue
                    deltaR_Q1       = getDeltaR(event, "GenTopHad_Q1", iHadQ1)
                    deltaR_Q1_swp   = getDeltaR(event, "GenTopHad_Q2", iHadQ1)
                    if deltaR_Q1 > dRCut: continue
        
                    for iHadQ2 in range(nJets):
                        if iHadQ2 == iHadQ1 or iHadQ2 == iLepB or iHadQ2 == iHadB: continue
                        deltaR_Q2       = getDeltaR(event, "GenTopHad_Q2", iHadQ2)
                        deltaR_Q2_swp   = getDeltaR(event, "GenTopHad_Q1", iHadQ2)
                        if deltaR_Q2 > dRCut: continue

                        # ignore Q1/Q2 swapped combinations that are worse than the non-swapped
                        if (deltaR_Q1_swp+deltaR_Q2_swp < deltaR_Q1+deltaR_Q2): continue
                
                        totalDeltaR = deltaR_bHad + deltaR_bLep + deltaR_Q1 + deltaR_Q2

                        if totalDeltaR < bestDeltaR:
                            bestDeltaR = totalDeltaR
                            foundComb  = True

                            bestIndices = [iHadB, iLepB, iHadQ1, iHadQ2]
    
        # if no valid combination was found at the end of the loop, continue to next event
        if not foundComb: continue
        valid_events += 1

        lepton, neutrino = getLepAndNeutrino(event)

        assignments = []
        assignments.append(np.array(bestIndices))

        # generate random wrong assignments
        for iWrong in range(nWrongAssignments):
            foundNew = False
            while not foundNew:
                wrong = np.random.permutation(nJets)[:4]
                foundNew = True
                for p in assignments:
                    if (p==wrong).any():
                        foundNew = False
            assignments.append(wrong)

        # fill assignments
        for idx, ass in enumerate(assignments):
            # fill variables
            entry = {v: None for v in dataframe_columns}
            for v in additionalVariables:
                entry[v] = event[v]
            if idx == 0:
                entry["is_ttbar"] = 1
            else:
                entry["is_ttbar"] = 0
            
            # fill assigned jets
            # DANGERZONE: make sure the order of entries in assignedJets list is the same as in the assignments list
            for it, j in enumerate(assignedJets):
                for v in jetVars:
                    entry["RecoTT_"+j+"_"+v] = event["Jet_{}[{}]".format(v, ass[it])]

            # calculate ttbar system and write variables
            ttbarSystem = reconstruct_ttbar(entry, assignedJets, lepton, neutrino)
            for p in recoParticles:
                for v in particleVars:
                    entry["RecoTT_"+p+"_"+v] = ttbarSystem.get(p,v)

            # add special variables
            entry["RecoTT_ttbar_Phi"] = correctPhi( entry["RecoTT_TopHad_Phi"] - entry["RecoTT_TopLep_Phi"] )
            entry["RecoTT_ttbar_energy_fraction"] = (entry["RecoTT_TopHad_Pt"] + entry["RecoTT_TopLep_Pt"])/(event["Evt_HT"])

            new_df = new_df.append(entry, ignore_index = True)
            
            del ttbarSystem
            del entry

    print("added {}/{} events".format(valid_events, nevents))
    return new_df

#function to correct a difference of two angulars phi which is in [-2pi,2pi] to the correct interval [-pi,pi]
def getDeltaR(event, genVar, jetIndex):
    return np.sqrt(
        (event["Jet_Eta[{}]".format(jetIndex)] - event[genVar+"_Eta"])**2 + (correctPhi(event["Jet_Phi[{}]".format(jetIndex)] - event[genVar+"_Phi"]))**2
        )

def correctPhi(phi):
    if(phi  <=  -np.pi):
        phi += 2*np.pi
    if(phi  >    np.pi):
        phi -= 2*np.pi
    return phi

def getLepAndNeutrino(event):
    # get neutrino and lepton
    lepton_4vec = ROOT.TLorentzVector()
    lepton_4vec.SetPtEtaPhiE( event["LooseLepton_Pt[0]"], event["LooseLepton_Eta[0]"], event["LooseLepton_Phi[0]"], event["LooseLepton_E[0]"] )

    MET_pT  = event["Evt_MET_Pt"]
    MET_Phi = event["Evt_MET_Phi"]
    mW      = 80.4

    nu_4vec = ROOT.TLorentzVector( MET_pT*cos(MET_Phi), MET_pT*sin(MET_Phi), 0., MET_pT )
    mu      = ((mW*mW)/2.) + lepton_4vec.Px()*nu_4vec.Px() + lepton_4vec.Py()*nu_4vec.Py()
    a       = (mu*lepton_4vec.Pz())/(lepton_4vec.Pt()**2)
    a2      = a**2
    b       = (lepton_4vec.E()**2*nu_4vec.Pt()**2 - mu**2)/(lepton_4vec.Pt()**2)
    
    if a2 < b: nu_4vec.SetPz(a)
    else:
        pz1 = a+(a2-b)**0.5
        pz2 = a-(a2-b)**0.5
        if( abs(pz1) <= abs(pz2) ):
            nu_4vec.SetPz(pz1)
        else:
            nu_4vec.SetPz(pz2)

    nu_4vec.SetE(nu_4vec.P())

    return lepton_4vec, nu_4vec

class reconstruct_ttbar:
    def __init__(self, entry, jets, lepvec, nuvec):
        vectors = {}
        vectors["Lepton"]   = lepvec
        vectors["Nu"]       = nuvec

        for j in jets:
            vectors[j] = ROOT.TLorentzVector()
            vectors[j].SetPtEtaPhiE( entry["RecoTT_"+j+"_Pt"], entry["RecoTT_"+j+"_Eta"], entry["RecoTT_"+j+"_Phi"], entry["RecoTT_"+j+"_E"] )
            
        vectors["WHad"] = vectors["TopHad_Q1"] + vectors["TopHad_Q2"]
        vectors["WLep"] = vectors["Lepton"] + vectors["Nu"]
        vectors["TopHad"] = vectors["WHad"] + vectors["TopHad_B"]
        vectors["TopLep"] = vectors["WLep"] + vectors["TopLep_B"]

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
        else:
            exit("error in get reco var")

