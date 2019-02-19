import os
import sys
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(os.path.dirname(filedir))
sys.path.append(basedir)

import root2pandas
import variable_sets.ntuplesVariablesWithIndex as variable_set



# define a base event selection which is applied for all Samples
base_selection = "(\
(N_LooseMuons == 0 and N_TightElectrons == 1) \
or \
(N_LooseElectrons == 0 and N_TightMuons == 1) \
)"

categories = root2pandas.EventCategories()
categories.addCategory("ttbar", selection = None)

# initialize dataset class
dataset = root2pandas.Dataset(
    outputdir   = "/nfs/dust/cms/user/vdlinden/DNNInputFiles/ttbarMatcher/",
    naming      = "input",
    addCNNmap   = False,
    addMEM      = False,
    maxEntries  = 200000)

# add base event selection
dataset.addBaseSelection(base_selection)

# add samples to dataset
dataset.addSample(
    sampleName  = "ttbar",
    ntuples     = "/nfs/dust/cms/user/vdlinden/ttH_2018/ntuples/ntuples_ttbarMatching/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root",
    categories  = categories)
    
# initialize variable list 
dataset.addAllVariablesNoIndex()

ignoredVariables = [
    "Evt_HT",
    "Evt_HT_Jets",
    "Evt_M3",
    "Evt_MHT",
    "Evt_MTW",
    "Evt_M_Total",
    "Evt_Phi_GenMET",
    "Evt_Phi_MET",
    "Evt_Pt_GenMET",
    "Evt_Pt_MET",
    "ttMatchInputMET_CSV_DNN",
    "ttMatchInputMET_DeepCSV_b",
    "ttMatchInputMET_DeepCSV_bb",
    "ttMatchInputMET_DeepCSV_c",
    "ttMatchInputMET_DeepCSV_udsg",
    "ttMatchInputMET_DeepFlavor_lepb",
    "N_BTagsL",
    "N_BTagsT",
    "N_GenPVs",
    "N_LooseElectrons",
    "N_LooseJets",
    "N_LooseMuons",
    "N_PrimaryVertices",
    "N_TightElectrons",
    "N_TightLeptons",
    "N_TightMuons",
    "CSV",
    "CSV_DNN",
    "Electron_Charge",
    "Electron_E",
    "Electron_Eta",
    "Electron_Eta_Supercluster",
    "Electron_M",
    "Electron_Phi",
    "Electron_Pt",
    "Electron_Pt_BeforeRun2Calibration",
    "Electron_RelIso",
    "Jet_CSV",
    "Jet_CSV_DNN",
    "Jet_Charge",
    "Jet_DeepCSV_b",
    "Jet_DeepCSV_bb",
    "Jet_DeepCSV_c",
    "Jet_DeepCSV_udsg",
    "Jet_DeepFlavour_b",
    "Jet_DeepFlavour_bb",
    "Jet_DeepFlavour_c",
    "Jet_DeepFlavour_g",
    "Jet_DeepFlavour_lepb",
    "Jet_DeepFlavour_uds",
    "Jet_E",
    "Jet_Eta",
    "Jet_Flav",
    "Jet_GenJet_Eta",
    "Jet_GenJet_Pt",
    "Jet_M",
    "Jet_PartonFlav",
    "Jet_Phi",
    "Jet_PileUpID",
    "Jet_PileUpMVA",
    "Jet_Pt",
    "LooseLepton_E",
    "LooseLepton_Eta",
    "LooseLepton_M",
    "LooseLepton_Phi",
    "LooseLepton_Pt",
    "Muon_Charge",
    "Muon_E",
    "Muon_Eta",
    "Muon_M",
    "Muon_Phi",
    "Muon_Pt",
    "Muon_Pt_BeForeRC",
    "Muon_RelIso",
    "ttMatchInputJet_CSV_DNN",
    "ttMatchInputJet_DeepCSV_b",
    "ttMatchInputJet_DeepCSV_bb",
    "ttMatchInputJet_DeepCSV_c",
    "ttMatchInputJet_DeepCSV_udsg",
    "ttMatchInputJet_DeepFlavor_lepb",
    "ttMatchInputLepton_CSV_DNN",
    "ttMatchInputLepton_DeepCSV_b",
    "ttMatchInputLepton_DeepCSV_bb",
    "ttMatchInputLepton_DeepCSV_c",
    "ttMatchInputLepton_DeepCSV_udsg",
    "ttMatchInputLepton_DeepFlavor_lepb",

    "ttMatchTargetQ1_Pt",
    "ttMatchTargetLepton_Pz",
    "ttMatchTargetLepton_Py",
    "ttMatchTargetLepton_Px",
    "ttMatchTargetLepton_Pt",
    "ttMatchTargetLepB_Phi",
    "ttMatchTargetLepB_Eta",
    "ttMatchTargetLepton_Phi",
    "ttMatchTargetLepW_Eta",
    "ttMatchTargetQ1_Eta",
    "ttMatchTargetQ1_Pz",
    "ttMatchTargetHadB_Pz",
    "ttMatchTargetHadB_Px",
    "ttMatchTargetHadB_Py",
    "ttMatchTargetQ1_Py",
    "ttMatchTargetQ1_Px",
    "ttMatchTargetHadB_Pt",
    "ttMatchTargetLepton_M",
    "ttMatchTargetQ2_Pt",
    "ttMatchTargetQ2_Pz",
    "ttMatchTargetQ2_Px",
    "ttMatchTargetLepton_E",
    "ttMatchTargetLepW_Phi",
    "ttMatchTargetHadW_Eta",
    "ttMatchTargetHadW_Py",
    "ttMatchTargetHadW_Px",
    "ttMatchTargetHadW_Pz",
    "ttMatchTargetLepW_E",
    "ttMatchTargetLepW_M",
    "ttMatchTargetHadB_Phi",
    "ttMatchTargetQ1_Phi",
    "ttMatchTargetLepB_E",
    "ttMatchTargetQ1_E",
    "ttMatchTargetLepB_M",
    "ttMatchTargetLepB_Pt",
    "ttMatchTargetQ2_Py",
    "ttMatchTargetLepB_Pz",
    "ttMatchTargetLepB_Px",
    "ttMatchTargetLepB_Py",
    "ttMatchTargetQ1_M",
    "ttMatchTargetLepton_Eta",
    "ttMatchTargetHadB_Eta",
    "ttMatchTargetHadW_M",
    "ttMatchTargetHadW_E",
    "ttMatchTargetQ2_E",
    "ttMatchTargetQ2_M",
    "ttMatchTargetQ2_Eta",
    "ttMatchTargetLepW_Pt",
    "ttMatchTargetQ2_Phi",
    "ttMatchTargetHadW_Phi",
    "ttMatchTargetHadB_M",
    "ttMatchTargetHadB_E",
    "ttMatchTargetHadW_Pt",
    "ttMatchTargetLepW_Px",
    "ttMatchTargetLepW_Pz",
    ]

dataset.removeVariables(ignoredVariables)

# run the preprocessing
dataset.runPreprocessing(figureOutVectors = True)
