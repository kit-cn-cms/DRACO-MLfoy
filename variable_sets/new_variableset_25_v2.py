variables = {}

variables["ge4j_ge3t"] = [
    "Evt_HT_jets",
    "LooseLepton_Pt[0]",
    "CSV[2]",
    "Evt_Dr_TaggedJetsAverage",
    "Evt_Dr_minDrJets",
    "Evt_Deta_JetsAverage",
    "Evt_M_Total",
    "Evt_Dr_JetsAverage",
    "Evt_Pt_minDrJets",
    "Evt_M_TaggedJetsAverage",
    "Evt_Eta_TaggedJetsAverage",
    "Evt_Dr_minDrTaggedJets",
    "Evt_CSV_avg",
    "Evt_M2_minDrTaggedJets",
    "Evt_Dr_minDrLepTag",
    "LooseLepton_Eta[0]",
    "Evt_Deta_TaggedJetsAverage",
    "Jet_M[0]",
    "Evt_M_minDrLepTag",
    "Evt_Pt_minDrTaggedJets",
    "Jet_Pt[0]",
    "CSV[3]",
    "N_Jets",
    "Evt_M3",
    "Evt_M2_closestTo125TaggedJets",
    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
