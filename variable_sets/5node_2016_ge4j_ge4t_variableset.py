variables = {}

variables["ge4j_ge4t"] = [
    "Jet_Pt[0]",
    "Evt_HT_tags",
    "Jet_DeepJetCSV[0]",
    "N_BTagsM",
    "Evt_M_Total",
    "Evt_CSV_avg",
    "Evt_E_TaggedJetsAverage",
    "Evt_CSV_dev",
    "Evt_CSV_min_tagged",
    "Evt_M2_TaggedJetsAverage",
    "Evt_CSV_dev_tagged",
    "Evt_MTW",
    "Jet_M[0]",
    "CSV[3]",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_M_JetsAverage",
    "Evt_Pt_minDrTaggedJets",
    "Evt_CSV_avg_tagged",
    "CSV[2]",
    "Evt_M_TaggedJetsAverage",
    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
