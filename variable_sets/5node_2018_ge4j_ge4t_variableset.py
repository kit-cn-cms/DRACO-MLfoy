variables = {}

variables["ge4j_ge4t"] = [
    "Evt_HT_tags",
    "Evt_CSV_min_tagged",
    "Evt_CSV_avg",
    "Jet_DeepJetCSV[0]",
    "Evt_M_Total",
    "Jet_Pt[0]",
    "N_BTagsM",
    "Evt_E_TaggedJetsAverage",
    "Evt_CSV_dev_tagged",
    "Evt_CSV_dev",
    "Jet_M[0]",
    "Evt_M2_TaggedJetsAverage",
    "Evt_MTW",
    "Evt_Pt_minDrTaggedJets",
    "CSV[3]",
    "Evt_M_JetsAverage",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_CSV_avg_tagged",
    "CSV[2]",
    "Evt_M_TaggedJetsAverage",
    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
