variables = {}

variables["ge4j_ge4t"] = [
    "Evt_E_TaggedJetsAverage",
    "Evt_Dr_minDrTaggedJets",
    "Jet_E[0]",
    "Evt_CSV_min_tagged",
    "Evt_M_Total",
    "Evt_M2_minDrTaggedJets",
    "Evt_CSV_avg",
    "Evt_HT_tags",
    "Jet_M[0]",
    "Evt_CSV_dev",
    "Evt_CSV_dev_tagged",
    "Evt_M2_TaggedJetsAverage",
    "Evt_M_JetsAverage",
    "Evt_MTW",
    "CSV[3]",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_Pt_minDrTaggedJets",
    "Evt_CSV_avg_tagged",
    "CSV[2]",
    "Evt_M_TaggedJetsAverage"
    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
