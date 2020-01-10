variables = {}

variables["5j_ge3t"] = [
    "Evt_M3_oneTagged",
    "Evt_M_TaggedJetsAverage",
    "Evt_Dr_JetsAverage",
    "Evt_CSV_dev",
    "Evt_Pt_minDrTaggedJets",
    "Evt_M2_TaggedJetsAverage",
    "CSV[2]",
    "CSV[3]",
    "Evt_blr_transformed",
    "Evt_Deta_TaggedJetsAverage",
    ]


variables["ge6j_ge3t"] = [
    "Evt_blr",
    "Evt_CSV_dev",
    "Evt_Dr_closestTo91TaggedJets",
    "Evt_Deta_JetsAverage",
    "Evt_M2_TaggedJetsAverage",
    "Evt_Pt_minDrTaggedJets",
    "CSV[3]",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_blr_transformed",
    "CSV[2]",
    ]


variables["4j_ge3t"] = [
    "Evt_M_TaggedJetsAverage",
    "CSV[2]",
    "Evt_MHT",
    "Evt_M2_TaggedJetsAverage",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_Pt_minDrJets",
    "Evt_Pt_minDrTaggedJets",
    "Evt_MTW",
    "Evt_M3",
    "Evt_blr_transformed",
    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
