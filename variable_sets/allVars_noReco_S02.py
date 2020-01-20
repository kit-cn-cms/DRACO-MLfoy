variables = {}

variables["ge4j_3t"] = [
    "Evt_HT_jets",
    "Evt_HT_wo_MET",
    "CSV[2]",
    "Evt_CSV_min_tagged",
    "Evt_CSV_avg_tagged",
    "Evt_M2_TaggedJetsAverage",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_blr_transformed",
    "Evt_CSV_avg",
    "Evt_Pt_minDrTaggedJets",
    ]


variables["5j_ge3t"] = [
    "Evt_MTW",
    "Evt_Dr_JetsAverage",
    "Evt_CSV_dev",
    "Evt_M3_oneTagged",
    "Evt_M2_TaggedJetsAverage",
    "CSV[3]",
    "CSV[2]",
    "Evt_Pt_minDrTaggedJets",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_blr_transformed",
    ]


variables["ge6j_ge3t"] = [
    "Jet_Pt[5]",
    "Evt_Deta_JetsAverage",
    "Evt_blr",
    "Evt_CSV_dev",
    "Evt_M2_TaggedJetsAverage",
    "Evt_Pt_minDrTaggedJets",
    "CSV[3]",
    "Evt_Deta_TaggedJetsAverage",
    "CSV[2]",
    "Evt_blr_transformed",
    ]


variables["ge4j_ge4t"] = [
    "Evt_h0",
    "CSV[3]",
    "Evt_MTW",
    "N_LooseJets",
    "Evt_CSV_avg_tagged",
    "CSV[1]",
    "Evt_M2_closestTo91TaggedJets",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_blr_transformed",
    "CSV[2]",
    ]


variables["4j_ge3t"] = [
    "Evt_Dr_JetsAverage",
    "Evt_HT_wo_MET",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_Pt_minDrJets",
    "Evt_MHT",
    "Evt_MTW",
    "Evt_M2_TaggedJetsAverage",
    "Evt_blr_transformed",
    "Evt_Pt_minDrTaggedJets",
    "Evt_M3",
    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
