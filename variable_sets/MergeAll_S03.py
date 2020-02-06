variables = {}

variables["ge4j_3t"] = [
    "CSV[3]",
    "Evt_CSV_dev",
    "CSV[2]",
    "Evt_CSV_min_tagged",
    "RecoHiggs_Chi2",
    "Evt_M2_TaggedJetsAverage",
    "RecoHiggs_M",
    "Evt_Deta_TaggedJetsAverage",
    "RecoTTZ_TopHad_M",
    "Evt_blr_transformed",
    "Evt_HT_wo_MET",
    "Evt_HT_jets",
    "RecoTTZ_Chi2Total_log",
    "Evt_CSV_avg_tagged",
    "Evt_CSV_avg",
    ]


variables["5j_ge3t"] = [
    "Evt_M2_minDrTaggedJets",
    "Evt_CSV_dev",
    "RecoTTZ_TopHad_M",
    "RecoHiggs_Chi2",
    "RecoZ_Chi2_log",
    "RecoZ_Chi2",
    "RecoHiggs_M",
    "Evt_M2_TaggedJetsAverage",
    "RecoHiggs_Chi2_log",
    "CSV[3]",
    "Evt_Deta_TaggedJetsAverage",
    "CSV[2]",
    "RecoTTZ_Chi2Total_log",
    "Evt_blr_transformed",
    "Evt_CSV_avg",
    ]


variables["ge6j_ge3t"] = [
    "Evt_M2_TaggedJetsAverage",
    "Evt_M2_closestTo91TaggedJets",
    "Evt_Deta_JetsAverage",
    "RecoTTZ_Z_M_log",
    "Evt_CSV_dev",
    "Jet_Pt[5]",
    "RecoZ_Chi2_log",
    "Evt_M2_minDrTaggedJets",
    "RecoHiggs_Chi2_log",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_blr",
    "Evt_CSV_avg",
    "CSV[3]",
    "CSV[2]",
    "Evt_blr_transformed",
    ]


variables["ge4j_ge4t"] = [
    "Evt_M2_closestTo91TaggedJets",
    "RecoZ_M_log",
    "RecoHiggs_Chi2_log",
    "Evt_M2_minDrTaggedJets",
    "RecoZ_Pt",
    "CSV[3]",
    "Evt_CSV_avg_tagged",
    "RecoZ_Chi2_log",
    "CSV[1]",
    "N_LooseJets",
    "RecoHiggs_Chi2",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_blr_transformed",
    "RecoZ_Chi2",
    "CSV[2]",
    ]


variables["4j_ge3t"] = [
    "Evt_MHT",
    "Evt_Dr_JetsAverage",
    "Evt_M2_TaggedJetsAverage",
    "RecoTTZ_TopHad_W_M",
    "RecoTTZ_Chi2WHad",
    "RecoTTZ_TopLep_W_M",
    "RecoTTZ_Chi2Total",
    "RecoZ_Chi2",
    "RecoTTZ_Chi2TopHad",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_M_TaggedJetsAverage",
    "RecoTTZ_TopHad_M",
    "CSV[2]",
    "Evt_blr_transformed",
    "RecoTTZ_Chi2Total_log",
    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
