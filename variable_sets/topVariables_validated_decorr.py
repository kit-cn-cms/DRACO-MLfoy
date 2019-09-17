variables = {}

variables["4j_ge3t"] = [
    "CSV[2]",
    "CSV[3]",
    #"Evt_CSV_avg",
    "Evt_CSV_avg_tagged",
    "Evt_CSV_dev",
    "Evt_CSV_min",
    "Evt_CSV_min_tagged",
    "Evt_Deta_JetsAverage",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_Dr_minDrJets",
    "Evt_Dr_minDrTaggedJets",
    #"Evt_M2_TaggedJetsAverage",
    "Evt_M2_closestTo125TaggedJets",
    "Evt_M2_closestTo91TaggedJets",
    #"Evt_M2_minDrJets",
    #"Evt_M2_minDrTaggedJets",
    "Evt_MHT",
    #"Evt_MTW",
    #"Evt_M_TaggedJetsAverage",
    "Evt_Pt_TaggedJetsAverage",
    #"Evt_Pt_minDrJets",
    #"Evt_Pt_minDrTaggedJets",
    "Evt_blr_transformed",
    "RecoTTZ_Chi2Total_log",
    "RecoTTZ_Chi2WHad_log",
    "RecoTTZ_TopHad_M",
    #"RecoTTZ_TopHad_Pt",
    "RecoTTZ_TopHad_W_M",
    ]


variables["5j_ge3t"] = [
    "CSV[2]",
    "CSV[3]",
    "Evt_CSV_avg",
    "Evt_CSV_avg_tagged",
    "Evt_CSV_dev",
    "Evt_CSV_min",
    #"Evt_Deta_JetsAverage",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_Dr_JetsAverage",
    "Evt_Dr_TaggedJetsAverage",
    "Evt_Dr_minDrJets",
    "Evt_Dr_minDrTaggedJets",
    "Evt_M2_TaggedJetsAverage",
    "Evt_M2_closestTo125TaggedJets",
    "Evt_M2_closestTo91TaggedJets",
    "Evt_M2_minDrJets",
    #"Evt_M2_minDrTaggedJets",
    "Evt_M_TaggedJetsAverage",
    "Evt_M_Total",
    "Evt_Pt_minDrJets",
    #"Evt_Pt_minDrTaggedJets",
    "Evt_blr",
    "Evt_blr_transformed",
    "Evt_h1",
    "RecoTTZ_Chi2Total_log",
    "RecoTTZ_Chi2WHad_log",
    ]


variables["ge6j_ge3t"] = [
    #"CSV[2]",
    "CSV[3]",
    "Evt_CSV_avg",
#"Evt_CSV_avg_tagged",
    "Evt_CSV_dev",
    "Evt_Deta_JetsAverage",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_M2_TaggedJetsAverage",
    "Evt_M2_closestTo125TaggedJets",
    "Evt_M2_closestTo91TaggedJets",
    "Evt_M2_minDrTaggedJets",
    "Evt_M_TaggedJetsAverage",
    "Evt_Pt_minDrTaggedJets",
    "Evt_blr",
    "Evt_blr_transformed",
    "Jet_Pt[0]",
    "N_BTagsM",
    #"RecoTTZ_Chi2Total_log",
    "RecoTTZ_Chi2Z_log",
    "RecoTTZ_Z_M_log",
    "RecoTTZ_Z_Pt",
    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
