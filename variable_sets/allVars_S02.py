variables = {}

variables["ge4j_3t"] = [
    "CSV[3]",
    "Evt_HT_jets",
    "Evt_CSV_min_tagged",
    "CSV[2]",
    "Evt_CSV_dev",
    "Evt_M2_TaggedJetsAverage",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_blr_transformed",
    "RecoTTZ_Chi2Total_log",
    "Evt_CSV_avg",
    ]


variables["5j_ge3t"] = [
    "Evt_CSV_avg",
    "Evt_Dr_JetsAverage",
    "Evt_CSV_dev",
    "RecoZ_Chi2_log",
    "CSV[2]",
    "RecoHiggs_Chi2_log",
    "RecoTTZ_Chi2Total_log",
    "CSV[3]",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_blr_transformed",
    ]


variables["ge6j_ge3t"] = [
    "Jet_Pt[5]",
    "Evt_blr",
    "RecoZ_Chi2_log",
    "N_BTagsM",
    "Evt_CSV_dev",
    "RecoHiggs_Chi2_log",
    "Evt_Deta_TaggedJetsAverage",
    "CSV[3]",
    "Evt_blr_transformed",
    "CSV[2]",
    ]


variables["ge4j_ge4t"] = [
    "CSV[1]",
    "Evt_CSV_avg_tagged",
    "Evt_h0",
    "N_LooseJets",
    "RecoHiggs_Chi2",
    "RecoHiggs_Chi2_log",
    "Evt_Deta_TaggedJetsAverage",
    "RecoZ_Chi2_log",
    "Evt_blr_transformed",
    "CSV[2]",
    ]


variables["4j_ge3t"] = [
    "Evt_M_TaggedJetsAverage",
    "RecoZ_Chi2_log",
    "RecoTTZ_Chi2WHad",
    "RecoTTZ_TopLep_W_M",
    "Evt_Deta_TaggedJetsAverage",
    "CSV[2]",
    "RecoTTZ_TopHad_M",
    "Evt_MHT",
    "Evt_blr_transformed",
    "RecoTTZ_Chi2Total_log",
    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
