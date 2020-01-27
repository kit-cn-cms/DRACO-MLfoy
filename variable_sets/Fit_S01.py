variables = {}

variables["ge4j_3t"] = [
    "RecoTTZ_Chi2Total",
    "RecoHiggs_M_log",
    "RecoTTZ_TopLep_M",
    "RecoTTZ_Chi2WHad_log",
    "RecoTTZ_TopLep_W_Pt",
    "RecoTTZ_Chi2TopHad",
    "RecoZ_M",
    "RecoTTZ_TopLep_W_M",
    "RecoZ_Pt",
    "RecoZ_Chi2",
    "RecoZ_BJet2_Pt",
    "RecoTTZ_TopHad_M",
    "RecoHiggs_Chi2",
    "RecoTTZ_TopHad_W_Pt",
    "RecoTTZ_TopLep_Pt",
    "RecoZ_Chi2_log",
    "RecoHiggs_Chi2_log",
    "RecoTTZ_TopHad_Pt",
    "RecoTTZ_Dphi_topLep_topHad",
    "RecoTTZ_Chi2Total_log",
    
    "Evt_M_Total",
    "Evt_CSV_avg",
    "Evt_M2_TaggedJetsAverage",
    "Evt_M2_minDrTaggedJets",
    "Evt_Pt_minDrTaggedJets",
    ]


variables["ge4j_ge4t"] = [
    "RecoHiggs_BJet2_Pt",
    "RecoTTZ_cosdTheta_topLep_bHad",
    "RecoTTZ_Deta_Lep_bHad",
    "RecoHiggs_BJet2_M",
    "RecoZ_BJet1_Pt",
    "RecoHiggs_M_log",
    "RecoTTZ_TopHad_W_Pt",
    "RecoZ_M",
    "RecoZ_BJet1_M",
    "RecoZ_Dr",
    "RecoTTZ_Dphi_topLep_topHad",
    "RecoZ_Pt",
    "RecoTTZ_Chi2Total_log",
    "RecoZ_M_log",
    "RecoZ_BJet2_Pt",
    "RecoTTZ_TopLep_W_M",
    "RecoZ_Chi2",
    "RecoHiggs_Chi2",
    "RecoHiggs_Chi2_log",
    "RecoZ_Chi2_log",
    
    "N_Jets",

    #"Evt_Dr_TaggedJetsAverage",
    "Evt_blr_transformed",
    "Evt_Dr_minDrTaggedJets",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_M2_minDrTaggedJets",
    "Evt_Pt_minDrTaggedJets",
    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
