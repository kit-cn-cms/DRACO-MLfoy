variables = {}

variables["4j_ge3t"] = [
    "CSV[2]",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_Dr_TaggedJetsAverage",
    "Evt_Dr_maxDrTaggedJets",
    "Evt_E_TaggedJetsAverage",
    "Evt_M_TaggedJetsAverage",
    "Evt_Pt_minDrTaggedJets",
    "Evt_h1",
    "Jet_M[3]",
    "Jet_Pt[1]",
    "N_BTagsM",
    "RecoHiggs_BJet1_E",
    "RecoHiggs_Chi2",
    "RecoHiggs_Chi2_log",
    "RecoHiggs_Dr",
    "RecoHiggs_M",
    "RecoHiggs_M_log",
    "RecoTTZ_Chi2TopHad_log",
    "RecoTTZ_Chi2WHad_log",
    "RecoTTZ_Deta_Lep_topHad",
    "RecoZ_BJet1_M",
    "RecoZ_Chi2",
    "RecoZ_Chi2_log",
    "RecoZ_Pt",
    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
