variables = {}

variables["4j_ge3t"] = [
    "Evt_Deta_TaggedJetsAverage",
    "Evt_M2_closestTo125TaggedJets",
    "Evt_M2_closestTo91TaggedJets",
    "Evt_M2_minDrJets",
    "Evt_M2_minDrTaggedJets",
    "RecoHiggs_BJet1_Pt",
    "RecoHiggs_Chi2_log",
    "RecoTTZ_Chi2Total",
    "RecoZ_BJet1_M",
    "RecoZ_Chi2_log",
    ]


variables["5j_ge3t"] = [
    "Evt_Dr_closestTo91TaggedJets",
    "Evt_M2_closestTo125TaggedJets",
    "Evt_M2_closestTo91TaggedJets",
    "Evt_M2_minDrTaggedJets",
    "Evt_Pt_minDrTaggedJets",
    "RecoHiggs_Chi2",
    "RecoHiggs_Chi2_log",
    "RecoHiggs_M",
    "RecoHiggs_M_log",
    "RecoZ_Chi2_log",
    ]


variables["ge6j_ge3t"] = [
    "Evt_Dr_closestTo91TaggedJets",
    "Evt_M2_closestTo125TaggedJets",
    "Evt_M2_closestTo91TaggedJets",
    "Evt_M2_minDrTaggedJets",
    "Evt_Pt_minDrTaggedJets",
    "RecoHiggs_Chi2_log",
    "RecoTTZ_Chi2Z",
    "RecoTTZ_Chi2Z_log",
    "RecoTTZ_Z_M_log",
    "RecoZ_Chi2_log",
    ]


variables["ge4j_3t"] = [
    "Evt_M2_TaggedJetsAverage",
    "Evt_M2_closestTo125TaggedJets",
    "Evt_M2_closestTo91TaggedJets",
    "Evt_M2_minDrTaggedJets",
    "Evt_Pt_minDrTaggedJets",
    "RecoHiggs_Chi2_log",
    "RecoHiggs_M",
    "RecoHiggs_M_log",
    "RecoZ_BJet1_Eta",
    "RecoZ_Chi2_log",
    ]


variables["ge4j_ge4t"] = [
    "Evt_Deta_TaggedJetsAverage",
    "Evt_M2_closestTo125TaggedJets",
    "Evt_M2_closestTo91TaggedJets",
    "Evt_M2_minDrTaggedJets",
    "Evt_Pt_minDrTaggedJets",
    "RecoHiggs_Chi2",
    "RecoHiggs_Chi2_log",
    "RecoHiggs_M_log",
    "RecoZ_Chi2",
    "RecoZ_Chi2_log",
    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
