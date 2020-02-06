variables = {}

variables["ge4j_3t"] = [
    "Evt_M2_TaggedJetsAverage",
    "Evt_CSV_avg",
    "Evt_Deta_TaggedJetsAverage",
    "RecoHiggs_Chi2_log",
    "RecoZ_Chi2_log",
    ]


variables["5j_ge3t"] = [
    "RecoHiggs_M_log",
    "RecoHiggs_Chi2",
    "Evt_Deta_TaggedJetsAverage",
    "RecoHiggs_Chi2_log",
    "RecoZ_Chi2_log",
    ]


variables["ge6j_ge3t"] = [
    "RecoHiggs_Chi2_log",
    "Evt_CSV_avg",
    "RecoTTZ_Chi2Z_log",
    "RecoTTH_Chi2Higgs_log",
    "RecoZ_Chi2_log",
    ]


variables["ge4j_ge4t"] = [
    "Evt_Deta_TaggedJetsAverage",
    "RecoHiggs_Chi2_log",
    "RecoZ_Chi2",
    "RecoZ_Chi2_log",
    "RecoHiggs_Chi2",
    ]


variables["4j_ge3t"] = [
    "RecoZ_Chi2",
    "RecoTTZ_Chi2TopHad_log",
    "RecoTTZ_Chi2WHad_log",
    "RecoHiggs_Chi2_log",
    "RecoZ_Chi2_log",
    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
