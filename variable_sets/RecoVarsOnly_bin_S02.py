variables = {}

variables["ge4j_3t"] = [
    "RecoZ_BJet1_Eta",
    "RecoHiggs_M",
    "RecoHiggs_M_log",
    "RecoHiggs_Chi2_log",
    "RecoZ_Chi2_log",
    ]


variables["5j_ge3t"] = [
    "RecoHiggs_Chi2",
    "RecoHiggs_M",
    "RecoHiggs_M_log",
    "RecoHiggs_Chi2_log",
    "RecoZ_Chi2_log",
    ]


variables["ge6j_ge3t"] = [
    "RecoTTZ_Z_M_log",
    "RecoZ_Chi2_log",
    "RecoTTZ_Chi2Z",
    "RecoTTZ_Chi2Z_log",
    "RecoHiggs_Chi2_log",
    ]


variables["ge4j_ge4t"] = [
    "RecoHiggs_M_log",
    "RecoHiggs_Chi2_log",
    "RecoZ_Chi2_log",
    "RecoZ_Chi2",
    "RecoHiggs_Chi2",
    ]


variables["4j_ge3t"] = [
    "RecoTTZ_Chi2Total",
    "RecoHiggs_BJet1_Pt",
    "RecoZ_BJet1_M",
    "RecoHiggs_Chi2_log",
    "RecoZ_Chi2_log",
    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
