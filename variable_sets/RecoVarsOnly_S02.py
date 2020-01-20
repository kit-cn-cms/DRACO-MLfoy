variables = {}

variables["ge4j_3t"] = [
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
    ]


variables["5j_ge3t"] = [
    "RecoHiggs_Chi2",
    "RecoHiggs_M_log",
    "RecoZ_Chi2",
    "RecoTTZ_TopHad_W_M",
    "RecoTTZ_TopHad_W_Pt",
    "RecoTTZ_TopHad_M",
    "RecoTTZ_Chi2WHad_log",
    "RecoZ_Chi2_log",
    "RecoTTZ_Chi2Total_log",
    "RecoHiggs_Chi2_log",
    ]


variables["ge6j_ge3t"] = [
    "RecoTTZ_TopHad_M",
    "RecoTTZ_Chi2Total_log",
    "RecoTTZ_Z_BJet1_Pt",
    "RecoTTZ_TopHad_W_M",
    "RecoTTZ_Chi2Z",
    "RecoTTZ_Z_M_log",
    "RecoTTZ_Z_BJet2_Pt",
    "RecoTTZ_Chi2Z_log",
    "RecoZ_Chi2_log",
    "RecoHiggs_Chi2_log",
    ]


variables["ge4j_ge4t"] = [
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
    ]


variables["4j_ge3t"] = [
    "RecoTTZ_Chi2Total",
    "RecoTTZ_Chi2TopHad",
    "RecoZ_BJet2_Pt",
    "RecoHiggs_Chi2_log",
    "RecoZ_Chi2_log",
    "RecoTTZ_Chi2WHad",
    "RecoTTZ_TopHad_W_M",
    "RecoTTZ_TopLep_W_M",
    "RecoTTZ_TopHad_M",
    "RecoTTZ_Chi2Total_log",
    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
