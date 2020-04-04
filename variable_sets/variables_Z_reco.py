variables = {}
variables["ge4j_ge3t"] = [
    "RecoZ_B1_Eta",
    "RecoZ_B2_Eta",
    "RecoZ_B1_Phi",
    "RecoZ_B2_Phi",
    "RecoZ_B1_CSV",
    "RecoZ_B2_CSV",
    "RecoZ_B1_M",
    "RecoZ_B2_M",
    "RecoZ_B1_E",
    "RecoZ_B2_E",
    "RecoZ_B1_Pt",
    "RecoZ_B2_Pt",
    "RecoZ_B1_idx",
    "RecoZ_B2_idx",
    "N_Jets",
    "N_BTagsM",
    "RecoZ_Z_Pt",
    "RecoZ_Z_Eta",
    "RecoZ_Z_M",
    "RecoZ_Z_E",
    "RecoZ_Z_openingAngle",
    "RecoZ_Z_dPhi",
    "RecoZ_Z_dEta",
    "RecoZ_Z_dPt",
    "RecoZ_Z_dR",
    #"RecoZ_Z_dKin",
    ] 
all_variables = list(set( [v for key in variables for v in variables[key] ] ))


