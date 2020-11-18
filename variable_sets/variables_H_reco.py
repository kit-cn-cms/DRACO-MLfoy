variables = {}
variables["ge4j_ge3t"] = [
    "RecoHiggs_B1_Eta",
    "RecoHiggs_B2_Eta",
    "RecoHiggs_B1_Phi",
    "RecoHiggs_B2_Phi",
    "RecoHiggs_B1_btagValue",
    "RecoHiggs_B2_btagValue",
    "RecoHiggs_B1_M",
    "RecoHiggs_B2_M",
    "RecoHiggs_B1_E",
    "RecoHiggs_B2_E",
    "RecoHiggs_B1_Pt",
    "RecoHiggs_B2_Pt",
    "RecoHiggs_B1_idx",
    "RecoHiggs_B2_idx",
    "N_Jets",
    "N_BTagsM",
    "RecoHiggs_H_Pt",
    "RecoHiggs_H_Eta",
    "RecoHiggs_H_M",
    "RecoHiggs_H_E",
    "RecoHiggs_H_openingAngle",
    "RecoHiggs_H_dPhi",
    "RecoHiggs_H_dEta",
    "RecoHiggs_H_dPt",
    "RecoHiggs_H_dR",
    #"RecoHiggs_H_dKin",
    ] 
all_variables = list(set( [v for key in variables for v in variables[key] ] ))


