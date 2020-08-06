variables = {}
variables["ge4j_ge3t"] = [
    
    # Z boson stuff
    "RecoZ_B1_Eta",
    "RecoZ_B2_Eta",
    "RecoZ_B1_Phi",
    "RecoZ_B2_Phi",
    "RecoZ_B1_btagValue",
    "RecoZ_B2_btagValue",
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


    # Higgs stuff
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

