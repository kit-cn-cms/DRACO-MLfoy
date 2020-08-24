variables = {}
variables["ge4j_ge3t"] = [
    
    # X boson stuff
    "RecoX_B1_Eta",
    "RecoX_B2_Eta",
    "RecoX_B1_Phi",
    "RecoX_B2_Phi",
    "RecoX_B1_btagValue",
    "RecoX_B2_btagValue",
    "RecoX_B1_M",
    "RecoX_B2_M",
    "RecoX_B1_E",
    "RecoX_B2_E",
    "RecoX_B1_Pt",
    "RecoX_B2_Pt",
    "RecoX_B1_idx",
    "RecoX_B2_idx",
    "N_Jets",
    "N_BTagsM",
    "RecoX_X_Pt",
    "RecoX_X_Eta",
    "RecoX_X_M",
    "RecoX_X_E",
    "RecoX_X_openingAngle",
    "RecoX_X_dPhi",
    "RecoX_X_dEta",
    "RecoX_X_dPt",
    "RecoX_X_dR",
    #"RecoX_X_dKin",


    "TightLepton_E_0",
    "TightLepton_Eta_0",
    "TightLepton_M_0",
    "TightLepton_Phi_0",
    "TightLepton_Pt_0",

    "RecoX_B1_dEta_lept",
    "RecoX_B2_dEta_lept",
    "RecoX_B1_dPhi_lept",
    "RecoX_B2_dPhi_lept",
    "RecoX_B1_dR_lept",
    "RecoX_B2_dR_lept",
    "RecoX_X_btagAverage",


    ]
all_variables = list(set( [v for key in variables for v in variables[key] ] ))

