variables = {}
variables["ge4j_ge3t"] = [
<<<<<<< HEAD

    # X boson stuff
=======
    
    # recoX stuff
>>>>>>> c85827ec3c315ff52755b8a0cf32693c99568c89
    "RecoX_jet1_Eta",
    "RecoX_jet2_Eta",
    "RecoX_jet1_Phi",
    "RecoX_jet2_Phi",
    "RecoX_jet1_btagValue",
    "RecoX_jet2_btagValue",
    "RecoX_jet1_M",
    "RecoX_jet2_M",
    "RecoX_jet1_E",
    "RecoX_jet2_E",
    "RecoX_jet1_Pt",
    "RecoX_jet2_Pt",
    "RecoX_jet1_idx",
    "RecoX_jet2_idx",
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
<<<<<<< HEAD

    #"RecoX_X_dKin",
    "RecoX_X_btagAverage",

    # lepton variables
=======
    #"RecoX_X_dKin",


>>>>>>> c85827ec3c315ff52755b8a0cf32693c99568c89
    "TightLepton_E_0",
    "TightLepton_Eta_0",
    "TightLepton_M_0",
    "TightLepton_Phi_0",
    "TightLepton_Pt_0",

<<<<<<< HEAD
    # boson-lepton variables
=======
>>>>>>> c85827ec3c315ff52755b8a0cf32693c99568c89
    "RecoX_jet1_dEta_lept",
    "RecoX_jet2_dEta_lept",
    "RecoX_jet1_dPhi_lept",
    "RecoX_jet2_dPhi_lept",
    "RecoX_jet1_dR_lept",
    "RecoX_jet2_dR_lept",
<<<<<<< HEAD
=======
    "RecoX_X_btagAverage",

>>>>>>> c85827ec3c315ff52755b8a0cf32693c99568c89
    "RecoX_X_dEta_lept",
    "RecoX_X_dPhi_lept",
    "RecoX_X_dR_lept",

<<<<<<< HEAD
    "RecoX_jet1_CvsL_deepJet",
    "RecoX_jet1_CvsB_deepJet",
    "RecoX_jet2_CvsL_deepJet",
    "RecoX_jet2_CvsB_deepJet",

    ]

print variables
=======
    #"RecoX_jet1_CvsL_deepJet",
    #"RecoX_jet1_CvsB_deepJet",
    #"RecoX_jet2_CvsL_deepJet",
    #"RecoX_jet2_CvsB_deepJet",

    ]
>>>>>>> c85827ec3c315ff52755b8a0cf32693c99568c89
all_variables = list(set( [v for key in variables for v in variables[key] ] ))

