variables = {}
variables["ge4j_ge3t"] = [
    "ttbarReco_ttbar_Pt",
    "ttbarReco_ttbar_Eta",
    "ttbarReco_ttbar_M",
    "ttbarReco_ttbar_E",
    "ttbarReco_ttbar_openingAngle", 
    "ttbarReco_LepTop_Pt",  
    "ttbarReco_LepTop_Eta", 
    "ttbarReco_LepTop_M",
    "ttbarReco_LepTop_E",
    "ttbarReco_LepTop_Phi",
    "ttbarReco_HadTop_Pt",
    "ttbarReco_HadTop_Eta",
    "ttbarReco_HadTop_M",
    "ttbarReco_HadTop_E",
    "ttbarReco_HadTop_Phi",
    "ttbarReco_ttbar_dPhi",
    "ttbarReco_ttbar_dEta",
    "ttbarReco_ttbar_dPt",
    "ttbarReco_ttbar_dR",
    "ttbarReco_ttbar_dKin",
    "N_Jets",
    "N_BTagsM",
    ] 
all_variables = list(set( [v for key in variables for v in variables[key] ] ))


