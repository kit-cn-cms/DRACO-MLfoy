variables = {}
variables["ge4j_ge3t"] = [
    "N_Jets",
    "N_BTagsM",
    "isTagged",
    "jetEnergy",
    "jetPt",
    "jetMass",
    "jetPhi",
    "jetEta",
    "jetCharge",
    "deepJetValue",
    "deepJet_CvsL",
    "deepJet_CvsB",
    "leptonPt",
    "leptonEta",
    "leptonPhi",
    "leptonE",
    ] 
all_variables = list(set( [v for key in variables for v in variables[key] ] ))


