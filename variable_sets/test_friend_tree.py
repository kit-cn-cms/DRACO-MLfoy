variables = {}
variables["ge4j_ge3t"] = [
    "Jet_Pt[0]",
    "Jet_Eta[0]",
    "N_Jets",
    "N_BTagsM",
    "RecoDNN_Z.RecoDNN_Z_Z_M",
    "RecoDNN_Z.RecoDNN_Z_dR"
    ]


all_variables = list(set( [v for key in variables for v in variables[key] ] ))


