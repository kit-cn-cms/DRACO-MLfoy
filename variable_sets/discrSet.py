variables = {}
variables["ge4j_ge3t"] = [
    "DNNdiscr_ttHbb",
    "DNNdiscr_ttb",
    "DNNdiscr_ttcc",
    "DNNdiscr_ttlf",

    "DNNdiscr_Hbb_vs_b",
    "DNNdiscr_Hbb_vs_cc",
    "DNNdiscr_Hbb_vs_lf",
    "DNNdiscr_b_vs_cc",
    "DNNdiscr_b_vs_lf",
    "DNNdiscr_cc_vs_lf",
    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))

