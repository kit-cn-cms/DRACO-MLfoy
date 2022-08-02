variables = {}
variables["ge2j_ge1t_ge0A"] = [
    "Pair_Mu_Pt_corr",
    "Pair_Tau_Pt_corr",
    "Pair_M_corr",
    "Pair_Pt_corr",
    "N_taggedJets_corr_nom",
    "nontaggedJets_Pt_corr_nom[0]",
    "N_nontaggedJets_corr_nom",
    "nonbJetPair_dR_corr_nom",
    "nonbJetPair_M_corr_nom",
    "nonbJetPair_Pt_corr_nom",
    "taggedJets_Pt_corr_nom[0]",
    "taggedJets_Pt_corr_nom[1]",
    "nontaggedJets_Pt_corr_nom[1]",
    "bJetPair_bTagDisSorted_M_corr_nom",
    "bJetPair_bTagDisSorted_Pt_corr_nom",
    "System_bTagDisSorted_M_corr_nom",
    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))