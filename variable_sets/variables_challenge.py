variables = {}

variables["ge6j_ge3t"] = [
    "CSV[0]",
    "CSV[1]",
    "CSV[2]",
    "Jet_Pt[0]",
    "Jet_Pt[1]",
    "Jet_Pt[2]",

    "Jet_CSV[0]",
    "Jet_CSV[1]",
    "Jet_CSV[2]",

    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))


#######################
#additional variables #
#######################

# "memDBp",

# "Evt_JetPtOverJetE",
# "Evt_Dr_MinDeltaRTaggedJets",
# "Evt_Dr_TaggedJetsAverage",

# "BDT_common5_input_closest_tagged_dijet_mass",
# "Evt_M_MinDeltaRLeptonTaggedJet",

# "BDT_common5_input_sphericity_jets",
# "BDT_common5_input_dev_from_avg_disc_btags",
# "N_BTagsT",
# "Evt_blr_ETH",
# "Evt_Deta_TaggedJetsAverage",
# "Evt_M2_TaggedJetsAverage",
# "Evt_CSV_Average",
# "Evt_CSV_Average_Tagged",
# "Evt_CSV_Min_Tagged",
# "BDT_common5_input_HT_tag",
# "Evt_M_JetsAverage",
# "Evt_blr_ETH_transformed",
