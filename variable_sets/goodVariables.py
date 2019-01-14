print("using top 10 variables - number seven will shock you")
variables = {}
variables["4j_ge3t"] = [
    "N_BTagsT",
    "Evt_M2_TaggedJetsAverage",
    "Evt_Dr_MinDeltaRTaggedJets",
    #"Evt_M_JetsAverage",
    "BDT_common5_input_HT_tag",
    "Jet_Pt[3]",
    "BDT_common5_input_transverse_sphericity_jets",
    "Evt_blr_ETH_transformed",
    "Evt_CSV_Min_Tagged",
    "memDBp",
    ]


variables["5j_ge3t"] = [
    #"Evt_CSV_Min_Tagged",
    "BDT_common5_input_aplanarity_tags",
    "memDBp",
    "Evt_blr_ETH_transformed",
    "Evt_Dr_MinDeltaRTaggedJets",
    "BDT_common5_input_dev_from_avg_disc_btags",
    "BDT_common5_input_closest_tagged_dijet_mass",
    #"CSV[0]",
    "Jet_Pt[2]",
    "BDT_common5_input_sphericity_jets",
    ]


variables["ge6j_ge3t"] = [
    #"Evt_M_MinDeltaRLeptonTaggedJet",
    "Evt_CSV_Min_Tagged",
    #"BDT_common5_input_closest_tagged_dijet_mass",
    "Evt_M2_TaggedJetsAverage",
    "BDT_common5_input_HT_tag",
    #"Evt_CSV_Average_Tagged",
    #"Evt_blr_ETH_transformed",
    "Evt_Dr_TaggedJetsAverage",
    #"BDT_common5_input_max_dR_bb",
    "memDBp",
    #"BDT_common5_input_transverse_sphericity_jets",
    ]


all_variables = set( [v for key in variables for v in variables[key] ] )


