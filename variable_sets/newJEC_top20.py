variables = {}
variables["4j_ge3t"] = [
    "N_BTagsT",
    "BDT_common5_input_transverse_sphericity_tags",
    "Evt_Dr_MinDeltaRLeptonJet",
    "Jet_Pt[0]",
    "BDT_common5_input_h2",
    "BDT_common5_input_dev_from_avg_disc_btags",
    "BDT_common5_input_closest_tagged_dijet_mass",
    "BDT_common5_input_sphericity_jets",
    "Evt_blr_ETH",
    "memDBp",
    "Evt_Dr_MinDeltaRTaggedJets",
    "BDT_common5_input_transverse_sphericity_jets",
    "Evt_CSV_Min_Tagged",
    "Evt_CSV_Average_Tagged",
    "Evt_CSV_Average",
    "Evt_blr_ETH_transformed",
    "Evt_CSV_Min",
    "BDT_common5_input_HT_tag",
    "Evt_M_JetsAverage",
    "Evt_M2_TaggedJetsAverage",
    ]



variables["5j_ge3t"] = [
    "BDT_common5_input_h2",
    "N_BTagsT",
    "Evt_Dr_MinDeltaRLeptonTaggedJet",
    "BDT_common5_input_sphericity_jets",
    "Jet_Pt[2]",
    "Evt_Dr_MinDeltaRTaggedJets",
    "BDT_common5_input_closest_tagged_dijet_mass",
    "BDT_common5_input_transverse_sphericity_tags",
    "BDT_common5_input_sphericity_tags",
    "Evt_blr_ETH",
    "Jet_Pt[0]",
    "BDT_common5_input_HT_tag",
    "Evt_Dr_TaggedJetsAverage",
    "Evt_HT",
    "memDBp",
    "Evt_CSV_Average_Tagged",
    "Evt_CSV_Min_Tagged",
    "Evt_CSV_Average",
    "Evt_M_JetsAverage",
    "Evt_blr_ETH_transformed",
    ]



variables["ge6j_ge3t"] = [
    "BDT_common5_input_closest_tagged_dijet_mass",
    "BDT_common5_input_pt_all_jets_over_E_all_jets_tags",
    "BDT_common5_input_transverse_sphericity_jets",
    "BDT_common5_input_transverse_sphericity_tags",
    "Evt_M_MinDeltaRLeptonTaggedJet",
    "CSV[1]",
    "BDT_common5_input_sphericity_jets",
    "BDT_common5_input_dev_from_avg_disc_btags",
    "N_BTagsT",
    "BDT_common5_input_sphericity_tags",
    "Evt_blr_ETH",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_M2_TaggedJetsAverage",
    "Evt_CSV_Average",
    "Evt_CSV_Average_Tagged",
    "Evt_CSV_Min_Tagged",
    "BDT_common5_input_HT_tag",
    "Evt_M_JetsAverage",
    "memDBp",
    "Evt_blr_ETH_transformed",
    ]



all_variables = set( [v for key in variables for v in variables[key] ] )
