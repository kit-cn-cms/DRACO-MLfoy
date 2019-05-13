variables = {}
variables["4j_ge3t"] = [
    "Jet_DeepJetCSV[3]",
    "CSV[1]",
    "N_BTagsT",
    "MVA_dev_from_avg_disc_btags",
    "MVA_sphericity_jets",
    "MVA_blr",
    #"memDBp",
    "MVA_transverse_sphericity_jets",
    "Evt_CSV_Min_Tagged",
    "Evt_CSV_Average_Tagged",
    "MVA_Evt_CSV_Average",
    "MVA_blr_transformed",
    "MVA_HT_tag",
    "MVA_Evt_M2_TaggedJetsAverage",
    ]



variables["5j_ge3t"] = [
    "MVA_max_dR_jj",
    "CSV[1]",
    "N_BTagsT",
    "Evt_Dr_MinDeltaRLeptonTaggedJet",
    "MVA_sphericity_jets",
    "Jet_Pt[2]",
    "MVA_closest_tagged_dijet_mass",
    "MVA_sphericity_tags",
    "Jet_Pt[0]",
    "MVA_HT_tag",
    "Evt_Dr_TaggedJetsAverage",
    #"memDBp",
    "Evt_CSV_Average_Tagged",
    "Evt_CSV_Min_Tagged",
    "MVA_Evt_CSV_Average",
    ]



variables["ge6j_ge3t"] = [
    "Evt_Dr_MinDeltaRTaggedJets",
    "Evt_Dr_TaggedJetsAverage",
    "Evt_M_MinDeltaRLeptonTaggedJet",
    "CSV[1]",
    "MVA_dev_from_avg_disc_btags",
    "MVA_blr",
    "MVA_Evt_CSV_Average",
    "Evt_CSV_Average_Tagged",
    "MVA_HT_tag",
    "Evt_M_JetsAverage",
    #"memDBp",
    "MVA_blr_transformed",
    ]



all_variables = list(set( [v for key in variables for v in variables[key] ] ))
