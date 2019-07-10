variables = {}

variables["ge6j_ge3t"] = [
    "CSV[0]", # highest b-tagger output
    "CSV[1]",
    "CSV[2]",
    "Jet_Pt[0]",
    "Jet_Pt[1]",
    "Jet_Pt[2]",

    "Jet_CSV[0]", # b-tagger output of hardest jet
    "Jet_CSV[1]",
    "Jet_CSV[2]",

    "N_BTagsT",

    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))


#######################
#additional variables #
#######################

# "Evt_JetPtOverJetE",
# "Evt_Dr_MinDeltaRTaggedJets",
# "Evt_Dr_TaggedJetsAverage",

# "BDT_common5_input_closest_tagged_dijet_mass",
# "Evt_M_MinDeltaRLeptonTaggedJet",

# "Evt_Deta_TaggedJetsAverage",
# "Evt_CSV_Average",
# "Evt_CSV_Average_Tagged",
# "Evt_CSV_Min_Tagged",
# "BDT_common5_input_HT_tag",
# "Evt_M_JetsAverage",
