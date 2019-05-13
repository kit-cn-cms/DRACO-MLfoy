all_variables = {
        "MEM",
        "HT_tags",
        "ptSum_jets_leptons",
        "btagDiscriminatorAverage_untagged",
        "btagDiscriminatorAverage_tagged",
        "mass_tag_tag_min_deltaR",
        "mass_jet_tag_min_deltaR",
        "mass_tag_tag_max_mass",
        "mass_jet_jet_jet_max_pT",
        "pT_tag_tag_min_deltaR",
        "avgDeltaR_tag_tag",
        "minDeltaR_tag_tag",
        "maxDeltaEta_jet_jet",
        "maxDeltaEta_tag_tag",
        "H0_jet",
        "twist_tag_tag_max_mass",
        "multiplicity_higgsLikeDijet15",
        "pT_jet_jet_min_deltaR",
        "avgDeltaR_jet_jet",
        "avgDeltaR_jet_tag",
        "C_jet",
        "C_tag",
        "R1_tag",
        "mass_higgsLikeDijet",
        "minDeltaR_jet_jet",
        "R1_jet",
        "twist_jet_jet_max_mass",
        "mass_jet_jet_min_deltaR"
}


undefined_variables = [
    #"dank_MEM",
    "max_CSV_tags", #already covered
    #"sphericityT_jets",
    #"sphericityT_tags",
    #"HT_tag",
    #"aplanarity_tags",
    #"aplanarity_jets",
    #"sphericity_jets",
    #"sphericity_tags",
    #"max_dR_bb",
    #"max_dR_jj",
    #"centrality_tags",
    ]


# categories

variables_3j_2t = [
    "HT_tags",
    "ptSum_jets_leptons",
    "btagDiscriminatorAverage_untagged",
    "mass_jet_jet_min_deltaR",
    "mass_tag_tag_min_deltaR",
    "mass_jet_jet_jet_max_pT",
    "pT_tag_tag_min_deltaR",
    "avgDeltaR_jet_jet",
    "minDeltaR_jet_jet",
    "maxDeltaEta_jet_jet",
    "C_jet",
    "C_tag",
    "H0_jet"
    ]

variables_3j_3t = [
    "ptSum_jets_leptons",
    "btagDiscriminatorAverage_tagged",
    "mass_higgsLikeDijet",
    "mass_jet_jet_min_deltaR",
    "mass_tag_tag_max_mass",
    "pT_jet_jet_min_deltaR",
    "avgDeltaR_jet_jet",
    "minDeltaR_jet_jet",
    "maxDeltaEta_jet_jet",
    "C_jet",
    "R1_jet",
    "twist_jet_jet_max_mass"
    ]

variables_ge4j_2t = [
    "btagDiscriminatorAverage_untagged",
    "mass_tag_tag_min_deltaR",
    "mass_jet_tag_min_deltaR",
    "pT_tag_tag_min_deltaR",
    "pT_jet_jet_min_deltaR",
    "avgDeltaR_jet_jet",
    "avgDeltaR_jet_tag",
    "maxDeltaEta_jet_jet",
    "C_jet",
    "C_tag",
    "R1_tag",
    "multiplicity_higgsLikeDijet15"
    ]

variables_ge4j_3t = [
    "MEM",
    "ptSum_jets_leptons",
    "btagDiscriminatorAverage_untagged",
    "btagDiscriminatorAverage_tagged",
    "mass_tag_tag_min_deltaR",
    "mass_tag_tag_max_mass",
    "pT_tag_tag_min_deltaR",
    "avgDeltaR_tag_tag",
    "minDeltaR_tag_tag",
    "maxDeltaEta_tag_tag",
    "H0_jet",
    "twist_tag_tag_max_mass"
    ]

variables_ge4j_ge4t = [
    "MEM",
    "HT_tags",
    "btagDiscriminatorAverage_tagged",
    "mass_tag_tag_min_deltaR",
    "mass_jet_tag_min_deltaR",
    "mass_jet_jet_jet_max_pT",
    "pT_tag_tag_min_deltaR",
    "avgDeltaR_tag_tag",
    "minDeltaR_tag_tag",
    "maxDeltaEta_jet_jet",
    "maxDeltaEta_tag_tag",
    "multiplicity_higgsLikeDijet15"
    ]

variables_ge4j_ge3t = [
    "MEM",
    "HT_tags",
    "ptSum_jets_leptons",
    "btagDiscriminatorAverage_untagged",
    "btagDiscriminatorAverage_tagged",
    "mass_tag_tag_min_deltaR",
    "mass_jet_tag_min_deltaR",
    "mass_tag_tag_max_mass",
    "mass_jet_jet_jet_max_pT",
    "pT_tag_tag_min_deltaR",
    "avgDeltaR_tag_tag",
    "minDeltaR_tag_tag",
    "maxDeltaEta_jet_jet",
    "maxDeltaEta_tag_tag",
    "H0_jet",
    "twist_tag_tag_max_mass",
    "multiplicity_higgsLikeDijet15"
    ]


all_variables_list = [var for var in all_variables if not var in undefined_variables]
print(all_variables_list)
