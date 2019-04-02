variables = {}
variables["ge4j_ge4t"] = [
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



all_variables = set( [v for key in variables for v in variables[key] ] )
