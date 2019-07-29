variables = {}

variables["ge3j_ge2t"] = [
    "N_jets",
    "N_btags",
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

variables["jets"]= [

#    "pT_jet_jet_min_deltaR",
#    "avgDeltaR_jet_jet",
#    "avgDeltaR_jet_tag",
#    "C_jet",
#    "C_tag",
#    "R1_tag",
#    "mass_higgsLikeDijet",
#    "minDeltaR_jet_jet",
#    "R1_jet",
#    "twist_jet_jet_max_mass",
#    "mass_jet_jet_min_deltaR",
#    "centrality_tags",
#    "mass_higgsLikeDijet2",
#    "circularity_jet",
#    "centrality_jets_leps",
#    "R4_tag",
#    "R2_jet",
#    "twist_jet_tag_max_mass",
#    "aplanarity_jet",
#    "twist_tag_tag_min_deltaR",
#    "R4_jet",
#    "H0_tag",
#    "H3_tag",

    "jet1_pt",
    "jet1_eta",
    "jet1_phi",
    "jet1_M",
    "jet1_btag",

    "jet2_pt",
    "jet2_eta",
    "jet2_phi",
    "jet2_M",
    "jet2_btag",

    "jet3_pt",
    "jet3_eta",
    "jet3_phi",
    "jet3_M",
    "jet3_btag",

    "jet4_pt",
    "jet4_eta",
    "jet4_phi",
    "jet4_M",
    "jet4_btag",

    "jet5_pt",
    "jet5_eta",
    "jet5_phi",
    "jet5_M",
    "jet5_btag",

    "jet6_pt",
    "jet6_eta",
    "jet6_phi",
    "jet6_M",
    "jet6_btag",

#    "bjet1_pt",
#    "bjet1_eta",
#    "bjet2_pt",
#    "bjet2_eta",
#    "bjet3_pt",
#    "bjet3_eta",
#    "bjet1_btag",
#    "bjet2_btag",
#    "bjet3_btag",
]

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

variables["ge4j_3t"] = [
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

variables["ge4j_2t"] = [
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

variables["3j_3t"] = [
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

variables["3j_2t"] = [
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

variables["allvar"] = [
     "lepton1_pt",
     "lepton2_pt",

     "lepton1_eta",
     "lepton2_eta",

      "jet1_pt",
      "jet2_pt",
      "jet3_pt",
      "jet4_pt",
      "jet5_pt",
      "jet6_pt",
      "bjet1_pt",
      "bjet2_pt",
      "bjet3_pt",
      "bjet4_pt",

      "jet1_eta",
      "jet2_eta",
      "jet3_eta",
      "jet4_eta",
      "jet5_eta",
      "jet6_eta",
      "bjet1_eta",
      "bjet2_eta",
      "bjet3_eta",
      "bjet4_eta",

      "jet1_btag",
      "jet2_btag",
      "jet3_btag",
      "jet4_btag",
      "jet5_btag",
      "jet6_btag",
      "bjet1_btag",
      "bjet2_btag",
      "bjet3_btag",
      "bjet4_btag",

     "dilepton_pt",
     "dilepton_eta",
     "dilepton_M",

     "met_pt",
     "met_phi",

     "MEM",

     "multiplicity_higgsLikeDijet15",

     "btagDiscriminatorAverage_tagged",
     "btagDiscriminatorAverage_untagged",

     "sphericity_jet",
     "sphericity_tag" ,

     "aplanarity_jet",
     "aplanarity_tag",

     "circularity_jet",
     "circularity_tag",

     "isotropy_jet",
     "isotropy_tag",

     "C_jet",
     "C_tag",

     "D_jet",
     "D_tag",

     "transSphericity_jet",
     "transSphericity_tag",

     "H0_jet",
     "H1_jet",
     "H2_jet",
     "H3_jet",
     "H4_jet",

     "H0_tag",
     "H1_tag",
     "H2_tag",
     "H3_tag",
     "H4_tag",

     "R1_jet",
     "R2_jet",
     "R3_jet",
     "R4_jet",

     "R1_tag",
     "R2_tag",
     "R3_tag",
     "R4_tag",

     "avgDeltaR_jet_jet",
     "avgDeltaR_jet_tag",
     "avgDeltaR_tag_tag",

     "centrality_jets_leps",
     "centrality_tags",

     "circularity_jet",
     "circularity_tag",

     "mass_higgsLikeDijet",
     "mass_higgsLikeDijet2",

     "isotropy_jet",
     "isotropy_tag",

     "mass_jet_jet_min_deltaR",
     "mass_jet_tag_min_deltaR",
     "mass_tag_tag_min_deltaR",
     "mass_tag_tag_max_mass",
     "mass_jet_jet_jet_max_pT",
     "mass_jet_tag_tag_max_pT",

     "maxDeltaEta_jet_jet",
     "maxDeltaEta_tag_tag",

     "median_mass_jet_jet",

     "minDeltaR_jet_jet",
     "minDeltaR_tag_tag",

     "pT_jet_jet_min_deltaR",
     "pT_jet_tag_min_deltaR",
     "pT_tag_tag_min_deltaR",

     "ptSum_jets_leptons",

     "HT_jets",
     "HT_tags",

     "twist_jet_jet_max_mass",
     "twist_jet_tag_max_mass",
     "twist_tag_tag_max_mass",
     "twist_tag_tag_min_deltaR"
]

variables["ge4j_ge3t"] = list(set(variables["ge4j_ge4t"][:] + variables["ge4j_3t"][:]))
#variables["ge4j_ge3t"] = list(set(variables["ge4j_ge4t"][:] + variables["ge4j_3t"][:] + variables['jets']))

#!! variables["ge4j_ge3t"] = ['jet1_pt', 'jet2_pt'] #!!!!

all_variables = set( [v for key in variables for v in variables[key] ] )
