variables = {}

variables["ge4j_ge4t"] = [
    'memDBp',
    "Reco_tHq_h_pt",
    "Reco_JABDT_tHq_log_h_pt",
    "Reco_JABDT_tHq_log_min_hdau1_pt_hdau2_pt",
    "Reco_tHW_h_pt",
    "Reco_tHW_h_dr",
    "Reco_JABDT_tHW_log_h_pt",
    "Reco_ttH_h_dr",
    "Reco_ttH_h_m",
    "Reco_JABDT_ttH_log_h_pt",
    "Reco_JABDT_ttH_log_toplep_m",


    "Evt_Pt_TaggedJetsAverage",
    "Evt_Dr_TaggedJetsAverage",
    "Evt_M_TaggedJetsAverage",
    "Evt_Pt_minDrTaggedJets",
    "Evt_Pt_JetsAverage",

    # didn't pass validation
    # "Reco_tHq_h_dr",
    ]


variables["ge4j_3t"] = [
    'memDBp',
    "Reco_JABDT_tHW_log_h_m",
    "Reco_tHq_h_pt",
    "Reco_JABDT_tHq_log_h_m",
    "Reco_JABDT_tHq_log_h_pt",
    "Reco_JABDT_tHq_log_min_hdau1_pt_hdau2_pt",
    "Reco_tHW_h_pt",
    # "Reco_tHW_h_dr", -> failed new 2D validation in 2018
    "Reco_JABDT_tHW_log_h_pt",
    "Reco_ttH_h_dr",

    "Evt_Pt_JetsAverage",
    "Evt_Pt_TaggedJetsAverage",
    "Evt_Dr_TaggedJetsAverage",

    # "TaggedJet_Pt[0]", -> failed new 2D validation in 2018

    # didn't pass validation
    # "Evt_HT",
    # "Evt_HT_jets",
    # "Evt_HT_tags",
    # "Evt_M_TaggedJetsAverage",
    # "Reco_tHW_bestJABDToutput",
    # "Reco_tHq_h_dr",
    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
