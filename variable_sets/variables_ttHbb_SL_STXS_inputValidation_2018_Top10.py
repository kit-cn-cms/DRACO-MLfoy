variables = {}

variables["ge4j_ge4t"] = [
    "Reco_tHq_h_pt",
    "Reco_tHW_h_pt",
    "Evt_Pt_TaggedJetsAverage",
    "Reco_JABDT_tHq_log_min_hdau1_pt_hdau2_pt",
    "Reco_JABDT_ttH_log_toplep_m",
    "Reco_tHq_h_dr",
    "Evt_Pt_minDrTaggedJets",
    "Evt_Dr_TaggedJetsAverage",
    "Reco_ttH_h_dr",
    "Reco_JABDT_ttH_log_h_pt",
    "Evt_M_TaggedJetsAverage",
    "Reco_ttH_h_m",
    "Evt_Pt_JetsAverage",
    "Reco_JABDT_tHW_log_h_pt",
    "Reco_tHW_h_dr",
    "Reco_JABDT_tHq_log_h_pt",
    ]


variables["ge4j_3t"] = [
    "Reco_JABDT_tHW_log_h_m",
    "Reco_tHq_h_pt",
    "Reco_tHW_h_pt",
    "Evt_Pt_TaggedJetsAverage",
    "Evt_HT_jets",
    "Reco_tHW_bestJABDToutput",
    "Evt_Dr_TaggedJetsAverage",
    "Evt_M_TaggedJetsAverage",
    "Reco_ttH_h_dr",
    "Reco_tHq_h_dr",
    "Reco_JABDT_tHq_log_min_hdau1_pt_hdau2_pt",
    "Evt_HT_tags",
    "Evt_HT",
    "TaggedJet_Pt[0]",
    "Reco_JABDT_tHq_log_h_pt",
    "Evt_Pt_JetsAverage",
    "Reco_JABDT_tHW_log_h_pt",
    "Reco_tHW_h_dr",
    "Reco_JABDT_tHq_log_h_m",
    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
