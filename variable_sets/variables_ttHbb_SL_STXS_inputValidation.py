variables = {}

variables["ge4j_ge4t"] = [
    "Evt_Pt_TaggedJetsAverage",
    "Reco_tHq_h_dr",
    "Reco_ttH_h_pt",
    "Reco_tHW_h_dr",
    "Reco_tHW_h_pt",
    "Reco_JABDT_tHW_log_h_m",
    "Reco_ttH_hdau_m1",
    "Reco_tHq_bestJABDToutput",
    "Evt_HT_wo_MET",
    "TaggedJet_Pt[1]",
    "Evt_HT",
    "Reco_JABDT_tHW_log_wb_m",
    "Evt_Dr_TaggedJetsAverage",
    "Evt_Pt_JetsAverage",
    "Reco_JABDT_tHW_Jet_CSV_hdau2",
    "Reco_JABDT_tHq_log_min_hdau1_pt_hdau2_pt",
    "Reco_JABDT_tHW_top_pt__P__h_pt__P__wb_pt__DIV__Evt_HT__P__Evt_Pt_MET__P__Lep_Pt",
    "Reco_tHq_hdau_pt1",
    "Evt_Pt_minDrTaggedJets",
    "Reco_ttH_h_dr",
    "Evt_Deta_TaggedJetsAverage",
    ]


variables["ge4j_3t"] = [
    "Reco_ttH_h_pt",
    "Reco_JABDT_ttH_log_h_m",
    "Reco_tHq_h_dr",
    "Reco_JABDT_tHq_log_h_pt",
    "TaggedJet_Pt[0]",
    "Evt_Pt_minDrTaggedJets",
    "Reco_tHW_h_pt",
    "Evt_Dr_minDrTaggedJets",
    "Evt_Dr_TaggedJetsAverage",
    "Reco_ttH_hdau_pt2",
    "Reco_ttH_hdau_pt1",
    "Evt_Pt_TaggedJetsAverage",
    "Evt_HT_tags",
    "Evt_M_JetsAverage",
    "Evt_Pt_JetsAverage",
    "Reco_JABDT_ttH_tophad_pt__P__toplep_pt__P__h_pt__DIV__Evt_HT__P__Evt_Pt_MET__P__Lep_Pt",
    "Evt_HT",
    "TaggedJet_M[0]",
    "Reco_tHq_hdau_pt1",
    "Reco_tHW_hdau_pt1",
    "Reco_tHq_h_pt",
    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
