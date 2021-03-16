variables = {}

variables["ge4j_ge4t"] = [
    "Reco_ttbar_toplep_m",
    "Reco_ttbar_bestJABDToutput",
    "Reco_ttH_toplep_m",
    "Reco_ttH_bestJABDToutput",
    "Reco_tHq_bestJABDToutput",
    "Reco_JABDT_ttbar_Jet_CSV_whaddau2",
    "Reco_JABDT_ttH_Jet_CSV_hdau2",
    "Reco_JABDT_tHq_abs_ljet_eta",
    "Reco_JABDT_tHq_Jet_CSV_hdau1",
    "Reco_JABDT_tHW_top_pt__P__h_pt__P__wb_pt__DIV__Evt_HT__P__Evt_Pt_MET__P__Lep_Pt",
    "Reco_JABDT_tHW_Jet_CSV_btop",
    "N_Jets",
    "Evt_blr_transformed",
    "Evt_Pt_minDrTaggedJets",
    "Evt_Pt_JetsAverage",
    "Evt_M_minDrLepTag",
    "Evt_M_TaggedJetsAverage",
    "Evt_M_JetsAverage",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_Deta_JetsAverage",
    "Evt_CSV_avg_tagged",
    "Evt_CSV_avg",
    "CSV[2]",
]
variables["ge4j_3t"] = [
    # "Reco_ttbar_whad_m",
    # "Reco_ttbar_toplep_m",
    "Reco_ttbar_bestJABDToutput",
    "Reco_ttH_bestJABDToutput",
    "Reco_tHq_bestJABDToutput",
    # "Reco_tHW_whad_dr",
    # "Reco_tHW_bestJABDToutput",
    # "Reco_JABDT_ttbar_Jet_CSV_whaddau1",
    # "Reco_JABDT_ttbar_Jet_CSV_btoplep",
    # "Reco_JABDT_ttH_Jet_CSV_btoplep",
    # "Reco_JABDT_tHW_Jet_CSV_whaddau1",
    "Reco_JABDT_tHW_Jet_CSV_btop",
    # "N_Jets",
    # "Evt_h1",
    # "Evt_blr_transformed",
    # "Evt_Pt_minDrTaggedJets",
    "Evt_M_Total",
    # "Evt_M_TaggedJetsAverage",
    "Evt_M_JetsAverage",
    # "Evt_JetPt_over_JetE",
    # "Evt_HT_tags",
    # "Evt_Deta_TaggedJetsAverage",
    # "Evt_Deta_JetsAverage",
    "Evt_CSV_dev",
    # "Evt_CSV_avg_tagged",
    "Evt_CSV_avg",
]
kNN_vars = """
    Evt_HT_jets

    Jet_Pt[0]
""".split()

for key in variables:
    variables[key] = list(set(kNN_vars+variables[key]))

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
