variables = {}

variables["ge4j_ge4t"] = [
    "Evt_CSV_avg_tagged",
    "Evt_CSV_avg",
    "Evt_M_minDrLepTag",
    "Evt_Pt_JetsAverage",
    "Evt_Pt_minDrTaggedJets",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_Deta_JetsAverage",
    "N_Jets",
    "Reco_JABDT_ttbar_Jet_CSV_whaddau2",
    "Reco_ttbar_toplep_m",
    "Reco_tHq_bestJABDToutput",
    "Reco_JABDT_tHq_abs_ljet_eta",
    "Reco_JABDT_tHq_Jet_CSV_hdau1",
    "Reco_JABDT_tHW_Jet_CSV_btop",
    # "memDBp",

    # failed 2D validation
    # "Evt_M_JetsAverage",
    # "Reco_ttbar_bestJABDToutput",
    # "Reco_ttH_toplep_m",
    # "Reco_ttH_bestJABDToutput",
    # "Evt_blr_transformed",
    # "Reco_JABDT_ttH_Jet_CSV_hdau2",
    # "Reco_JABDT_ttbar_Jet_CSV_whaddau1",

    # failed 2D validation with lnN
    # "Reco_tHW_bestJABDToutput",
    # "Reco_JABDT_tHW_top_pt__P__h_pt__P__wb_pt__DIV__Evt_HT__P__Evt_Pt_MET__P__Lep_Pt",
    # "Reco_JABDT_tHW_Jet_CSV_whaddau1",
    # "CSV[2]",
    # "Evt_M_TaggedJetsAverage",

    ]

variables["ge4j_3t"] = [
    # "memDBp",
    "Evt_Deta_JetsAverage",
    "Evt_M_TaggedJetsAverage", 
    "Evt_CSV_avg", 
    "Evt_blr_transformed",
    "N_Jets",
    "Reco_ttbar_toplep_m",
    "Reco_ttbar_whad_m",
    "Reco_JABDT_ttH_Jet_CSV_btoplep",
    "Reco_JABDT_tHW_log_wb_m",

    # failed 2D validation
    # "Evt_CSV_dev",
    # "Evt_M_Total",
    # "Evt_CSV_avg_tagged",
    # "Reco_ttH_bestJABDToutput",
    # "Reco_tHq_bestJABDToutput",
    # "Reco_tHW_whad_dr",
    # "Reco_tHW_bestJABDToutput",
    # "Reco_JABDT_tHW_Jet_CSV_btop",
    # "Reco_JABDT_tHW_Jet_CSV_whaddau1",
    # "Reco_ttbar_bestJABDToutput",  
    # "Reco_JABDT_ttbar_Jet_CSV_btophad",
    # "Reco_JABDT_ttbar_Jet_CSV_btoplep",
   
    # failed 2D validation with lnN
    # "Reco_JABDT_ttbar_Jet_CSV_whaddau1",
    # "Evt_JetPt_over_JetE",
    # "Evt_h1",
    # "Evt_Pt_minDrTaggedJets",
    # "Evt_HT_tags",
    # "Evt_Deta_TaggedJetsAverage",
    # "Evt_M_JetsAverage"


    ]
kNN_vars = """
    Evt_Deta_maxDetaJetJet
    Evt_Deta_maxDetaTagTag
    Evt_Dr_TaggedJetsAverage
    Evt_M3
    Evt_HT_jets
    Evt_HT_tags

    Jet_Pt[0]
    Jet_Pt[1]
    Jet_Pt[2]
    Jet_Pt[3]
    Evt_M2_minDrTaggedJets

    Evt_TaggedJetPt_over_TaggedJetE
    
    Evt_Pt_TaggedJetsAverage

""".split()
bu = """
Evt_Pt_minDrTaggedJets
""".split()

for key in variables:
    variables[key] += kNN_vars

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
