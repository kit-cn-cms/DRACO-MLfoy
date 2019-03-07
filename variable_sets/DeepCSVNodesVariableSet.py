reco_variables = [
    'BDT_common5_input_HT_tag',
    'BDT_common5_input_aplanarity_jets',
    'BDT_common5_input_aplanarity_tags',
    'BDT_common5_input_closest_tagged_dijet_mass',
    'BDT_common5_input_dev_from_avg_disc_btags',
    'BDT_common5_input_h1',
    'BDT_common5_input_h2',
    'BDT_common5_input_h3',
    'BDT_common5_input_max_dR_bb',
    'BDT_common5_input_max_dR_jj',
    'BDT_common5_input_pt_all_jets_over_E_all_jets_tags',
    'BDT_common5_input_sphericity_jets',
    'BDT_common5_input_sphericity_tags',
    'BDT_common5_input_tagged_dijet_mass_closest_to_125',
    'BDT_common5_input_transverse_sphericity_jets',
    'BDT_common5_input_transverse_sphericity_tags',
    'Evt_Deta_TaggedJetsAverage',
    'Evt_Dr_MinDeltaRJets',
    'Evt_Dr_MinDeltaRLeptonJet',
    'Evt_Dr_MinDeltaRLeptonTaggedJet',
    'Evt_Dr_MinDeltaRTaggedJets',
    'Evt_Dr_TaggedJetsAverage',
    'Evt_HT',
    'Evt_JetPtOverJetE',
    'Evt_M2_TaggedJetsAverage',
    'Evt_M_JetsAverage',
    'Evt_M_MinDeltaRLeptonTaggedJet',
    'Evt_blr_ETH',
    'Evt_blr_ETH_transformed',
    'Jet_Eta[0]',
    'Jet_Eta[1]',
    'Jet_Eta[2]',
    'Jet_Eta[3]',
    'Jet_Pt[0]',
    'Jet_Pt[1]',
    'Jet_Pt[2]',
    'Jet_Pt[3]',
    'LooseLepton_Eta[0]',
    'LooseLepton_Pt[0]',
    'N_BTagsT',
    'memDBp',
    ]

csv_variables = [
    'CSV[0]',
    'CSV[1]',
    'Evt_CSV_Average',
    'Evt_CSV_Average_Tagged',
    'Evt_CSV_Min',
    'Evt_CSV_Min_Tagged',
    ]

deepcsv_comb = [
    'Jet_CSV[0]',
    'Jet_CSV[1]',
    'Jet_CSV[2]',
    'Jet_CSV[3]',
    ]

deepflav_comb = [
    'Jet_DeepFlavourCSV[0]',
    'Jet_DeepFlavourCSV[1]',
    'Jet_DeepFlavourCSV[2]',
    'Jet_DeepFlavourCSV[3]',
    ]

deepcsv_nodes = [
    'Jet_DeepCSV_b[0]',
    'Jet_DeepCSV_b[1]',
    'Jet_DeepCSV_b[2]',
    'Jet_DeepCSV_b[3]',
    'Jet_DeepCSV_bb[0]',
    'Jet_DeepCSV_bb[1]',
    'Jet_DeepCSV_bb[2]',
    'Jet_DeepCSV_bb[3]',
    'Jet_DeepCSV_c[0]',
    'Jet_DeepCSV_c[1]',
    'Jet_DeepCSV_c[2]',
    'Jet_DeepCSV_c[3]',
    'Jet_DeepCSV_udsg[0]',
    'Jet_DeepCSV_udsg[1]',
    'Jet_DeepCSV_udsg[2]',
    'Jet_DeepCSV_udsg[3]',
    ]

deepflav_nodes = [
    'Jet_DeepFlavour_b[0]',
    'Jet_DeepFlavour_b[1]',
    'Jet_DeepFlavour_b[2]',
    'Jet_DeepFlavour_b[3]',
    'Jet_DeepFlavour_bb[0]',
    'Jet_DeepFlavour_bb[1]',
    'Jet_DeepFlavour_bb[2]',
    'Jet_DeepFlavour_bb[3]',
    'Jet_DeepFlavour_c[0]',
    'Jet_DeepFlavour_c[1]',
    'Jet_DeepFlavour_c[2]',
    'Jet_DeepFlavour_c[3]',
    'Jet_DeepFlavour_g[0]',
    'Jet_DeepFlavour_g[1]',
    'Jet_DeepFlavour_g[2]',
    'Jet_DeepFlavour_g[3]',
    'Jet_DeepFlavour_lepb[0]',
    'Jet_DeepFlavour_lepb[1]',
    'Jet_DeepFlavour_lepb[2]',
    'Jet_DeepFlavour_lepb[3]',
    'Jet_DeepFlavour_uds[0]',
    'Jet_DeepFlavour_uds[1]',
    'Jet_DeepFlavour_uds[2]',
    'Jet_DeepFlavour_uds[3]',
    ]

current_set = reco_variables + deepcsv_nodes
variables = {}
variables["4j_ge3t"] = current_set
variables["5j_ge3t"] = current_set
variables["ge6j_ge3t"] = current_set
