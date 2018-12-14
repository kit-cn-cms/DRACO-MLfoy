print("using top 10 variables - number seven will shock you")
variables_4j_ge3t = [
    'CSV[0]',
    'BDT_common5_input_closest_tagged_dijet_mass',
    'BDT_common5_input_h2',
    'BDT_common5_input_sphericity_jets',
    'LooseLepton_Eta[0]',
    'Jet_Pt[3]',
    'Jet_CSV[1]',
    'Evt_blr_ETH_transformed',
    'Evt_CSV_Min_Tagged',
    'BDT_common5_input_h3',
    'Evt_CSV_Average_Tagged',
    'BDT_common5_input_HT_tag',
    'MEM',
    'BDT_common5_input_transverse_sphericity_jets',
    'Evt_Dr_MinDeltaRTaggedJets',
    'CSV[1]',
    'Jet_Eta[1]',
    'Evt_M_JetsAverage',
    'Evt_M2_TaggedJetsAverage',
    'Evt_CSV_Average',
    'N_BTagsT',
    'Evt_CSV_Min',
    ]


variables_5j_ge3t = [
    'MEM',
    'Evt_Dr_MinDeltaRTaggedJets',
    'CSV[0]',
    'BDT_common5_input_sphericity_jets',
    'Evt_Dr_MinDeltaRLeptonTaggedJet',
    'Jet_Pt[1]',
    'BDT_common5_input_HT_tag',
    'Jet_CSV[2]',
    'Evt_blr_ETH_transformed',
    'BDT_common5_input_aplanarity_tags',
    'BDT_common5_input_dev_from_avg_disc_btags',
    'Evt_CSV_Average',
    'Jet_Pt[2]',
    'Evt_M_JetsAverage',
    'Evt_CSV_Average_Tagged',
    'BDT_common5_input_closest_tagged_dijet_mass',
    ]


variables_ge6j_ge3t = [
    'Evt_CSV_Average',
    'Evt_Dr_MinDeltaRTaggedJets',
    'BDT_common5_input_HT_tag',
    'Evt_M2_TaggedJetsAverage',
    'MEM',
    'Evt_Dr_TaggedJetsAverage',
    'BDT_common5_input_transverse_sphericity_tags',
    'Evt_CSV_Min_Tagged',
    'Evt_M_JetsAverage',
    'Evt_blr_ETH_transformed',
    ]


all_variables = set(variables_4j_ge3t + variables_5j_ge3t + variables_ge6j_ge3t)
print("all variables:")
for v in all_variables:
    print(v)
print("-"*30)
