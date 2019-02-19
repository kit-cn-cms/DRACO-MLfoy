
variables = {}

variables["4j_ge3t"] = [
    'Evt_CSV_Average_Tagged',
    'Evt_CSV_Min',
    'Evt_Deta_TaggedJetsAverage',
    'Evt_CSV_Min_Tagged',
    'BDT_common5_input_h1',
    'Evt_CSV_Average',
    'Evt_M2_TaggedJetsAverage',
    'Evt_blr_ETH_transformed',
    'Evt_HT',
    'Evt_M_JetsAverage',
    ]


variables["5j_ge3t"] = [
    'BDT_common5_input_HT_tag',
    'Evt_CSV_Average_Tagged',
    'Evt_CSV_Min_Tagged',
    'Evt_JetPtOverJetE',
    'BDT_common5_input_h1',
    'Evt_M2_TaggedJetsAverage',
    'Evt_HT',
    'Evt_CSV_Average',
    'Evt_blr_ETH_transformed',
    'Evt_M_JetsAverage',
    ]


variables["ge6j_ge3t"] = [
    'Evt_Deta_TaggedJetsAverage',
    'BDT_common5_input_h1',
    'Evt_CSV_Average_Tagged',
    'Evt_CSV_Min_Tagged',
    'Evt_M2_TaggedJetsAverage',
    'Evt_HT',
    'Evt_JetPtOverJetE',
    'Evt_CSV_Average',
    'Evt_M_JetsAverage',
    'Evt_blr_ETH_transformed',
    ]

all_variables = set( [v for key in variables for v in variables[key] ] )

