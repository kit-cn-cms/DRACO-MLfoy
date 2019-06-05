variables = {}
variables["4j_ge3t"] = [
    'CSV[0]',
    'CSV[1]',
    'CSV[2]',
    'CSV[3]',
    'Jet_DeepJetCSV[0]',
    'Jet_DeepJetCSV[1]',
    'Jet_DeepJetCSV[2]',
    'Jet_DeepJetCSV[3]',
    'Jet_E[0]',
    'Jet_E[1]',
    'Jet_E[2]',
    'Jet_E[3]',
    'Jet_Eta[0]',
    'Jet_Eta[1]',
    'Jet_Eta[2]',
    'Jet_Eta[3]',
    'Jet_M[0]',
    'Jet_M[1]',
    'Jet_M[2]',
    'Jet_M[3]',
    'Jet_Phi[0]',
    'Jet_Phi[1]',
    'Jet_Phi[2]',
    'Jet_Phi[3]',
    'Jet_Pt[0]',
    'Jet_Pt[1]',
    'Jet_Pt[2]',
    'Jet_Pt[3]',
    'LooseLepton_E[0]',
    'LooseLepton_Eta[0]',
    'LooseLepton_Phi[0]',
    'LooseLepton_Pt[0]',
    ]
variables["ge4j_ge3t"] = variables["4j_ge3t"]
variables["le5j_ge3t"] = variables["4j_ge3t"]
variables["ge4j_3t"] = [v for v in variables["4j_ge3t"] if not v == "N_BTagsM"]
variables["ge4j_ge4t"] = variables["4j_ge3t"]

variables["5j_ge3t"] = variables["4j_ge3t"] + [
    'CSV[4]',
    'Jet_DeepJetCSV[4]',
    'Jet_E[4]',
    'Jet_Eta[4]',
    'Jet_M[4]',
    'Jet_Phi[4]',
    'Jet_Pt[4]',
    ]

variables["ge6j_ge3t"] = variables["5j_ge3t"] + [
    'CSV[5]',
    'Jet_DeepJetCSV[5]',
    'Jet_E[5]',
    'Jet_Eta[5]',
    'Jet_M[5]',
    'Jet_Phi[5]',
    'Jet_Pt[5]',
    ]


all_variables = list(set( [v for key in variables for v in variables[key] ] ))
