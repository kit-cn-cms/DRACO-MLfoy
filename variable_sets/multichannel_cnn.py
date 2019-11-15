variables = {}
variables["6j_ge3t"] = [
    'Jet_Pt[0-16]_Hist',
    'TaggedJet_Pt[0-9]_Hist',
    'Electron_Pt[0-9]_Hist',
    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
