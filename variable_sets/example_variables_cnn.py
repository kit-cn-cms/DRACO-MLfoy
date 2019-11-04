variables = {}
variables["test"] = [
    'Jet_Pt[0-16]_Hist',
    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
