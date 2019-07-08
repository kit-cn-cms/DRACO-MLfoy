variables = {}
variables["example"] = [
    "E1",
    "E2",
    "pt1",
    "pt2",
    "dPhi"
    ]



all_variables = list(set( [v for key in variables for v in variables[key] ] ))
