variables = {}
features = ["Pt", "Eta", "Phi", "E"]
vlist = ["Jet_{feat}[{idx}]".format(feat = f, idx = i) for i in range(0,11) for f in features+["CSV"]]
vlist+= ["TightLepton_{feat}[0]".format(feat = f) for f in features]
vlist+= ["Evt_MET_Pt", "Evt_MET_Phi", "Evt_MET"] 
variables["ge4j_ge3t"] = vlist


all_variables = list(set( [v for key in variables for v in variables[key] ] ))


