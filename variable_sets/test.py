variables = {}
variables["ge4j_ge3t"] = [
	"Jet_Pt[0]",
	"Jet_Pt[1]",
	"Jet_Pt[2]",
	"Jet_Pt[3]",
	"Jet_Pt[4]",
	#"GenHiggs_logM"
	]


all_variables = list(set( [v for key in variables for v in variables[key] ] ))
