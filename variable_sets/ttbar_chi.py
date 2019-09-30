variables = {}
variables["ge4j_ge3t"] = [
	"reco_TopHad_M",
	"reco_TopLep_M"
	]

variables["ge4j_ge2t"] = [
        "reco_TopHad_M",
        "reco_TopLep_M",
	"reco_WHad_M"
        ]


all_variables = list(set( [v for key in variables for v in variables[key] ] ))

