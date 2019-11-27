variables = {}
variables["ge4j_ge3t"] = [
	"Reco_Higgs_B1_Eta",
	"Reco_Higgs_B1_E",
	"Reco_Higgs_B1_CSV",
	"Reco_Higgs_B1_logPt",
	"Reco_Higgs_B1_logE",
	"Reco_Higgs_B2_Eta",
	"Reco_Higgs_B2_E",
	"Reco_Higgs_B2_CSV",
	"Reco_Higgs_B2_logPt",
	"Reco_Higgs_B2_logE",
	"Reco_Higgs_Pt",
	"Reco_Higgs_Eta",
	"Reco_Higgs_M",
	"Reco_Higgs_logM",
	"Reco_Higgs_E",
	"Reco_Higgs_logE",
	"Reco_Higgs_logPt",
	"Delta_R",
	"Delta_Phi",
	"Delta_Eta",
	"Delta_R3D",
	"Boosted1_Pt",
	"Boosted1_Eta",
	"Boosted1_logPt",
	"Boosted2_Pt",
	"Boosted2_Eta",
	"Boosted2_logPt",
	"Angle"
	]


all_variables = list(set( [v for key in variables for v in variables[key] ] ))
