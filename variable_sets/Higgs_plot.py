variables = {}
variables["ge4j_ge3t"] = [
	#"N_Jets",
	#"N_BTags_M",
	"Higgs_B1_Pt",
	"Higgs_B1_Eta",
	"Higgs_B1_Phi",
	"Higgs_B1_E",
	"Higgs_B1_CSV",
	"Higgs_B2_Pt",
	"Higgs_B2_Eta",
	"Higgs_B2_Phi",
	"Higgs_B2_E",
	"Higgs_B2_CSV",
	"reco_Higgs_Pt",
	"reco_Higgs_Eta",
	"reco_Higgs_Phi",
	"reco_Higgs_M",
	"reco_Higgs_logM",
	"GenHiggs_B1_Phi",
	"GenHiggs_B1_Eta",
	"GenHiggs_B2_Phi",
	"GenHiggs_B2_Eta",
	"DeltaR_B1",
	"DeltaR_B2",
	"DeltaR_B1_B2"	
	]

variables["ge4j_ge2t"] = [
	"Higgs_B1_Pt",
	"Higgs_B1_Eta",
	"Higgs_B1_Phi",
	"Higgs_B1_E",
	"Higgs_B1_CSV",
	"Higgs_B2_Pt",
	"Higgs_B2_Eta",
	"Higgs_B2_Phi",
	"Higgs_B2_E",
	"Higgs_B2_CSV"
	]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
