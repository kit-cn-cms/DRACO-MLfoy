variables = {}
variables["ge4j_ge3t"] = [
	#"N_Jets",
	#"N_BTags_M",
	"TopHad_B_Pt",
	"TopHad_B_Eta",
	"TopHad_B_Phi",
	"TopHad_B_E",
	"TopHad_B_CSV",
	"TopLep_B_Pt",
	"TopLep_B_Eta",
	"TopLep_B_Phi",
	"TopLep_B_E",
	"TopLep_B_CSV",
	"TopHad_Q1_Pt",
	"TopHad_Q1_Eta",
	"TopHad_Q1_Phi",
	"TopHad_Q1_E",
	"TopHad_Q1_CSV",
	"TopHad_Q2_Pt",
	"TopHad_Q2_Eta",
	"TopHad_Q2_Phi",
	"TopHad_Q2_E",
	"TopHad_Q2_CSV"
	]

variables["ge4j_ge2t"] = [
	"TopHad_B_Pt",
	"TopHad_B_Eta",
	"TopHad_B_Phi",
	"TopHad_B_E",
	"TopHad_B_CSV",
	"TopLep_B_Pt",
	"TopLep_B_Eta",
	"TopLep_B_Phi",
	"TopLep_B_E",
	"TopLep_B_CSV",
	"TopHad_Q1_Pt",
	"TopHad_Q1_Eta",
	"TopHad_Q1_Phi",
	"TopHad_Q1_E",
	"TopHad_Q1_CSV",
	"TopHad_Q2_Pt",
	"TopHad_Q2_Eta",
	"TopHad_Q2_Phi",
	"TopHad_Q2_E",
	"TopHad_Q2_CSV"
	]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
