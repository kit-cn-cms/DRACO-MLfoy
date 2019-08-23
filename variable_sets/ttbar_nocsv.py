variables = {}
variables["ge4j_ge3t"] = [
	"TopHad_B_Pt",
	"TopHad_B_Eta",
	"TopHad_B_Phi",
	"TopHad_B_E",
	"TopLep_B_Pt",
	"TopLep_B_Eta",
	"TopLep_B_Phi",
	"TopLep_B_E",
	"TopHad_Q1_Pt",
	"TopHad_Q1_Eta",
	"TopHad_Q1_Phi",
	"TopHad_Q1_E",
	"TopHad_Q2_Pt",
	"TopHad_Q2_Eta",
	"TopHad_Q2_Phi",
	"TopHad_Q2_E"
	]

variables["ge4j_ge2t"] = [
	"TopHad_B_Pt",
	"TopHad_B_Eta",
	"TopHad_B_Phi",
	"TopHad_B_E",
	"TopLep_B_Pt",
	"TopLep_B_Eta",
	"TopLep_B_Phi",
	"TopLep_B_E",
	"TopHad_Q1_Pt",
	"TopHad_Q1_Eta",
	"TopHad_Q1_Phi",
	"TopHad_Q1_E",
	"TopHad_Q2_Pt",
	"TopHad_Q2_Eta",
	"TopHad_Q2_Phi",
	"TopHad_Q2_E"
	]
all_variables = list(set( [v for key in variables for v in variables[key] ] ))
