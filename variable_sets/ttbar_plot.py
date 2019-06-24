variables = {}
variables["4j_ge3t"] = [
	"TopHad_B_Jet_Pt",
	"TopHad_B_Jet_Eta",
	"TopHad_B_Jet_CSV",
	"TopHad_B_Jet_Phi",
	"TopHad_B_Jet_E",
	"TopLep_B_Jet_Pt",
	"TopLep_B_Jet_Eta",
	"TopLep_B_Jet_Phi",
	"TopLep_B_Jet_E",
	"TopLep_B_Jet_CSV",
	"TopHad_Q1_Jet_Pt",
	"TopHad_Q1_Jet_Eta",
	"TopHad_Q1_Jet_Phi",
	"TopHad_Q1_Jet_E",
	"TopHad_Q1_Jet_CSV",
	"TopHad_Q2_Jet_Pt",
	"TopHad_Q2_Jet_Eta",
	"TopHad_Q2_Jet_Phi",
	"TopHad_Q2_Jet_E",
	"TopHad_Q2_Jet_CSV"
	]


all_variables = list(set( [v for key in variables for v in variables[key] ] ))
