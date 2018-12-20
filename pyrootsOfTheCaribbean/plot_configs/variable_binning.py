import numpy as np
default = 30
binning = {

"Jet_Pt[0]":		                                    {"nbins": default, "bin_range": [30.,400.]},
"Jet_Eta[0]":		                                    {"nbins": default, "bin_range": [-2.4,2.4]},
"Jet_CSV[0]":		                                    {"nbins": default, "bin_range": [-0.2,1.]},

"Jet_Pt[1]":		                                    {"nbins": default, "bin_range": [30.,300.]},
"Jet_Eta[1]":		                                    {"nbins": default, "bin_range": [-2.4,2.4]},
"Jet_CSV[1]":		                                    {"nbins": default, "bin_range": [-0.2,1.]},

"Jet_Pt[2]":		                                    {"nbins": default, "bin_range": [30.,200.]},
"Jet_Eta[2]":		                                    {"nbins": default, "bin_range": [-2.4,2.4]},
"Jet_CSV[2]":		                                    {"nbins": default, "bin_range": [-0.2,1.]},

"Jet_Pt[3]":		                                    {"nbins": default, "bin_range": [30.,100.]},
"Jet_Eta[3]":		                                    {"nbins": default, "bin_range": [-2.4,2.4]},
"Jet_CSV[3]":		                                    {"nbins": default, "bin_range": [-0.2,1.]},

"LooseLepton_Pt[0]":                                        {"nbins": default, "bin_range": [0.,600.]},
"LooseLepton_Eta[0]":		                            {"nbins": default, "bin_range": [-2.4,2.4]},

"Evt_HT":		                                    {"nbins": default, "bin_range": [0.,1000.]},
"BDT_common5_input_HT_tag":                                 {"nbins": default , "bin_range": [0.,800.]},

"Evt_Dr_MinDeltaRJets":		                            {"nbins": default, "bin_range": [0.,3.5]},
"Evt_Dr_MinDeltaRTaggedJets":		                    {"nbins": default, "bin_range": [0.,3.5]},

"BDT_common5_input_max_dR_jj":		                    {"nbins": default, "bin_range": [0.,2.*np.pi]}, 
"BDT_common5_input_max_dR_bb":		                    {"nbins": default, "bin_range": [0.,2.*np.pi]},

"BDT_common5_input_aplanarity_jets":		            {"nbins": default , "bin_range": [0., 0.45]},
"BDT_common5_input_aplanarity_tags":		            {"nbins": default , "bin_range": [0., 0.2]},

"Evt_JetPtOverJetE":		                            {"nbins": default, "bin_range": [0.2,1.0]},
"BDT_common5_input_pt_all_jets_over_E_all_jets_tags":	    {"nbins": default, "bin_range": [0.0,1.0]},

"BDT_common5_input_sphericity_jets":		            {"nbins": default, "bin_range": [0.,1.]},
"BDT_common5_input_sphericity_tags":		            {"nbins": default, "bin_range": [0.,1.]},


"BDT_common5_input_transverse_sphericity_jets":		    {"nbins": default, "bin_range": [0.,1.]},
"BDT_common5_input_transverse_sphericity_tags":		    {"nbins": default, "bin_range": [0.,1.]},

"Evt_CSV_Average":		                            {"nbins": default, "bin_range": [0.3,0.9]},
"Evt_CSV_Average_Tagged":		                    {"nbins": default, "bin_range": [0.5,1.]},

"CSV[0]":		                                    {"nbins": default, "bin_range": [0.75,1.]},
"CSV[1]":		                                    {"nbins": default, "bin_range": [0.5,1.]},

"Evt_CSV_Min":		                                    {"nbins": default, "bin_range": [0.,1.]},
"Evt_CSV_Min_Tagged":		                            {"nbins": default, "bin_range": [0.5,1.]},

"Evt_Dr_MinDeltaRLeptonJet":		                    {"nbins": default, "bin_range": [0.,3.5]},
"Evt_Dr_MinDeltaRLeptonTaggedJet":		            {"nbins": default, "bin_range": [0.,3.5]},


"BDT_common5_input_h1":		                            {"nbins": default, "bin_range": [-0.2,0.3]},
"BDT_common5_input_h2":		                            {"nbins": default, "bin_range": [-0.2,0.3]},
"BDT_common5_input_h3":		                            {"nbins": default, "bin_range": [-0.2,0.3]},

"Evt_M_MinDeltaRLeptonTaggedJet":		            {"nbins": default, "bin_range": [0.,500.]},

"Evt_Deta_TaggedJetsAverage":		                    {"nbins": default, "bin_range": [0.,np.pi]},
"Evt_Dr_TaggedJetsAverage":		                    {"nbins": default, "bin_range": [0.5,3.5]},

"BDT_common5_input_closest_tagged_dijet_mass":		    {"nbins": default , "bin_range": [0.,400.]},
"BDT_common5_input_dev_from_avg_disc_btags":		    {"nbins": default , "bin_range": [0.,0.06]},

"Evt_M_JetsAverage":                                        {"nbins": default, "bin_range": [5.,20.]},
"Evt_M2_TaggedJetsAverage":		                    {"nbins": default, "bin_range": [50.,500.]},

"N_BTagsT":	                                            {"nbins": 5, "bin_range": [-0.5,4.5]},

"BDT_common5_input_tagged_dijet_mass_closest_to_125":	    {"nbins": default, "bin_range": [0.,400.]},

"Evt_blr_ETH":		                                    {"nbins": default, "bin_range": [0.,1.]},
"Evt_blr_ETH_transformed":		                    {"nbins": default, "bin_range": [-5.,10.]},
"memDBp":	                                            {"nbins": default, "bin_range": [0.,1.]},

"GenAdd_BB_inacceptance":		                    {"nbins": 2, "bin_range": [-0.5,1.5]},
"GenAdd_B_inacceptance":		                    {"nbins": 2, "bin_range": [-0.5,1.5]},
"GenHiggs_BB_inacceptance":		                    {"nbins": 2, "bin_range": [-0.5,1.5]},
"GenHiggs_B_inacceptance":		                    {"nbins": 2, "bin_range": [-0.5,1.5]},
"GenTopHad_B_inacceptance":		                    {"nbins": 2, "bin_range": [-0.5,1.5]},
"GenTopHad_QQ_inacceptance":		                    {"nbins": 2, "bin_range": [-0.5,1.5]},
"GenTopHad_Q_inacceptance":		                    {"nbins": 2, "bin_range": [-0.5,1.5]},
"GenTopLep_B_inacceptance":                                 {"nbins": 2, "bin_range": [-0.5,1.5]},

"GenAdd_BB_inacceptance_jet":		                    {"nbins": 2, "bin_range": [-0.5,1.5]},
"GenAdd_B_inacceptance_jet":		                    {"nbins": 2, "bin_range": [-0.5,1.5]},
"GenHiggs_BB_inacceptance_jet":		                    {"nbins": 2, "bin_range": [-0.5,1.5]},
"GenHiggs_B_inacceptance_jet":		                    {"nbins": 2, "bin_range": [-0.5,1.5]},
"GenTopHad_B_inacceptance_jet":		                    {"nbins": 2, "bin_range": [-0.5,1.5]},
"GenTopHad_QQ_inacceptance_jet":		            {"nbins": 2, "bin_range": [-0.5,1.5]},
"GenTopHad_Q_inacceptance_jet":		                    {"nbins": 2, "bin_range": [-0.5,1.5]},
"GenTopLep_B_inacceptance_jet":                             {"nbins": 2, "bin_range": [-0.5,1.5]},

"GenAdd_BB_inacceptance_part":		                    {"nbins": 2, "bin_range": [-0.5,1.5]},
"GenAdd_B_inacceptance_part":		                    {"nbins": 2, "bin_range": [-0.5,1.5]},
"GenHiggs_BB_inacceptance_part":		            {"nbins": 2, "bin_range": [-0.5,1.5]},
"GenHiggs_B_inacceptance_part":		                    {"nbins": 2, "bin_range": [-0.5,1.5]},
"GenTopHad_B_inacceptance_part":		            {"nbins": 2, "bin_range": [-0.5,1.5]},
"GenTopHad_QQ_inacceptance_part":		            {"nbins": 2, "bin_range": [-0.5,1.5]},
"GenTopHad_Q_inacceptance_part":		            {"nbins": 2, "bin_range": [-0.5,1.5]},
"GenTopLep_B_inacceptance_part":                            {"nbins": 2, "bin_range": [-0.5,1.5]},

"Weight_XS":                                                {"nbins": 3, "bin_range": [0.0,0.0005]},
"Weight_CSV":                                               {"nbins": default, "bin_range": [0.,2.4]},
"Weight_GEN_nom":                                           {"nbins": 10, "bin_range": [0.,350.]}
}
