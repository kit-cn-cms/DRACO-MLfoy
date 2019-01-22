import numpy as np
default = 25
variables = {}

class Variable:
    def __init__(self, bin_range, nbins = default):
        self.bin_range  = bin_range
        self.nbins      = nbins


# add variable settings
variables["Jet_Pt[0]"]		                                        = Variable(bin_range = [30.,400.])
variables["BDT_common5_input_HT"]                                   = Variable(bin_range = [50.,1500.])
variables["BDT_common5_input_HT_tag"]		                        = Variable(bin_range = [50.0,800.0])
variables["BDT_common5_input_aplanarity_jets"]		                = Variable(bin_range = [0.0,0.3])
variables["BDT_common5_input_aplanarity_tags"]		                = Variable(bin_range = [0.0,0.15])
variables["BDT_common5_input_closest_tagged_dijet_mass"]		    = Variable(bin_range = [0.0,400.0])
variables["BDT_common5_input_dev_from_avg_disc_btags"]		        = Variable(bin_range = [0.0,0.06])
variables["BDT_common5_input_h0"]                                   = Variable(bin_range = [0.2,0.45])
variables["BDT_common5_input_h1"]		                            = Variable(bin_range = [-0.2,0.3])
variables["BDT_common5_input_h2"]		                            = Variable(bin_range = [-0.2,0.3])
variables["BDT_common5_input_h3"]		                            = Variable(bin_range = [-0.2,0.3])
variables["BDT_common5_input_max_dR_bb"]		                    = Variable(bin_range = [1.0,5.])
variables["BDT_common5_input_max_dR_jj"]		                    = Variable(bin_range = [1.,5.])
variables["BDT_common5_input_pt_all_jets_over_E_all_jets_tags"]		= Variable(bin_range = [0.2,1.0])
variables["BDT_common5_input_pt_all_jets_over_E_all_jets"]		    = Variable(bin_range = [0.2,1.0])
variables["BDT_common5_input_sphericity_jets"]		                = Variable(bin_range = [0.0,1.0])
variables["BDT_common5_input_sphericity_tags"]		                = Variable(bin_range = [0.0,1.0])
variables["BDT_common5_input_tagged_dijet_mass_closest_to_125"]		= Variable(bin_range = [0.0,400.0])
variables["BDT_common5_input_transverse_sphericity_jets"]		    = Variable(bin_range = [0.0,1.0])
variables["BDT_common5_input_transverse_sphericity_tags"]		    = Variable(bin_range = [0.0,1.0])
variables["CSV[0]"]		                                            = Variable(bin_range = [0.75,1.0])
variables["CSV[1]"]		                                            = Variable(bin_range = [0.5,1.0])
variables["Evt_CSV_Average"]		                                = Variable(bin_range = [0.3,0.9])
variables["Evt_CSV_Average_Tagged"]		                            = Variable(bin_range = [0.5,1.0])
variables["Evt_CSV_Min"]		                                    = Variable(bin_range = [0.0,0.5])
variables["Evt_CSV_Min_Tagged"]		                                = Variable(bin_range = [0.5,1.0])
variables["Evt_Deta_TaggedJetsAverage"]		                        = Variable(bin_range = [0.0,3.])
variables["Evt_Eta_TaggedJetsAverage"]                              = Variable(bin_range = [-2.,2.])
variables["Evt_Eta_UntaggedJetsAverage"]                            = Variable(bin_range = [-2.5,2.5])
variables["Evt_Dr_MinDeltaRJets"]		                            = Variable(bin_range = [0.5,2.5])
variables["Evt_Dr_MinDeltaRLeptonJet"]	                        	= Variable(bin_range = [0.5,3.])
variables["Evt_Dr_MinDeltaRLeptonTaggedJet"]                		= Variable(bin_range = [0.5,3.])
variables["Evt_Dr_MinDeltaRTaggedJets"]		                        = Variable(bin_range = [0.5,3.5])
variables["Evt_Dr_TaggedJetsAverage"]		                        = Variable(bin_range = [0.5,3.5])
variables["Evt_Dr_UntaggedJetsAverage"]                             = Variable(bin_range = [0.5,4.])
variables["Evt_HT"]		                                            = Variable(bin_range = [100.0,1500.0])
variables["Evt_HT_Jets"]                                            = Variable(bin_range = [0.,1500.0])
variables["Evt_JetPtOverJetE"]		                                = Variable(bin_range = [0.2,1.0])
variables["Evt_M2_TaggedJetsAverage"]		                        = Variable(bin_range = [50.0,500.0])
variables["Evt_M_JetsAverage"]		                                = Variable(bin_range = [0.0,15.0])
variables["Evt_M_MinDeltaRLeptonTaggedJet"]		                    = Variable(bin_range = [0.0,300.0])
variables["Evt_blr_ETH"]		                                    = Variable(bin_range = [0.0,1.0])
variables["Evt_blr_ETH_transformed"]		                        = Variable(bin_range = [-5.0,10.0])
variables["GenAdd_BB_inacceptance"]		                            = Variable(bin_range = [-0.5,1.5],          nbins = 2)
variables["GenAdd_BB_inacceptance_jet"]		                        = Variable(bin_range = [-0.5,1.5],          nbins = 2)
variables["GenAdd_BB_inacceptance_part"]		                    = Variable(bin_range = [-0.5,1.5],          nbins = 2)
variables["GenAdd_B_inacceptance"]		                            = Variable(bin_range = [-0.5,1.5],          nbins = 2)
variables["GenAdd_B_inacceptance_jet"]		                        = Variable(bin_range = [-0.5,1.5],          nbins = 2)
variables["GenAdd_B_inacceptance_part"]		                        = Variable(bin_range = [-0.5,1.5],          nbins = 2)
variables["GenHiggs_BB_inacceptance"]		                        = Variable(bin_range = [-0.5,1.5],          nbins = 2)
variables["GenHiggs_BB_inacceptance_jet"]		                    = Variable(bin_range = [-0.5,1.5],          nbins = 2)
variables["GenHiggs_BB_inacceptance_part"]		                    = Variable(bin_range = [-0.5,1.5],          nbins = 2)
variables["GenHiggs_B_inacceptance"]		                        = Variable(bin_range = [-0.5,1.5],          nbins = 2)
variables["GenHiggs_B_inacceptance_jet"]		                    = Variable(bin_range = [-0.5,1.5],          nbins = 2)
variables["GenHiggs_B_inacceptance_part"]		                    = Variable(bin_range = [-0.5,1.5],          nbins = 2)
variables["GenTopHad_B_inacceptance"]		                        = Variable(bin_range = [-0.5,1.5],          nbins = 2)
variables["GenTopHad_B_inacceptance_jet"]		                    = Variable(bin_range = [-0.5,1.5],          nbins = 2)
variables["GenTopHad_B_inacceptance_part"]		                    = Variable(bin_range = [-0.5,1.5],          nbins = 2)
variables["GenTopHad_QQ_inacceptance"]		                        = Variable(bin_range = [-0.5,1.5],          nbins = 2)
variables["GenTopHad_QQ_inacceptance_jet"]		                    = Variable(bin_range = [-0.5,1.5],          nbins = 2)
variables["GenTopHad_QQ_inacceptance_part"]		                    = Variable(bin_range = [-0.5,1.5],          nbins = 2)
variables["GenTopHad_Q_inacceptance"]		                        = Variable(bin_range = [-0.5,1.5],          nbins = 2)
variables["GenTopHad_Q_inacceptance_jet"]		                    = Variable(bin_range = [-0.5,1.5],          nbins = 2)
variables["GenTopHad_Q_inacceptance_part"]		                    = Variable(bin_range = [-0.5,1.5],          nbins = 2)
variables["GenTopLep_B_inacceptance"]		                        = Variable(bin_range = [-0.5,1.5],          nbins = 2)
variables["GenTopLep_B_inacceptance_jet"]		                    = Variable(bin_range = [-0.5,1.5],          nbins = 2)
variables["GenTopLep_B_inacceptance_part"]		                    = Variable(bin_range = [-0.5,1.5],          nbins = 2)
variables["Jet_CSV[0]"]		                                        = Variable(bin_range = [0.,1.0])
variables["Jet_CSV[1]"]		                                        = Variable(bin_range = [0.,1.0])
variables["Jet_CSV[2]"]		                                        = Variable(bin_range = [0.,1.0])
variables["Jet_CSV[3]"]		                                        = Variable(bin_range = [0.,1.0])
variables["Jet_Eta[0]"]		                                        = Variable(bin_range = [-2.4,2.4])
variables["Jet_Eta[1]"]		                                        = Variable(bin_range = [-2.4,2.4])
variables["Jet_Eta[2]"]		                                        = Variable(bin_range = [-2.4,2.4])
variables["Jet_Eta[3]"]		                                        = Variable(bin_range = [-2.4,2.4])
variables["Jet_Pt[0]"]		                                        = Variable(bin_range = [30.0,400.0])
variables["Jet_Pt[1]"]		                                        = Variable(bin_range = [30.0,300.0])
variables["Jet_Pt[2]"]		                                        = Variable(bin_range = [30.0,200.0])
variables["Jet_Pt[3]"]		                                        = Variable(bin_range = [30.0,100.0])
variables["LooseLepton_Eta[0]"]		                                = Variable(bin_range = [-2.4,2.4])
variables["LooseLepton_Pt[0]"]		                                = Variable(bin_range = [0.0,600.0])
variables["N_BTagsT"]		                                        = Variable(bin_range = [-0.5,4.5],          nbins = 5)
variables["Weight_CSV"]		                                        = Variable(bin_range = [0.0,2.4])
variables["Weight_XS"]		                                        = Variable(bin_range = [0.0,0.00050],       nbins = 3)
variables["memDBp"]		                                            = Variable(bin_range = [0.0,1.0])
variables["Evt_Pt_MET"]                             = Variable(bin_range = [20.0,1000.0])
variables["Evt_Pt_MinDeltaRJets"]                   = Variable(bin_range = [7.0,1500.0])
variables["Evt_Pt_MinDeltaRTaggedJets"]             = Variable(bin_range = [1.0,1500.0])
variables["Evt_Pt_MinDeltaRUntaggedJets"]           = Variable(bin_range = [0.0,1500.0])
variables["Evt_Pt_PrimaryLepton"]                   = Variable(bin_range = [30.0,1000.0])
variables["Evt_TaggedJet_MaxDeta_Jets"]             = Variable(bin_range = [0.0,4.0])
variables["Evt_TaggedJet_MaxDeta_TaggedJets"]       = Variable(bin_range = [0.0,5.5])
variables["N_AK8Jets"]                              = Variable(bin_range = [-0.5,6.5],          nbins = 7)
variables["N_BTagsL"]                               = Variable(bin_range = [2.5,9.5],           nbins = 7)
variables["N_BTagsM"]                               = Variable(bin_range = [2.5,7.5],           nbins = 5)
variables["N_Jets"]                                 = Variable(bin_range = [3.5,10.5],          nbins = 7)
variables["N_LooseJets"]                            = Variable(bin_range = [3.5,10.5],          nbins = 7)
variables["CSV[2]"]                                 = Variable(bin_range = [0.5,1.0])
variables["CSV[3]"]                                 = Variable(bin_range = [0.5,1.0])
variables["Jet_CSV_DNN[0]"]                         = Variable(bin_range = [-2.0,1.0])
variables["Jet_E[0]"]                               = Variable(bin_range = [35.0,4000.0])
variables["Jet_M[0]"]                               = Variable(bin_range = [0.5,250.0])
variables["Jet_Phi[0]"]                             = Variable(bin_range = [-np.pi,np.pi])
variables["Jet_CSV_DNN[1]"]                         = Variable(bin_range = [-2.0,1.0])
variables["Jet_E[1]"]                               = Variable(bin_range = [30.0,3000.0])
variables["Jet_M[1]"]                               = Variable(bin_range = [0.0,200.0])
variables["Jet_Phi[1]"]                             = Variable(bin_range = [-np.pi,np.pi])
variables["Jet_CSV_DNN[2]"]                         = Variable(bin_range = [-2.0,1.0])
variables["Jet_E[2]"]                               = Variable(bin_range = [30.0,2500.0])
variables["Jet_M[2]"]                               = Variable(bin_range = [0.5,130.0])
variables["Jet_Phi[2]"]                             = Variable(bin_range = [-np.pi,np.pi])
variables["Jet_CSV_DNN[3]"]                         = Variable(bin_range = [-2.0,1.0])
variables["Jet_E[3]"]                               = Variable(bin_range = [30.0,1500.0])
variables["Jet_M[3]"]                               = Variable(bin_range = [1.0,75.0])
variables["Jet_Phi[3]"]                             = Variable(bin_range = [-np.pi,np.pi])
variables["LooseLepton_E[0]"]                       = Variable(bin_range = [30.0,2000.0])
variables["LooseLepton_M[0]"]                       = Variable(bin_range = [-0.5,0.5])
variables["LooseLepton_Phi[0]"]                     = Variable(bin_range = [-np.pi,np.pi])
variables["BDT_common5_input_all_sum_pt_with_met"]  = Variable(bin_range = [0.,1000.])
variables["BDT_common5_input_avg_btag_disc_btags"]  = Variable(bin_range = [0.6,1.])
variables["BDT_common5_input_avg_dr_tagged_jets"]   = Variable(bin_range = [1.,3.5])
variables["BDT_common5_input_best_higgs_mass"]      = Variable(bin_range = [0., 250.])
variables["BDT_common5_input_blr_transformed"]      = Variable(bin_range = [-5.,15.])
variables["BDT_common5_input_cos_theta_blep_bhad"]  = Variable(bin_range = [-1.,1.])
variables["BDT_common5_input_cos_theta_l_bhad"]     = Variable(bin_range = [-1.,1.])
variables["BDT_common5_input_delta_eta_blep_bhad"]  = Variable(bin_range = [0.,4.])
variables["BDT_common5_input_delta_eta_l_bhad"]     = Variable(bin_range = [0.,4.])
variables["BDT_common5_input_delta_phi_blep_bhad"]  = Variable(bin_range = [0.,2.*np.pi])
variables["BDT_common5_input_delta_phi_l_bhad"]     = Variable(bin_range = [0.,2.*np.pi])
variables["BDT_common5_input_dEta_fn"]              = Variable(bin_range = [0.,3.])
variables["BDT_common5_input_dr_between_lep_and_closest_jet"]   = Variable(bin_range = [0.5,3.])
variables["BDT_common5_input_Evt_CSV_Average"]                  = Variable(bin_range = [0.3,0.9])
variables["BDT_common5_input_Evt_Deta_JetsAverage"]             = Variable(bin_range = [0.25,2.5])
variables["Evt_Deta_JetsAverage"]                               = Variable(bin_range = [0.25,2.5])
variables["BDT_common5_input_Evt_Deta_TaggedJetsAverage"]       = Variable(bin_range = [0.,3.])
variables["Evt_Deta_TaggedJetsAverage"]                         = Variable(bin_range = [0.,3.])
variables["BDT_common5_input_Evt_M2_TaggedJetsAverage"]         = Variable(bin_range = [0.,500.])
variables["BDT_common5_input_fifth_highest_CSV"]                = Variable(bin_range = [0.,0.4])
variables["BDT_common5_input_first_jet_pt"]                     = Variable(bin_range = [0., 600.])
variables["BDT_common5_input_fourth_highest_btag"]              = Variable(bin_range = [0.,1.])
variables["BDT_common5_input_fourth_jet_pt"]                    = Variable(bin_range = [0.,200.])
variables["BDT_common5_input_invariant_mass_of_everything"]     = Variable(bin_range = [0.,2000.])
variables["BDT_common5_input_lowest_btag"]                      = Variable(bin_range = [0.5,1.])
variables["BDT_common5_input_M3"]                               = Variable(bin_range = [0.,1000.])
variables["BDT_common5_input_maxeta_jet_jet"]                   = Variable(bin_range = [0.,1.6])
variables["BDT_common5_input_maxeta_jet_tag"]                   = Variable(bin_range = [0.,1.6])
variables["BDT_common5_input_maxeta_tag_tag"]                   = Variable(bin_range = [0.,1.6])
variables["BDT_common5_input_MET"]                              = Variable(bin_range = [0.,400.])
variables["BDT_common5_input_MHT"]                              = Variable(bin_range = [0.,400.])
variables["Evt_MHT"]                                            = Variable(bin_range = [0.,400.])
variables["BDT_common5_input_min_dr_tagged_jets"]               = Variable(bin_range = [0.5,3.])
variables["BDT_common5_input_Mlb"]                              = Variable(bin_range = [0.,200.])
variables["BDT_common5_input_second_higest_btag"]               = Variable(bin_range = [0.5,1.])
variables["BDT_common5_input_second_jet_pt"]                    = Variable(bin_range = [0.,300.])
variables["BDT_common5_input_sphericity"]                       = Variable(bin_range = [0.,1.])
variables["BDT_common5_input_tagged_dijet_mass_closest_to_125"] = Variable(bin_range = [50.,250.])
variables["BDT_common5_input_third_highest_btag"]               = Variable(bin_range = [0.5,1.])
variables["BDT_common5_input_third_jet_pt"]                     = Variable(bin_range = [0.,200.])
variables["BDT_common5_input_transverse_sphericity"]            = Variable(bin_range = [0.,1.])
variables["Evt_CSV_Dev"]                                        = Variable(bin_range = [0.,0.25])
variables["Evt_CSV_Dev_Tagged"]                                 = Variable(bin_range = [0.,0.06])
variables["Evt_Deta_UntaggedJetsAverage"]                       = Variable(bin_range = [0.,3.])
variables["Evt_Dr_Jets_Average"]                                = Variable(bin_range = [1.5,3.5])
variables["Evt_Dr_MinDeltaRUntaggedJets"]                       = Variable(bin_range = [0.5,4.])
variables["Evt_E_PrimaryLepton"]                                = Variable(bin_range = [0.,500.])
variables["Evt_Eta_JetsAverage"]                                = Variable(bin_range = [-2.,2.])
variables["Evt_Eta_PrimaryLepton"]                              = Variable(bin_range = [-2.5,2.5])
variables["Evt_M2_JetsAverage"]                                 = Variable(bin_range = [50.,400.])
variables["Evt_M2_UntaggedJetsAverage"]                         = Variable(bin_range = [50.,400.])
variables["Evt_M3"]                                             = Variable(bin_range = [0.,1000.])
variables["Evt_M_MedianTaggedJets"]                             = Variable(bin_range = [0.,500.])
variables["Evt_M_MinDeltaRJets"]                                = Variable(bin_range = [0.,200.])
variables["Evt_M_MinDeltaRLeptonJet"]                           = Variable(bin_range = [0.,200.])
variables["BDT_common5_input_aplanarity"]                       = Variable(bin_range = [0.,0.5])
variables["BDT_common5_input_second_highest_btag"]              = Variable(bin_range = [0.5,1.])
variables["Evt_Dr_JetsAverage"]                                 = Variable(bin_range = [0.7,3.5])
variables["Evt_Jet_MaxDeta_Jets"]                               = Variable(bin_range = [0.,5.])
variables["Evt_M_MinDeltaRTaggedJets"]                          = Variable(bin_range = [50.,1000.])
variables["Evt_M_PrimaryLepton"]                                = Variable(bin_range = [-0.25,0.35])
variables["Evt_M_TaggedJetsAverage"]                            = Variable(bin_range = [5.,100.])
variables["Evt_M_TaggedJetsClosestTo125"]                       = Variable(bin_range = [0.,250.])
variables["Evt_M_Total"]                                        = Variable(bin_range = [250.,3000.])
variables["Evt_M_UntaggedJetsAverage"]                          = Variable(bin_range = [0.0,150.])
variables["Evt_Phi_MET"]                                        = Variable(bin_range = [-np.pi,np.pi])
variables["Evt_Phi_PrimaryLepton"]                              = Variable(bin_range = [-np.pi,np.pi])

variables["dEta_Lepton_hadB"]           = Variable(bin_range = [0.0,5.0])
variables["dPhi_hadB_BosonB2"]          = Variable(bin_range = [0.0,np.pi])
variables["dPhi_hadB_BosonB1"]          = Variable(bin_range = [0.0,np.pi])
variables["dR_Lepton_BosonB2"]          = Variable(bin_range = [0.0,7.0])
variables["dR_Lepton_BosonB1"]          = Variable(bin_range = [0.0,7.0])
variables["dEta_Lepton_lepB"]           = Variable(bin_range = [0.0,5.0])
variables["dPhi_lepB_BosonB2"]          = Variable(bin_range = [0.0,np.pi])
variables["dPhi_lepB_BosonB1"]          = Variable(bin_range = [0.0,np.pi])
variables["dR_hadB_BosonB2"]            = Variable(bin_range = [0.0,7.0])
variables["dEta_lepB_BosonB1"]          = Variable(bin_range = [0.0,5.0])
variables["dEta_lepB_BosonB2"]          = Variable(bin_range = [0.0,5.0])
variables["dEta_Lepton_BosonB1"]            = Variable(bin_range = [0.0,5.0])
variables["dPhi_BosonB1_BosonB2"]           = Variable(bin_range = [0.0,np.pi])
variables["dR_Lepton_lepB"]         = Variable(bin_range = [0.0,7.0])
variables["dPhi_Lepton_hadB"]           = Variable(bin_range = [0.0,np.pi])
variables["dR_lepB_BosonB2"]            = Variable(bin_range = [0.0,7.0])
variables["dEta_Lepton_BosonB2"]            = Variable(bin_range = [0.0,5.0])
variables["dEta_hadB_BosonB1"]          = Variable(bin_range = [0.0,5.0])
variables["dEta_BosonB1_BosonB2"]           = Variable(bin_range = [0.0,5.0])
variables["dEta_hadB_BosonB2"]          = Variable(bin_range = [0.0,5.0])
variables["dR_BosonB1_BosonB2"]         = Variable(bin_range = [0.0,7.0])
variables["dR_Lepton_hadB"]         = Variable(bin_range = [0.0,7.0])
variables["dR_hadB_BosonB1"]            = Variable(bin_range = [0.0,7.0])
variables["dR_lepB_BosonB1"]            = Variable(bin_range = [0.0,7.0])
variables["dEta_hadB_lepB"]         = Variable(bin_range = [0.0,5.0])
variables["dR_hadB_lepB"]           = Variable(bin_range = [0.0,7.0])
variables["dPhi_hadB_lepB"]         = Variable(bin_range = [0.0,np.pi])
variables["dPhi_Lepton_BosonB1"]            = Variable(bin_range = [0.0,np.pi])
variables["dPhi_Lepton_BosonB2"]            = Variable(bin_range = [0.0,np.pi])
variables["dPhi_Lepton_lepB"]           = Variable(bin_range = [0.0,np.pi])

variables["pT_lepB"]            = Variable(bin_range = [0.,400.])
variables["eta_lepB"]           = Variable(bin_range = [-3.,3.])
variables["phi_hadB"]           = Variable(bin_range = [-np.pi,np.pi])
variables["phi_lepB"]           = Variable(bin_range = [-np.pi,np.pi])
variables["phi_Lepton"]         = Variable(bin_range = [-np.pi,np.pi])
variables["pT_hadB"]            = Variable(bin_range = [0.,400.])
variables["eta_hadB"]           = Variable(bin_range = [-3.,3.])
variables["pT_Lepton"]          = Variable(bin_range = [0.,300.])
variables["phi_BosonB2"]            = Variable(bin_range = [-np.pi,np.pi])
variables["phi_BosonB1"]            = Variable(bin_range = [-np.pi,np.pi])
variables["pT_BosonB2"]         = Variable(bin_range = [0.,400.])
variables["pT_BosonB1"]         = Variable(bin_range = [0.,400.])
variables["eta_Lepton"]         = Variable(bin_range = [-3.,3.])
variables["eta_BosonB2"]            = Variable(bin_range = [-3.,3.])
variables["eta_BosonB1"]            = Variable(bin_range = [-3.,3.])



def getNbins(variable):
    if variable in variables:
        return variables[variable].nbins
    else:
        return default

def getBinrange(variable):
    if variable in variables:
        return variables[variable].bin_range
    else:
        print("no binrange found for variable {}".format(variable))
        return None
