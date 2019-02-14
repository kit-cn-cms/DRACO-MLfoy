# import ROOT in batch mode
import sys
import ROOT
import numpy as np
import pandas as pd
import copy
# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so");
ROOT.gSystem.Load("libDataFormatsFWLite.so");
ROOT.FWLiteEnabler.enable()

# load FWlite python libraries
from DataFormats.FWLite import Handle, Events

# define which variables to add
def calculateVariables(evt):
    # get variables already saved in the event
    event_data = evt.variables
    
    # ttbar event kinematics

    # all interesting objects
    #   'Boson':    Z or H Boson
    #   'Lepton':   Lepton from t->b(W->lnu) decay
    #   'hadTop':   hadronically (t->b(W->qq)) decaying top quark
    #   'lepTop':   leptonically (t->b(W->lnu)) decaying top quark
    #   'lepB':     b quark from leptonic top decay
    #   'hadB':     b quark from hadronci top decay
    #objects = ["Boson", "Lepton", "hadTop", "lepTop", "hadB", "lepB"]
    objects = ["Lepton", "hadB", "lepB"]

    # loop over all the objects
    for i, obj1 in enumerate(objects):
        # get interesting variables for these objects

        # transverse momentum
        event_data["pT_{}".format(obj1)] =          [evt.get_pT(obj1)]
        # pseudo rapidity
        event_data["eta_{}".format(obj1)] =         [evt.get_eta(obj1)]
        # rapidity
        #event_data["y_{}".format(obj1)] =           [evt.get_y(obj1)]
        # cosine of azimuth angle
        #event_data["theta_{}".format(obj1)] =       [evt.get_theta(obj1)]
        #event_data["costheta_{}".format(obj1)] =    [evt.get_costheta(obj1)]
        # polar angle
        #event_data["phi_{}".format(obj1)] =         [evt.get_phi(obj1)]
        # mass of particle
        #event_data["mass_{}".format(obj1)] =        [evt.get_mass(obj1)]

        # loop over all the objects again for delta variables
        for j, obj2 in enumerate(objects):
            if j <= i: continue
            # pseudo rapidity difference
            event_data["dEta_{}_{}".format(obj1,obj2)] =        [evt.get_dEta(obj1,obj2)]
            # rapidity difference
            #event_data["dY_{}_{}".format(obj1,obj2)] =          [evt.get_dY(obj1,obj2)]
            # difference in cosine of azimuth angle
            #event_data["dTheta_{}_{}".format(obj1,obj2)] =      [evt.get_dTheta(obj1,obj2)]
            #event_data["dcosTheta_{}_{}".format(obj1,obj2)] =   [evt.get_dcosTheta(obj1,obj2)]
            # difference in polar angle
            #event_data["dPhi_{}_{}".format(obj1,obj2)] =        [evt.get_dPhi(obj1,obj2)]
            # spacial difference in R = sqrt(dEta^2+dPhi^2)
            #event_data["dR_{}_{}".format(obj1,obj2)] =          [evt.get_dR(obj1,obj2)]

    
    # get the invariant ttbar system by adding both the top quarks
    ttbar = evt.objects["hadTop"].p4() + evt.objects["lepTop"].p4()   
    
    # save kinematic variables for ttbar system
    event_data["pT_{}".format("ttbar")] =       [ttbar.Pt()]
    event_data["eta_{}".format("ttbar")] =      [ttbar.Eta()]
    #event_data["y_{}".format("ttbar")] =        [ttbar.Rapidity()]
    #event_data["theta_{}".format("ttbar")] =    [ttbar.Theta()]
    #event_data["costheta_{}".format("ttbar")] = [np.cos(ttbar.Theta())]
    #event_data["phi_{}".format("ttbar")] =      [ttbar.Phi()]
    #event_data["mass_{}".format("ttbar")] =     [ttbar.M()]

    '''
    # get the boson (H/Z)
    boson = evt.objects["Boson"].p4()

    # save kinematic differences between Boson and ttbar system
    deta = np.abs(boson.Eta()-ttbar.Eta())
    event_data["dEta_{}_{}".format("Boson","ttbar")] =      [deta]
    #event_data["dTheta_{}_{}".format("Boson","ttbar")] =    [np.abs(boson.Theta() - ttbar.Theta())]
    #event_data["dcosTheta_{}_{}".format("Boson","ttbar")] = [np.abs(np.cos(boson.Theta())-np.cos(ttbar.Theta()))]
    event_data["dY_{}_{}".format("Boson","ttbar")] =        [np.abs(boson.Rapidity()-ttbar.Rapidity())]
    dphi = np.abs(boson.Phi()-ttbar.Phi())
    if dphi > np.pi: dphi = 2.*np.pi-dphi
    event_data["dPhi_{}_{}".format("Boson","ttbar")] =      [dphi]
    event_data["dR_{}_{}".format("Boson","ttbar")] =        [np.sqrt(deta**2+dphi**2)]
    
    # get ttX system by adding ttbar system and boson
    ttX = ttbar + boson
    
    # save kinematic variables for ttX system
    event_data["pT_{}".format("ttX")] =       [ttX.Pt()]
    event_data["eta_{}".format("ttX")] =      [ttX.Eta()]
    event_data["y_{}".format("ttX")] =        [ttX.Rapidity()]
    #event_data["theta_{}".format("ttX")] =    [ttX.Theta()]
    #event_data["costheta_{}".format("ttX")] = [np.cos(ttX.Theta())]
    event_data["phi_{}".format("ttX")] =      [ttX.Phi()]
    #event_data["mass_{}".format("ttX")] =     [ttX.M()]

    # save some extra fancy variables
    event_data["dEta_fn"] =     [np.sqrt( evt.get_dEta("Boson","hadTop")*evt.get_dEta("Boson","lepTop") )]
    event_data["dY_fn"] =       [np.sqrt( evt.get_dY("Boson","hadTop")*evt.get_dY("Boson","lepTop") )]
    event_data["dR_fn"] =       [np.sqrt( evt.get_dR("Boson","hadTop")*evt.get_dR("Boson","lepTop")  )]
    event_data["dPhi_fn"] =     [np.sqrt( evt.get_dPhi("Boson","hadTop")*evt.get_dPhi("Boson","lepTop") )]
    '''
    
    event_data = calculate_full_angles(event_data, evt)
    event_data = calculate_helicity_variables(event_data, evt)

    # create dataframe-type dictionary from variables
    df = pd.DataFrame.from_dict(event_data)
    return df



def calculate_full_angles(event_data, evt):
    # get some vectors
    hadtop = get_TLorentzVector(evt.objects["hadTop"])
    leptop = get_TLorentzVector(evt.objects["lepTop"])

    hadb = get_TLorentzVector(evt.objects["hadB"])
    lepb = get_TLorentzVector(evt.objects["lepB"])

    lepton = get_TLorentzVector(evt.objects["Lepton"])

    event_data["dXi_Lepton_hadB"] = lepton.Angle(hadb.Vect())
    event_data["dXi_hadTop_lepTop"] = hadtop.Angle(leptop.Vect())
    event_data["dXi_hadB_lepB"] = hadb.Angle(lepb.Vect())
    event_data["dXi_hadTop_hadB"] = hadtop.Angle(hadb.Vect())
    event_data["dXi_lepTop_lepB"] = leptop.Angle(lepb.Vect())
    event_data["dXi_Lepton_lepTop"] = lepton.Angle(leptop.Vect())

    event_data["dcosXi_Lepton_hadB"] = np.cos(lepton.Angle(hadb.Vect()))
    event_data["dcosXi_hadTop_lepTop"] = np.cos(hadtop.Angle(leptop.Vect()))
    event_data["dcosXi_hadB_lepB"] = np.cos(hadb.Angle(lepb.Vect()))
    event_data["dcosXi_hadTop_hadB"] = np.cos(hadtop.Angle(hadb.Vect()))
    event_data["dcosXi_lepTop_lepB"] = np.cos(leptop.Angle(lepb.Vect()))
    event_data["dcosXi_Lepton_lepTop"] = np.cos(lepton.Angle(leptop.Vect()))
    

    

    return event_data

def calculate_helicity_variables(event_data, evt):
    
    # get some tlorentzvectors
    hadtop = get_TLorentzVector(evt.objects["hadTop"])
    leptop = get_TLorentzVector(evt.objects["lepTop"])

    hadb = get_TLorentzVector(evt.objects["hadB"])
    hadw = get_TLorentzVector(evt.objects["hadW"])

    lepb = get_TLorentzVector(evt.objects["lepB"])

    lepton = get_TLorentzVector(evt.objects["Lepton"])

    ttbar_lepton = get_TLorentzVector(evt.objects["Lepton"])
    ttbar_lepb = get_TLorentzVector(evt.objects["lepB"])
    ttbar_hadb = get_TLorentzVector(evt.objects["hadB"])

    # ttbar system
    ttbar = hadtop+leptop
    boostvector_ttbar = ttbar.BoostVector()
    # hadt restframe
    boostvector_hadtop = hadtop.BoostVector()
    # lept restframe
    boostvector_leptop = leptop.BoostVector()

    # boost both tops into ttbar com frame
    hadtop.Boost(-boostvector_ttbar)
    xyz_hadtop = hadtop.Vect()
    leptop.Boost(-boostvector_ttbar)
    xyz_leptop = leptop.Vect()
    # boost lepton and bs in ttbar com frame
    ttbar_lepton.Boost(-boostvector_ttbar)
    ttbar_lepb.Boost(-boostvector_ttbar)
    ttbar_hadb.Boost(-boostvector_ttbar)

    # boost ttbar into ttbar com frame
    ttbar.Boost(-boostvector_ttbar)

    # boost hadb in hadtop restframe
    hadb.Boost(-boostvector_hadtop)
    xyz_hadb = hadb.Vect()
    hadw.Boost(-boostvector_hadtop)

    # boost lep and lepb in leptop restframes
    lepb.Boost(-boostvector_leptop)
    xyz_lepb = lepb.Vect()
    lepton.Boost(-boostvector_leptop)
    xyz_lepton = lepton.Vect()

    event_data["com_ttbar_pT_hadTop"] = hadtop.Pt()
    event_data["com_ttbar_pT_lepTop"] = leptop.Pt()

    event_data["com_ttbar_dXi_hadTop_hadB"] = hadtop.Angle(ttbar_hadb.Vect())
    event_data["com_ttbar_dXi_lepTop_lepB"] = leptop.Angle(ttbar_lepb.Vect())
    event_data["com_ttbar_dXi_lepTop_Lepton"] = leptop.Angle(ttbar_lepton.Vect())

    event_data["com_ttbar_dcosXi_hadTop_hadB"] = np.cos(hadtop.Angle(ttbar_hadb.Vect()))
    event_data["com_ttbar_dcosXi_lepTop_lepB"] = np.cos(leptop.Angle(ttbar_lepb.Vect()))
    event_data["com_ttbar_dcosXi_lepTop_Lepton"] = np.cos(leptop.Angle(ttbar_lepton.Vect()))

    event_data["com_hadTop_pT_hadB"] = hadb.Pt()
    event_data["com_hadTop_P_hadB"] = hadb.P()

    event_data["com_hadTop_pT_hadW"] = hadw.Pt()
    event_data["com_hadTop_P_hadW"] = hadw.P()

    event_data["com_lepTop_pT_lepB"] = lepb.Pt()
    event_data["com_lepTop_pT_Lepton"] = lepton.Pt()

    # helicity angles
    ha_lepton = (xyz_lepton*xyz_leptop)/(xyz_lepton.Mag()*xyz_leptop.Mag())
    event_data["HA_Lepton"] = ha_lepton
    ha_hadb = (xyz_hadb*xyz_hadtop)/(xyz_hadb.Mag()*xyz_hadtop.Mag())
    event_data["HA_hadB"] = ha_hadb
    ha_lepb = (xyz_lepb*xyz_leptop)/(xyz_lepb.Mag()*xyz_leptop.Mag())
    event_data["HA_lepB"] = ha_lepb

    event_data["HA_hadB_lepB"] = ha_lepb*ha_hadb

    #event_data["com_ttbar_dXi_hadTop_lepTop"] = hadtop.Angle(leptop.Vect())
    event_data["com_ttbar_dEta_hadTop_lepTop"] = abs(hadtop.Eta() - leptop.Eta())

    dxi = hadb.Angle(lepb.Vect())
    event_data["HF_dXi_hadB_lepB"] = dxi
    event_data["HF_dcosXi_hadB_lepB"] = np.cos(dxi)

    event_data["HF_dEta_hadB_lepB"] = abs(hadb.Eta() - lepb.Eta())
    dphi = abs(hadb.Phi() - lepb.Phi())
    if dphi > np.pi: dphi = 2.*np.pi-dphi
    event_data["HF_dPhi_hadB_lepB"] = dphi

    return event_data

def get_TLorentzVector(object):
    lv = object.p4()
    return ROOT.TLorentzVector(lv.Px(), lv.Py(), lv.Pz(), lv.E())
    
    
