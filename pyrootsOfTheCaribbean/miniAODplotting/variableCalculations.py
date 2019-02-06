# import ROOT in batch mode
import sys
import ROOT
import numpy as np
import pandas as pd
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
    objects = ["Boson", "Lepton", "hadTop", "lepTop", "hadB", "lepB"]

    # loop over all the objects
    for i, obj1 in enumerate(objects):
        # get interesting variables for these objects

        # transverse momentum
        event_data["pT_{}".format(obj1)] =          [evt.get_pT(obj1)]
        # pseudo rapidity
        event_data["eta_{}".format(obj1)] =         [evt.get_eta(obj1)]
        # rapidity
        event_data["y_{}".format(obj1)] =           [evt.get_y(obj1)]
        # cosine of azimuth angle
        #event_data["theta_{}".format(obj1)] =       [evt.get_theta(obj1)]
        #event_data["costheta_{}".format(obj1)] =    [evt.get_costheta(obj1)]
        # polar angle
        event_data["phi_{}".format(obj1)] =         [evt.get_phi(obj1)]
        # mass of particle
        event_data["mass_{}".format(obj1)] =        [evt.get_mass(obj1)]

        # loop over all the objects again for delta variables
        for j, obj2 in enumerate(objects):
            if j <= i: continue
            # pseudo rapidity difference
            event_data["dEta_{}_{}".format(obj1,obj2)] =        [evt.get_dEta(obj1,obj2)]
            # rapidity difference
            event_data["dY_{}_{}".format(obj1,obj2)] =          [evt.get_dY(obj1,obj2)]
            # difference in cosine of azimuth angle
            #event_data["dTheta_{}_{}".format(obj1,obj2)] =      [evt.get_dTheta(obj1,obj2)]
            #event_data["dcosTheta_{}_{}".format(obj1,obj2)] =   [evt.get_dcosTheta(obj1,obj2)]
            # difference in polar angle
            event_data["dPhi_{}_{}".format(obj1,obj2)] =        [evt.get_dPhi(obj1,obj2)]
            # spacial difference in R = sqrt(dEta^2+dPhi^2)
            event_data["dR_{}_{}".format(obj1,obj2)] =          [evt.get_dR(obj1,obj2)]

    # get the invariant ttbar system by adding both the top quarks
    ttbar = evt.objects["hadTop"].p4() + evt.objects["lepTop"].p4()   
    
    # save kinematic variables for ttbar system
    event_data["pT_{}".format("ttbar")] =       [ttbar.Pt()]
    event_data["eta_{}".format("ttbar")] =      [ttbar.Eta()]
    event_data["y_{}".format("ttbar")] =        [ttbar.Rapidity()]
    #event_data["theta_{}".format("ttbar")] =    [ttbar.Theta()]
    #event_data["costheta_{}".format("ttbar")] = [np.cos(ttbar.Theta())]
    event_data["phi_{}".format("ttbar")] =      [ttbar.Phi()]
    #event_data["mass_{}".format("ttbar")] =     [ttbar.M()]

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

    

    # create dataframe-type dictionary from variables
    df = pd.DataFrame.from_dict(event_data)
    return df
