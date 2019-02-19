import numpy as np
def calculateVariables(evt):
    ttbar = evt.objects["hadTop"].p4() + evt.objects["lepTop"].p4()
    boson = boson = evt.objects["Boson"].p4()
    ttX   = ttbar + boson

    evt.variables["pT_{}".format("ttbar")] =       [ttbar.Pt()]
    evt.variables["eta_{}".format("ttbar")] =      [ttbar.Eta()]
    evt.variables["y_{}".format("ttbar")] =        [ttbar.Rapidity()]
    evt.variables["theta_{}".format("ttbar")] =    [ttbar.Theta()]
    evt.variables["costheta_{}".format("ttbar")] = [np.cos(ttbar.Theta())]
    evt.variables["phi_{}".format("ttbar")] =      [ttbar.Phi()]

    evt.variables["pT_{}".format("Boson")] =       [ttbar.Pt()]
    evt.variables["eta_{}".format("Boson")] =      [ttbar.Eta()]
    evt.variables["y_{}".format("Boson")] =        [ttbar.Rapidity()]
    evt.variables["theta_{}".format("Boson")] =    [ttbar.Theta()]
    evt.variables["costheta_{}".format("Boson")] = [np.cos(ttbar.Theta())]
    evt.variables["phi_{}".format("Boson")] =      [ttbar.Phi()]

    evt.variables["pT_{}".format("ttX")] =         [ttX.Pt()]
    evt.variables["eta_{}".format("ttX")] =        [ttX.Eta()]
    evt.variables["y_{}".format("ttX")] =          [ttX.Rapidity()]
    evt.variables["theta_{}".format("ttX")] =      [ttX.Theta()]
    evt.variables["costheta_{}".format("ttX")] =   [np.cos(ttX.Theta())]
    evt.variables["phi_{}".format("ttX")] =        [ttX.Phi()]


    deta = np.abs(boson.Eta()-ttbar.Eta())
    evt.variables["dEta_{}_{}".format("Boson","ttbar")] =      [deta]
    evt.variables["dTheta_{}_{}".format("Boson","ttbar")] =    [np.abs(boson.Theta() - ttbar.Theta())]
    evt.variables["dcosTheta_{}_{}".format("Boson","ttbar")] = [np.abs(np.cos(boson.Theta())-np.cos(ttbar.Theta()))]
    evt.variables["dY_{}_{}".format("Boson","ttbar")] =        [np.abs(boson.Rapidity()-ttbar.Rapidity())]
    dphi = np.abs(boson.Phi()-ttbar.Phi())
    if dphi > np.pi: dphi = 2.*np.pi-dphi
    evt.variables["dPhi_{}_{}".format("Boson","ttbar")] =      [dphi]
    evt.variables["dR_{}_{}".format("Boson","ttbar")] =        [np.sqrt(deta**2+dphi**2)]

    evt.variables["dEta_fn"] =     [np.sqrt( evt.get_dEta("Boson","hadTop")*evt.get_dEta("Boson","lepTop") )]
    evt.variables["dY_fn"] =       [np.sqrt( evt.get_dY("Boson","hadTop")*evt.get_dY("Boson","lepTop") )]
    evt.variables["dR_fn"] =       [np.sqrt( evt.get_dR("Boson","hadTop")*evt.get_dR("Boson","lepTop")  )]
    evt.variables["dPhi_fn"] =     [np.sqrt( evt.get_dPhi("Boson","hadTop")*evt.get_dPhi("Boson","lepTop") )]



