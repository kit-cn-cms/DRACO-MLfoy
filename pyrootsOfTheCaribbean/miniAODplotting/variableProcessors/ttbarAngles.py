

def calculateVariables(evt):
    objects = ["Lepton", "hadTop", "lepTop", "hadB", "lepB"]
    for obj in objects:
        evt.variables["pT_{}".format(obj)]          = [evt.get_pT(obj)]
        evt.variables["eta_{}".format(obj)]         = [evt.get_eta(obj)]
        evt.variables["phi_{}".format(obj)]         = [evt.get_phi(obj)]
        evt.variables["y_{}".format(obj)]           = [evt.get_y(obj)]
        evt.variables["theta_{}".format(obj)]       = [evt.get_theta(obj)]
        evt.variables["costheta_{}".format(obj)]    = [evt.get_costheta(obj)]
