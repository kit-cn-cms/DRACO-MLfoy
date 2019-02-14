
def calculateVariables(evt):
    objects = ["Lepton", "hadB", "lepB"]
    for i, obj1 in enumerate(objects):
        for j, obj2 in enumerate(objects):
            if j<=i: continue
            
            evt.variables["dEta_{}_{}".format(obj1,obj2)] =        [evt.get_dEta(obj1,obj2)]
            evt.variables["dY_{}_{}".format(obj1,obj2)] =          [evt.get_dY(obj1,obj2)]
            evt.variables["dTheta_{}_{}".format(obj1,obj2)] =      [evt.get_dTheta(obj1,obj2)]
            evt.variables["dcosTheta_{}_{}".format(obj1,obj2)] =   [evt.get_dcosTheta(obj1,obj2)]
            evt.variables["dPhi_{}_{}".format(obj1,obj2)] =        [evt.get_dPhi(obj1,obj2)]
            evt.variables["dR_{}_{}".format(obj1,obj2)] =          [evt.get_dR(obj1,obj2)]

