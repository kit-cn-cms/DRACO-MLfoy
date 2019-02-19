
def calculateVariables(evt):
    object_pairs = [
        ["hadTop", "lepTop"],
        ["hadTop", "hadB"],
        ["lepTop", "lepB"],
        ["Lepton", "lepTop"],
        ]
    for obj in object_pairs:
        evt.variables["dEta_{}_{}".format(obj[0],obj[1])] =        [evt.get_dEta(obj[0],obj[1])]
        evt.variables["dY_{}_{}".format(obj[0],obj[1])] =          [evt.get_dY(obj[0],obj[1])]
        evt.variables["dTheta_{}_{}".format(obj[0],obj[1])] =      [evt.get_dTheta(obj[0],obj[1])]
        evt.variables["dcosTheta_{}_{}".format(obj[0],obj[1])] =   [evt.get_dcosTheta(obj[0],obj[1])]
        evt.variables["dPhi_{}_{}".format(obj[0],obj[1])] =        [evt.get_dPhi(obj[0],obj[1])]
        evt.variables["dR_{}_{}".format(obj[0],obj[1])] =          [evt.get_dR(obj[0],obj[1])]


