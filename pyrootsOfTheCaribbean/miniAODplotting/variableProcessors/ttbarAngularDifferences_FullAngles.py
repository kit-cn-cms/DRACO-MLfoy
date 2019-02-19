import ROOT
import numpy as np
def calculateVariables(evt):
    hadtop = get_TLorentzVector(evt.objects["hadTop"])
    leptop = get_TLorentzVector(evt.objects["lepTop"])

    hadb = get_TLorentzVector(evt.objects["hadB"])
    lepb = get_TLorentzVector(evt.objects["lepB"])

    lepton = get_TLorentzVector(evt.objects["Lepton"])

    evt.variables["dXi_Lepton_hadB"] = lepton.Angle(hadb.Vect())
    evt.variables["dXi_hadTop_lepTop"] = hadtop.Angle(leptop.Vect())
    evt.variables["dXi_hadB_lepB"] = hadb.Angle(lepb.Vect())
    evt.variables["dXi_hadTop_hadB"] = hadtop.Angle(hadb.Vect())
    evt.variables["dXi_lepTop_lepB"] = leptop.Angle(lepb.Vect())
    evt.variables["dXi_Lepton_lepTop"] = lepton.Angle(leptop.Vect())

    evt.variables["dcosXi_Lepton_hadB"] = np.cos(lepton.Angle(hadb.Vect()))
    evt.variables["dcosXi_hadTop_lepTop"] = np.cos(hadtop.Angle(leptop.Vect()))
    evt.variables["dcosXi_hadB_lepB"] = np.cos(hadb.Angle(lepb.Vect()))
    evt.variables["dcosXi_hadTop_hadB"] = np.cos(hadtop.Angle(hadb.Vect()))
    evt.variables["dcosXi_lepTop_lepB"] = np.cos(leptop.Angle(lepb.Vect()))
    evt.variables["dcosXi_Lepton_lepTop"] = np.cos(lepton.Angle(leptop.Vect()))


def get_TLorentzVector(object):
    lv = object.p4()
    return ROOT.TLorentzVector(lv.Px(), lv.Py(), lv.Pz(), lv.E())




