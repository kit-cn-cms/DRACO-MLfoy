import ROOT
import numpy as np
def calculateVariables(evt):

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

    evt.variables["com_ttbar_pT_hadTop"] = hadtop.Pt()
    evt.variables["com_ttbar_pT_lepTop"] = leptop.Pt()

    evt.variables["com_ttbar_dXi_hadTop_hadB"] = hadtop.Angle(ttbar_hadb.Vect())
    evt.variables["com_ttbar_dXi_lepTop_lepB"] = leptop.Angle(ttbar_lepb.Vect())
    evt.variables["com_ttbar_dXi_lepTop_Lepton"] = leptop.Angle(ttbar_lepton.Vect())

    evt.variables["com_ttbar_dcosXi_hadTop_hadB"] = np.cos(hadtop.Angle(ttbar_hadb.Vect()))
    evt.variables["com_ttbar_dcosXi_lepTop_lepB"] = np.cos(leptop.Angle(ttbar_lepb.Vect()))
    evt.variables["com_ttbar_dcosXi_lepTop_Lepton"] = np.cos(leptop.Angle(ttbar_lepton.Vect()))

    evt.variables["com_hadTop_pT_hadB"] = hadb.Pt()
    #evt.variables["com_hadTop_P_hadB"] = hadb.P()

    #evt.variables["com_hadTop_pT_hadW"] = hadw.Pt()
    #evt.variables["com_hadTop_P_hadW"] = hadw.P()

    evt.variables["com_lepTop_pT_lepB"] = lepb.Pt()
    evt.variables["com_lepTop_pT_Lepton"] = lepton.Pt()

    # helicity angles
    ha_lepton = (xyz_lepton*xyz_leptop)/(xyz_lepton.Mag()*xyz_leptop.Mag())
    evt.variables["HA_Lepton"] = ha_lepton
    ha_hadb = (xyz_hadb*xyz_hadtop)/(xyz_hadb.Mag()*xyz_hadtop.Mag())
    evt.variables["HA_hadB"] = ha_hadb
    ha_lepb = (xyz_lepb*xyz_leptop)/(xyz_lepb.Mag()*xyz_leptop.Mag())
    evt.variables["HA_lepB"] = ha_lepb

    evt.variables["HA_hadB_lepB"] = ha_hadb*ha_lepb
    evt.variables["HA_hadB_Lepton"] = ha_hadb*ha_lepton

    evt.variables["com_ttbar_dEta_hadTop_lepTop"] = abs(hadtop.Eta() - leptop.Eta())

    dxi = hadb.Angle(lepb.Vect())
    evt.variables["HF_dXi_hadB_lepB"] = dxi
    evt.variables["HF_dcosXi_hadB_lepB"] = np.cos(dxi)

    evt.variables["HF_dEta_hadB_lepB"] = abs(hadb.Eta() - lepb.Eta())
    dphi = abs(hadb.Phi() - lepb.Phi())
    if dphi > np.pi: dphi = 2.*np.pi-dphi
    evt.variables["HF_dPhi_hadB_lepB"] = dphi



def get_TLorentzVector(object):
    lv = object.p4()
    return ROOT.TLorentzVector(lv.Px(), lv.Py(), lv.Pz(), lv.E())

