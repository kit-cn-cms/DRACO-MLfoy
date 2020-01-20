variables = {}

variables["ge4j_3t"] = [
    "Evt_M2_TaggedJetsAverage",
    "Evt_M2_minDrTaggedJets",
    "Evt_Pt_minDrTaggedJets",
    "Evt_M2_closestTo91TaggedJets",
    "Evt_M2_closestTo125TaggedJets",
    ]


variables["5j_ge3t"] = [
    "Evt_Pt_minDrTaggedJets",
    "Evt_Dr_closestTo91TaggedJets",
    "Evt_M2_minDrTaggedJets",
    "Evt_M2_closestTo91TaggedJets",
    "Evt_M2_closestTo125TaggedJets",
    ]


variables["ge6j_ge3t"] = [
    "Evt_Dr_closestTo91TaggedJets",
    "Evt_M2_minDrTaggedJets",
    "Evt_Pt_minDrTaggedJets",
    "Evt_M2_closestTo125TaggedJets",
    "Evt_M2_closestTo91TaggedJets",
    ]


variables["ge4j_ge4t"] = [
    "Evt_Deta_TaggedJetsAverage",
    "Evt_M2_minDrTaggedJets",
    "Evt_Pt_minDrTaggedJets",
    "Evt_M2_closestTo91TaggedJets",
    "Evt_M2_closestTo125TaggedJets",
    ]


variables["4j_ge3t"] = [
    "Evt_Deta_TaggedJetsAverage",
    "Evt_M2_minDrJets",
    "Evt_M2_minDrTaggedJets",
    "Evt_M2_closestTo125TaggedJets",
    "Evt_M2_closestTo91TaggedJets",
    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))
