variables = {}
variables["inclusive"] = [
    "MET_pt_nom",
    "rec_top_m",
    "rec_delta_phi_top_leadingbjet",
    "wolframh3_30",
    "rec_delta_eta_top_from_subleadingbjet_leadingbjet",
    "rec_delta_eta_top_bjet",
    "rec_delta_phi_bjet_bjet",
    "rec_lep_pt",
    "rec_delta_eta_lep_leadingbjet",
    "rec_M_lep_subleadingbjet",
    "rec_sum_pt_bjet_bjet",
    "rec_costhetastar",
    "sumHtTotal_tag",
    "rec_lep_charge",
    "rec_leading_bjet_pt",
    "rec_leading_bjet_eta",
    "rec_subleading_bjet_pt",
    "rec_subleading_bjet_eta",
    ]

all_variables = list(set( [v for key in variables for v in variables[key] ] ))

