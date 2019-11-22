import numpy as np

dnnmodels = {

	"DNN2016" : {
			"data_era" 	: ["2016"],
			"DNNs"		: ["/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir/rt1_ge4j_ge4t_odd2016",
						 	"/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir/rt2_ge4j_ge4t_odd2016",
							"/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir/rt3_ge4j_ge4t_odd2016",
							"/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir/rt4_ge4j_ge4t_odd2016",
							"/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir/rt5_ge4j_ge4t_odd2016"],
			"binning"	: {
							"ttH": np.array([ 0.2, 0.34, 0.3867, 0.4333, 0.48, 0.5267, 0.5733, 0.62, 0.6667, 0.7133, 0.9 ]),
							"ttmb": np.array([0.2, 0.8]),
							"tt2b": np.array([0.2,0.63]),
							"ttcc": np.array([0.2,0.48]),
							"ttlf": np.array([0.2,0.69])
							}
				},

	"DNN2017" : {
			"data_era" 	: ["2017"],
			"DNNs"		: ["/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir/rt1_ge4j_ge4t_odd2017",
						 	"/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir/rt2_ge4j_ge4t_odd2017",
							"/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir/rt3_ge4j_ge4t_odd2017",
							"/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir/rt4_ge4j_ge4t_odd2017",
							"/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir/rt5_ge4j_ge4t_odd2017"],
			"binning"	: {
							"ttH": np.array([ 0.2,0.292,0.338,0.384,0.43,0.476,0.522,0.568,0.614,0.66,0.89 ]),
							"ttmb": np.array([0.2, 0.83]),
							"tt2b": np.array([0.2,0.62]),
							"ttcc": np.array([0.2,0.48]),
							"ttlf": np.array([0.2,0.7])
							}
				},


	"DNN2018" : {
			"data_era" 	: ["2018"],
			"DNNs"		: ["/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir/rt1_ge4j_ge4t_odd2018",
						 	"/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir/rt2_ge4j_ge4t_odd2018",
							"/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir/rt3_ge4j_ge4t_odd2018",
							"/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir/rt4_ge4j_ge4t_odd2018",
							"/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir/rt5_ge4j_ge4t_odd2018"],
			"binning"	: {
							"ttH": np.array([ 0.2, 0.2893, 0.334, 0.3787, 0.4233, 0.468, 0.5127, 0.5573, 0.602, 0.6467, 0.6913, 0.736, 0.87 ]),
							"ttmb": np.array([0.2, 0.84]),
							"tt2b": np.array([0.2,0.67]),
							"ttcc": np.array([0.2,0.45]),
							"ttlf": np.array([0.2,0.72])
							}
				},

	"DNNcombined" : {
			# "data_era" 	: ["2016"],
			"data_era" 	: ["2018","2016","2017"],
			"DNNs"		: ["/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir/rt1_ge4j_ge4t_odd",
						 	"/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir/rt2_ge4j_ge4t_odd",
							"/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir/rt3_ge4j_ge4t_odd",
							"/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir/rt4_ge4j_ge4t_odd",
							"/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir/rt5_ge4j_ge4t_odd"],
			"binning"	: {
							"ttH": np.array([ 0.2, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.95 ]),
							"ttmb": np.array([0.2, 0.86]),
							"tt2b": np.array([0.2,0.58]),
							"ttcc": np.array([0.2,0.47]),
							"ttlf": np.array([0.2,0.71])
							}
				},

	"DNN1718" : {
			"data_era" 	: ["2018","2017"],
			"DNNs"		: ["/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir/rt1_ge4j_ge4t_odd2017_2018",
						 	"/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir/rt2_ge4j_ge4t_odd2017_2018",
							"/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir/rt3_ge4j_ge4t_odd2017_2018",
							"/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir/rt4_ge4j_ge4t_odd2017_2018",
							"/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir/rt5_ge4j_ge4t_odd2017_2018"],
			"binning"	: {
							"ttH": np.array([ 0.1754, 0.3232, 0.3725, 0.4218, 0.4711, 0.5204, 0.5696, 0.6189, 0.6682, 0.7175, 0.89 ]),
							"ttmb": np.array([0.1775, 0.83]),
							"tt2b": np.array([0.1843,0.64]),
							"ttcc": np.array([0.1893,0.5]),
							"ttlf": np.array([0.1825,0.69])
							}
				},
}

outputDirectory="/home/lreuter/Documents/hiwi/DRACO-MLfoy/workdir"
inputData="/home/lreuter/Documents/hiwi/Robusttest/"
shuffleSeed=59074
signal_class = "ttH"
