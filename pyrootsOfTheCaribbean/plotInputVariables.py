import ROOT
import os
import sys
import ROOT
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

import variable_sets.ntuplesVariablesWithIndex as variable_set
from evaluationScripts.plotVariables import variablePlotter

# location of input dataframes
data_dir = "/ceph/vanderlinden/MLFoyTrainData/DNN_ttZ_v2/"

# output location of plots
plot_dir = "/ceph/vanderlinden/ttZ_2019/plots_ttZ/"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# plotting options
plotOptions = {
    "ratio":        True,
    "ratioTitle":   "#frac{ttZ}{ttH}",
    "logscale":     False,
    "scaleSignal":  -1,
    "lumiScale":    1,
    "KSscore":      True,
    }
#   scaleSignal:
#   -1:     scale to background Integral
#   float:  scale with float value
#   False:  dont scale

# additional variables to plot
additional_variables = [
    ]


# initialize plotter
plotter = variablePlotter(
    output_dir      = plot_dir,
    variable_set    = variable_set,
    add_vars        = additional_variables,
    plotOptions     = plotOptions
    )

# add samples
plotter.addSample(
    sampleName      = "ttH",
    sampleFile      = data_dir+"/ttHbb_dnn.h5",
    plotColor       = ROOT.kOrange,
    signalSample    = False)


plotter.addSample(
    sampleName      = "ttZ",
    sampleFile      = data_dir+"/ttZbb_dnn.h5",
    plotColor       = ROOT.kGreen+1,
    signalSample    = True)

'''
plotter.addSample(
    sampleName      = "ttbar",
    sampleFile      = data_dir+"/ttbar_dnn.h5",
    signalSample    = False)

plotter.addSample(
    sampleName      = "ttbb",
    sampleFile      = data_dir+"/ttbb_dnn.h5",
    plotColor       = ROOT.kRed+3)

plotter.addSample(
    sampleName      = "tt2b",
    sampleFile      = data_dir+"/tt2b_dnn.h5",
    plotColor       = ROOT.kRed+2)

plotter.addSample(
    sampleName      = "ttb",
    sampleFile      = data_dir+"/ttb_dnn.h5",
    plotColor       = ROOT.kRed-2)

plotter.addSample(
    sampleName      = "ttcc",
    sampleFile      = data_dir+"/ttcc_dnn.h5",
    plotColor       = ROOT.kRed+1)

plotter.addSample(
    sampleName      = "ttlf",
    sampleFile      = data_dir+"/ttlf_dnn.h5")
'''


# add JT categories
plotter.addCategory("4j_ge3t")
plotter.addCategory("5j_ge3t")
plotter.addCategory("ge6j_ge3t")
plotter.addCategory("ge4j_ge3t")


# perform plotting routine
plotter.plot()
