import os
import sys
import ROOT
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

import variable_sets.dnnVariableSet as variable_set
from evaluationScripts.plotVariables import variablePlotter

# location of input dataframes
data_dir = "/nfs/dust/cms/user/vdlinden/MLfoy/pyrootsOfTheCaribbean/miniAODGenLevelData/"

# output location of plots
plot_dir = "/nfs/dust/cms/user/vdlinden/MLfoy/pyrootsOfTheCaribbean/miniAODGenLevelData/plots/"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# plotting options
plotOptions = {
    "ratio":        False,
    "logscale":     False,
    "scaleSignal":  -1,
    "lumiScale":    1
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
    variable_set    = None,
    add_vars        = additional_variables,
    plotOptions     = plotOptions
    )

# add samples
plotter.addSample(
    sampleName      = "ttH",
    sampleFile      = data_dir+"/ttH.h5",
    signalSample    = False,
    plotColor       = ROOT.kYellow,
    apply_cut       = False)

plotter.addSample(
    sampleName      = "ttZ",
    sampleFile      = data_dir+"/ttZ.h5",
    signalSample    = True,
    plotColor       = ROOT.kBlack,
    apply_cut       = False)


# add JT categories
plotter.addCategory("ge4j_ge3t")

# perform plotting routine
plotter.plot()
