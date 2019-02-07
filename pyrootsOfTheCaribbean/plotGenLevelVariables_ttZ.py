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
data_dir = "/nfs/dust/cms/user/vdlinden/MLfoy/workdir/miniAODGenLevelData/ttZSystem/"
#"/nfs/dust/cms/user/vdlinden/MLfoy/workdir/miniAODGenLevelData/"

# output location of plots
plot_dir = data_dir+"/plots/"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# plotting options
plotOptions = {
    "ratio":        True,
    "ratioTitle":   "#frac{X}{ttZ(qq)}",
    "logscale":     False,
    "scaleSignal":  -1,
    "lumiScale":    1.0,
    "KSscore":      True
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
    sampleName      = "ttZ(qq)",
    sampleFile      = data_dir+"/ttZqq.h5",
    signalSample    = False,
    plotColor       = ROOT.kYellow)

plotter.addSample(
    sampleName      = "ttZ(ll)",
    sampleFile      = data_dir+"/ttZll.h5",
    signalSample    = True,
    plotColor       = ROOT.kBlack)

plotter.addSample(
    sampleName      = "ttZJets",
    sampleFile      = data_dir+"/ttZJets.h5",
    signalSample    = True,
    plotColor       = ROOT.kRed)

# add JT categories
plotter.addCategory("SL")
plotter.addCategory("1l_ge4j_ge3t")
plotter.addCategory("1l_4j_ge3t")
plotter.addCategory("1l_5j_ge3t")
plotter.addCategory("1l_ge6j_ge3t")

# perform plotting routine
plotter.plot()
