from ROOT import PyConfig, gROOT
PyConfig.IgnoreCommandLineOptions = True
gROOT.SetBatch(True)
import os
import sys
import ROOT
import optparse
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

import variable_sets.dnnVariableSet as variable_set
from evaluationScripts.plotVariables import variablePlotter


parser = optparse.OptionParser(usage="%prog [options]")
parser.add_option("-i","--input",dest="input_dir",metavar="INPUT",
    default = "/miniAODGenLevelData/ttbar/",
    help = "input directory relative to workdir")
parser.add_option("-p","--plots",dest="plot_dir",metavar="PLOTDIR",
    default = "/plots/",
    help = "directory for plots, relative to INPUT")
parser.add_option("--plotRatio",dest="ratio",action="store_true",default=False,metavar="RATIO",
    help = "add ratio plot to plots")
parser.add_option("--logScale",dest="logscale",action="store_true",default=False,metavar="LOGSCALE",
    help = "use logarithmic y-axis")
parser.add_option("--writeKS",dest="ksscore",action="store_true",default=False,metavar="KSSCORE",
    help = "write KS scores on the canvas")
parser.add_option("-m","--maxEvents",dest="max_events",default=None,metavar="MAXEVENTS",
    help = "maximum number of events per sample")
(opts, args) = parser.parse_args()


# location of input dataframes
data_dir = "/nfs/dust/cms/user/vdlinden/MLfoy/workdir/"+opts.input_dir+"/"
#"/nfs/dust/cms/user/vdlinden/MLfoy/workdir/miniAODGenLevelData/"


# output location of plots
plot_dir = data_dir+"/"+opts.plot_dir+"/"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# plotting options
plotOptions = {
    "ratio":        opts.ratio,
    "ratioTitle":   "#frac{X}{ttZ(qq)}",
    "logscale":     opts.logscale,
    "scaleSignal":  -1,
    "lumiScale":    1.0,
    "KSscore":      opts.ksscore
    }
#   scaleSignal:
#   -1:     scale to background Integral
#   float:  scale with float value
#   False:  dont scale

# additional variables to plot
additional_variables = [
    ]

ignoredVariables = [
    "Weight_XS",
    "Weight_GEN_nom",
    ]

# initialize plotter
plotter = variablePlotter(
    output_dir      = plot_dir,
    variable_set    = None,
    add_vars        = additional_variables,
    max_entries     = opts.max_events,
    plotOptions     = plotOptions
    )

# add samples
plotter.addSample(
    sampleName      = "ttZ(qq)",
    sampleFile      = data_dir+"/ttZqq.h5",
    signalSample    = False,
    plotColor       = ROOT.kYellow,
    apply_cut       = True)

#plotter.addSample(
#    sampleName      = "ttZ(ll)",
#    sampleFile      = data_dir+"/ttZll.h5",
#    signalSample    = True,
#    plotColor       = ROOT.kBlack,
#    apply_cut       = True)

plotter.addSample(
    sampleName      = "ttH(bb)",
    sampleFile      = data_dir+"/ttHbb.h5",
    signalSample    = True,
    plotColor       = ROOT.kRed)

#plotter.addSample(
#    sampleName      = "ttZJets",
#    sampleFile      = data_dir+"/ttZJets.h5",
#    signalSample    = True,
#    plotColor       = ROOT.kGreen,
#    apply_cut       = True)

plotter.addSample(
    sampleName      = "ttSL",
    sampleFile      = data_dir+"/ttSL.h5",
    signalSample    = True,
    plotColor       = ROOT.kBlack)

# add JT categories
plotter.addCategory("SL")
#plotter.addCategory("ge5j_ge2t")
#plotter.addCategory("ge4j_ge3t")
plotter.addCategory("4j_ge3t")
plotter.addCategory("5j_ge3t")
plotter.addCategory("ge6j_ge3t")


# perform plotting routine
plotter.plot(ignoredVariables)
