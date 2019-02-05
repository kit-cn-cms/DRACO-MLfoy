import os
import sys
import ROOT
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

import variable_sets.dnnVariableSet as variable_set
from evaluationScripts.plotVariables import variablePlotter2D

# location of input dataframes
data_dir = "/nfs/dust/cms/user/vdlinden/MLfoy/workdir/miniAODGenLevelData/ttbarSystem/"
#"/nfs/dust/cms/user/vdlinden/MLfoy/workdir/miniAODGenLevelData/"

# output location of plots
plot_dir = data_dir+"/plots2D/"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# plotting options
plotOptions = {
    "logscale":     False,
    }

# list of variables to plot
objects = ["Lepton", "hadTop", "hadB", "lepTop", "lepB"]

variables = [
    ]+[
    ["eta_{}".format(obj),"pT_{}".format(obj)] for obj in objects
    ]+[
    ["eta_{}".format(obj),"y_{}".format(obj)] for obj in objects
    ]+[
    ["phi_{}".format(obj1),"phi_{}".format(obj2)] for i,obj1 in enumerate(objects) for j,obj2 in enumerate(objects) if i<j
    ]+[
    ["y_{}".format(obj1),"y_{}".format(obj2)] for i,obj1 in enumerate(objects) for j,obj2 in enumerate(objects) if i<j
    ]+[
    ["eta_{}".format(obj1),"eta_{}".format(obj2)] for i,obj1 in enumerate(objects) for j,obj2 in enumerate(objects) if i<j
    ]+[
    ["pT_{}".format(obj1),"pT_{}".format(obj2)] for i,obj1 in enumerate(objects) for j,obj2 in enumerate(objects) if i<j
    ]

# initialize plotter
plotter = variablePlotter2D(
    output_dir      = plot_dir,
    variable_set    = None,
    add_vars        = variables,
    plotOptions     = plotOptions
    )

# add samples
'''
plotter.addSample(
    sampleName      = "ttZ(qq)",
    sampleFile      = data_dir+"/ttZqq.h5",
    apply_cut       = False)
'''
plotter.addSample(
    sampleName      = "ttZJets",
    sampleFile      = data_dir+"/ttZJets.h5",
    apply_cut       = False)

#plotter.addSample(
#    sampleName      = "ttH(bb)",
#    sampleFile      = data_dir+"/ttHbb.h5",
#    apply_cut       = False)

plotter.addSample(
    sampleName      = "ttbar",
    sampleFile      = data_dir+"/ttSL.h5",
    apply_cut       = False)


# add JT categories
plotter.addCategory("SL")
#plotter.addCategory("ge4j_ge3t")
#plotter.addCategory("4j_ge3t")
#plotter.addCategory("5j_ge3t")
#plotter.addCategory("ge6j_ge3t")

# perform plotting routine
plotter.plot()
