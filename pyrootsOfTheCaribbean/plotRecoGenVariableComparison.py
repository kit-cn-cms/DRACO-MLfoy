import ROOT
import os
import sys
import ROOT
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

import variable_sets.ttXRecoVariables as variable_set
from evaluationScripts.plotVariables import variablePlotterGenReco

# location of input dataframes
#data_dir = "/ceph/vanderlinden/MLFoyTrainData/DNN_ttZ_v3/"
data_dir = "/nfs/dust/cms/user/vdlinden/DNNInputFiles/ttZ_DNN_v3/"

# output location of plots
plot_dir = basedir + "/workdir/RecoGenComparison/"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# plotting options
plotOptions = {
    "plot1D":       True,
    "ratio":        True,
    "ratioTitle":   "#frac{gen}{reco}",
    "KSscore":      True,
    "logscale":     False,
    "lumiScale":    1.,
    }

# variables to plot
variables = [
    "DeltaEta_BHad_BLep",
    "DeltaEta_BHad_Lepton",
    "DeltaEta_TopHad_TopLep",
    "DeltaPhi_BHad_BLep",
    "DeltaPhi_BHad_Lepton",
    "DeltaPhi_TopHad_TopLep",
    #"DeltaTheta_BHad_BLep",
    #"DeltaTheta_BHad_Lepton",
    #"DeltaTheta_TopHad_TopLep",
    "Eta_BHad",
    "Eta_BLep",
    "Eta_Lepton",
    "Eta_TopHad",
    "Eta_TopLep",
    "Eta_WHad",
    "Eta_WLep",
    #"Theta_BHad",
    #"Theta_BLep",
    #"Theta_Lepton",
    #"Theta_TopHad",
    #"Theta_TopLep",
    #"Theta_WHad",
    #"Theta_WLep",
    "pT_BHad",
    "pT_BLep",
    "pT_Lepton",
    "pT_TopHad",
    "pT_TopLep",
    "pT_WHad",
    "pT_WLep",
    ]

variable_pairs = {v: ["genTTX_"+v, "recoTTX_"+v] for v in variables }


# initialize plotter
plotter = variablePlotterGenReco(
    output_dir      = plot_dir,
    variable_pairs  = variable_pairs,
    plotOptions     = plotOptions,
    )

# add samples
plotter.addSample(
    sampleName      = "ttH",
    sampleFile      = data_dir+"/ttHbb_dnn.h5",
    plotColor       = ROOT.kBlue+1,
    XSscaling       = 2.,) # due to even/odd splitting

'''
plotter.addSample(
    sampleName      = "ttZ",
    sampleFile      = data_dir+"/ttZbb_dnn.h5",
    plotColor       = ROOT.kGreen)
'''

plotter.addSample(
    sampleName      = "ttZJets",
    sampleFile      = data_dir+"/ttZJets_dnn.h5",
    plotColor       = ROOT.kGreen-2)

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
    sampleFile      = data_dir+"/ttlf_dnn.h5",
    plotColor       = ROOT.kRed-7)


# add JT categories
#plotter.addCategory("4j_ge3t")
#plotter.addCategory("5j_ge3t")
#plotter.addCategory("ge6j_ge3t")
plotter.addCategory("ge4j_ge3t")


# perform plotting routine
plotter.plot()
