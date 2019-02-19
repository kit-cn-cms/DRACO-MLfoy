import ROOT
import os
import sys
import ROOT
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
studydir = os.path.dirname(filedir)
basedir = os.path.dirname(studydir)
sys.path.append(basedir)

import variable_sets.ntuplesVariablesWithIndex as variable_set
import analyzerBoie

# location of input dataframes
data_dir = "/ceph/vanderlinden/MLFoyTrainData/DNN_ttZ_v2/"

# output location of plots
plot_dir = "/ceph/vanderlinden/ttZ_2019/inputFeatures_ttZ/"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# additional variables to plot
additional_variables = [
    ]


# initialize analyzer
analyzer = analyzerBoie.variableAnalyzer(
    output_dir     = plot_dir,
    variable_set   = variable_set,
    add_vars       = additional_variables)

# add samples
analyzer.addSample(
    sampleName      = "ttH",
    sampleFile      = data_dir+"/ttHbb_dnn.h5")


analyzer.addSample(
    sampleName      = "ttZ",
    sampleFile      = data_dir+"/ttZbb_dnn.h5")

analyzer.addSample(
    sampleName      = "ttbb",
    sampleFile      = data_dir+"/ttbb_dnn.h5")

analyzer.addSample(
    sampleName      = "tt2b",
    sampleFile      = data_dir+"/tt2b_dnn.h5")

analyzer.addSample(
    sampleName      = "ttb",
    sampleFile      = data_dir+"/ttb_dnn.h5")

analyzer.addSample(
    sampleName      = "ttcc",
    sampleFile      = data_dir+"/ttcc_dnn.h5")

analyzer.addSample(
    sampleName      = "ttlf",
    sampleFile      = data_dir+"/ttlf_dnn.h5")


# add JT categories
analyzer.addCategory("4j_ge3t")
analyzer.addCategory("5j_ge3t")
analyzer.addCategory("ge6j_ge3t")
#analyzer.addCategory("ge4j_ge3t")


# perform plotting routine
analyzer.perform1Danalysis(metric = "KS")
#analyzer.perform1Danalysis(metric = "Chi2")
