import pandas
import socket
import glob
import os
import numpy as np
import sys

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)
import utils.generateJTcut as JTcut
import plot_configs.variable_binning as binning
import plot_configs.setupPlots as setup

# import a variable set
import variable_sets.top_10_variables as variable_set

# current integrated lumi
lumi = 41.5

def load_dataframes(backgrounds, signals):
    # load dataframes and add "weight" variable to dataframe which consists of XS*GN_nom*Lumi
    bkg_dfs = {}
    sig_dfs = {}
    for key in signals:
        print("loading signal "+str(signals[key]))
        with pandas.HDFStore( signals[key], mode = "r" ) as store:
            df = store.select("data")
            sig_dfs[key] = df.assign(weight = lambda x: x.Weight_XS*x.Weight_GEN_nom*lumi)
            
    for key in backgrounds:
        print("loading backgroud "+str(backgrounds[key]))
        with pandas.HDFStore( backgrounds[key], mode = "r" ) as store:
            df = store.select("data")
            bkg_dfs[key] = df.assign(weight = lambda x: x.Weight_XS*x.Weight_GEN_nom*lumi)

    return bkg_dfs, sig_dfs

def cut_dataframes(bkg_dfs, sig_dfs, cat, category_vars):
    # generate cut query
    category_cut = JTcut.getJTstring(cat)
    
    cut_sig_dfs = {}
    cut_bkg_dfs = {}

    # cut bkg dataframes to JTcategory selection and used variables
    for key in bkg_dfs:
        cut_bkg_dfs[key] = bkg_dfs[key].query(category_cut)[category_vars+["weight"]]

    # cut sig dataframes to JTcategory selection and used variables
    for key in sig_dfs:
        cut_sig_dfs[key] = sig_dfs[key].query(category_cut)[category_vars+["weight"]]

    return cut_bkg_dfs, cut_sig_dfs

def hist_variable(variable, plot_name, bkgs, sigs, ordered_backgrounds, category, plotOptions = {}):
    # set plotting options
    defaultOptions = {
        "ratio":        False,
        "ratioTitle":   None,
        "logscale":     False,
        "scaleSignal":  -1}
    for key in plotOptions:
        defaultOptions[key] = plotOptions[key]
    plotOptions = defaultOptions

    # get number of bins and bin range from config file
    bins = binning.binning[variable]["nbins"]
    bin_range = binning.binning[variable]["bin_range"]

    bkgHists = []
    bkgLabels = []
    weightIntegral = 0

    # loop over backgrounds and fill hists
    for key in ordered_backgrounds:
        # get weights
        weights = bkgs[key]["weight"].values
        weightIntegral += sum(weights)

        # setup histogram 
        hist = setup.setupHistogram(
            values      = bkgs[key][variable].values,
            weights     = weights,
            nbins       = bins,
            bin_range   = bin_range,
            color       = setup.GetPlotColor(key),
            xtitle      = category+"_"+key+"_"+variable,
            ytitle      = setup.GetyTitle(),
            filled      = True)

        bkgHists.append(hist)
        bkgLabels.append(key)   

    sigHists = []
    sigLabels = []
    sigScales = []
    # loop over signals and fill hists
    for key in sigs:
        # get weights
        weights = sigs[key]["weight"].values

        # determine scale factor
        if plotOptions["scaleSignal"] == -1:
            scaleFactor = weightIntegral/(sum(weights)+1e-9)
        else:
            scaleFactor = float(plotOptions["scaleFactor"])
    
        # setup histogram
        hist = setup.setupHistogram(
            values      = sigs[key][variable].values,
            weights     = weights,
            nbins       = bins,
            bin_range   = bin_range,
            color       = setup.GetPlotColor(key),
            xtitle      = category+"_"+key+"_"+variable,
            ytitle      = setup.GetyTitle(),
            filled      = False)

        hist.Scale(scaleFactor)
        
        sigHists.append(hist)
        sigLabels.append(key)
        sigScales.append(scaleFactor)

    canvas = setup.drawHistsOnCanvas(
        sigHists, bkgHists, plotOptions,
        canvasName = category+"_"+variable)

    # setup legend
    legend = setup.getLegend()
    
    # add signal entries
    for iSig in range(len(sigHists)):
        legend.AddEntry(sigHists[iSig], sigLabels[iSig]+" x {:4.0f}".format(sigScales[iSig]), "L")

    # add background entries
    for iBkg in range(len(bkgHists)):
        legend.AddEntry(bkgHists[iBkg], bkgLabels[iBkg], "F")

    # draw legend
    legend.Draw("same")

    # add lumi and category to plot
    setup.printLumi(canvas, ratio = plotOptions["ratio"])
    setup.printCategoryLabel(canvas, JTcut.getJTlabel(cat), ratio = plotOptions["ratio"])

    # save canvas
    setup.saveCanvas(canvas, plot_name)

# ====================================================
# configs
# ====================================================


# directory of input hdf5 files
data_dir = "/ceph/vanderlinden/MLFoyTrainData/DNN/"
# output directory for plots
plot_dir = data_dir + "/plots/"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# JT categories
categories = ["4j_ge3t", "5j_ge3t", "ge6j_ge3t"]


# location of background samples
backgrounds = { "ttbb": data_dir + "/ttbb_dnn.h5",
                "tt2b": data_dir + "/tt2b_dnn.h5",
                "ttb":  data_dir + "/ttb_dnn.h5",
                "ttcc": data_dir + "/ttcc_dnn.h5",
                "ttlf": data_dir + "/ttlf_dnn.h5"}
# order in which to stack the backgrounds (lowest first)
ordered_backgrounds = ["ttbb", "tt2b", "ttb", "ttcc", "ttlf"]

#  location of signal samples
signals = { "ttH": data_dir + "/ttHbb_dnn.h5" }

# additional variables to plot
add_vars = [
    ]

# options for plotting
plotOptions = {
        "ratio":        False,
        "ratioTitle":   "#frac{scaled Signal}{Background}",
        "logscale":     False,
        "scaleSignal":  -1}
# scaleSignal: 
#       -1: scale to background integral
#       float: scale x float


# load the dataframes
bkg_dfs, sig_dfs = load_dataframes(backgrounds, signals)


# loop over categories and get list of variables
for cat in categories:
    print("starting with category "+str(cat))
    # generate directory for plots
    cat_dir = plot_dir + "/shapes_"+cat+"/"
    if not os.path.exists(cat_dir):
        os.makedirs(cat_dir)

    # load variables from variable set
    category_vars = variable_set.variables[cat]+add_vars

    # cut dataframes to JTcategory and variables to be plotted
    cut_bkg_dfs, cut_sig_dfs = cut_dataframes(bkg_dfs, sig_dfs, cat, category_vars)

    # loop over variables
    for variable in category_vars:
        print(variable)
        # generate plot output name
        plot_name = cat_dir + "/{}.pdf".format(variable)
        plot_name = plot_name.replace("[","_").replace("]","")
        
        # generate plot for this variable
        hist_variable(
            variable            = variable, 
            plot_name           = plot_name, 
            bkgs                = cut_bkg_dfs, 
            sigs                = cut_sig_dfs, 
            ordered_backgrounds = ordered_backgrounds, 
            category            = cat, 
            plotOptions         = plotOptions)

    # unite plots into one pdf
    cmd = "pdfunite "+cat_dir+"/*.pdf "+plot_dir+"/plots_"+cat+".pdf"
    print(cmd)
    os.system(cmd)
    



    
    
