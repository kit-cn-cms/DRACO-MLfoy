import rootpy
import pandas
import variable_info
import plot_configs.variable_binning as binning
import plot_configs.plotting_styles as ps
ps.init_plot_style()

import glob
import os
#import ROOT
import numpy as np
import rootpy.plotting as rp
import matplotlib.pyplot as plt

lumi = 41.3

categories = {
    "(N_Jets >= 6 and N_BTagsM >= 3)": variable_info.variables_4j_3b,
    "(N_Jets == 5 and N_BTagsM >= 3)": variable_info.variables_5j_3b,
    "(N_Jets == 4 and N_BTagsM >= 3)": variable_info.variables_6j_3b,
    }

category_names = {
    "(N_Jets >= 6 and N_BTagsM >= 3)": "ge6j_ge3t",
    "(N_Jets == 5 and N_BTagsM >= 3)": "5j_ge3t",
    "(N_Jets == 4 and N_BTagsM >= 3)": "4j_ge3t",
    }

data_dir = "/nfs/dust/cms/user/vdlinden/DRACO-MLfoy/workdir/AachenDNN_files"
backgrounds = { "ttbb": data_dir + "/ttbb.h5",
                "tt2b": data_dir + "/tt2b.h5",
                "ttb":  data_dir + "/ttb.h5",
                "ttcc": data_dir + "/ttcc.h5",
                "ttlf": data_dir + "/ttlf.h5"}
signals = { "ttH": data_dir + "/ttHbb.h5" }

plot_dir = "/nfs/dust/cms/user/vdlinden/DRACO-MLfoy/workdir/AachenDNN_files/plots/"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)



def hist_variable(variable, plot_name, bkgs, sigs, plt_title, log = False):

    # backgrounds ordered for nicer plotting
    ordered_bkgs = ["ttbb", "tt2b", "ttb", "ttcc", "ttlf"]

    bins = binning.binning[variable]["nbins"]
    bin_range = binning.binning[variable]["bin_range"]

    bkg_hists = []
    weight_integral = 0
    # loop over backgrounds and fill hists
    for key in ordered_bkgs:
        weights = bkgs[key]["weight"].values
        weight_integral += sum(weights)
        hist = rp.Hist( bins, *bin_range, title = key)
        ps.set_bkg_hist_style(hist, key)
        hist.fill_array(
            bkgs[key][variable].values, 
            weights )
    
        bkg_hists.append( hist )

    bkg_stack = rp.HistStack( bkg_hists, stacked = True , drawstyle ="HIST E1 X0")
    max_val = bkg_stack.GetMaximum()*1.3
    bkg_stack.SetMaximum(max_val)
    bkg_stack.SetMinimum(1e-4)


    # get signal values
    sig_key = "ttH"
    values = sigs[sig_key][variable].values

    # adjust weights to bkg integral
    weights = sigs[sig_key]["weight"].values
    weight_sum = sum(weights)
    scale_factor = 1.*weight_integral/weight_sum
    weights = [w*scale_factor for w in weights]

    # hist signal
    sig_title = sig_key + "*{:.3f}".format(scale_factor)
    sig_hist = rp.Hist( bins, *bin_range, title = sig_title)
    ps.set_sig_hist_style(sig_hist,sig_key)
    sig_hist.fill_array(values, weights)

    # create canvas
    canvas = ps.init_canvas()

    # draw histograms
    rp.utils.draw([bkg_stack, sig_hist], xtitle = variable, ytitle = "Events", pad = canvas)
    if log: canvas.cd().SetLogy()

    # add legend
    legend = ps.init_legend( bkg_hists+[sig_hist] )

    # add titles
    ps.add_lumi(canvas)
    ps.add_category_label(canvas, plt_title)

    # save plot
    if log: plot_name = plot_name.replace(".pdf","_log.pdf")
    ps.save_canvas(canvas, plot_name)

# load dataframes
grand_df = pandas.DataFrame()
for key in signals:
    print("loading signal "+str(signals[key]))
    with pandas.HDFStore( signals[key], mode = "r" ) as store:
        df = store.select("data")
        if grand_df.empty:
            grand_df = df
        else:
            grand_df.append( df )
        
for key in backgrounds:
    print("loading backgroud "+str(backgrounds[key]))
    with pandas.HDFStore( backgrounds[key], mode = "r" ) as store:
        df = store.select("data")
        grand_df.append( df )

#grand_df = grand_df.assign(weight = lambda x: x.Weight_XS*x.Weight_GEN_nom*lumi)

# loop over categories and get list of variables
for cat in categories:
    print("starting with category "+str(cat))
    category_cut = cat
    category_vars = categories[cat]


    matrix = []
    cut_df = grand_df.query(category_cut)[category_vars]
    for v1 in category_vars:
        line = []
        for v2 in category_vars:
            correlation = np.corrcoef( cut_df[v1].values, cut_df[v2].values )[0][1]
            line.append(correlation)
            #print("correlation of {} and {}: {}".format(
            #    v1, v2, correlation))       

        matrix.append(line)

    print(matrix)
    n_variables = len(category_vars)
    plt.figure(figsize = [15,12])
    
    y = np.arange(0., n_variables+1, 1)
    x = np.arange(0., n_variables+1, 1)
    
    xn, yn = np.meshgrid(x,y)

    plt.pcolormesh(xn, yn, matrix)
    plt.colorbar()
    
    plt.xlim(0, n_variables)
    plt.ylim(0, n_variables)

    plt.show()
    plt.clf()

    
    
