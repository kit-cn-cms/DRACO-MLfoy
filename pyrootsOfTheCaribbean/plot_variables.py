import rootpy
import pandas
import socket
import variable_info
import glob
import os
import numpy as np
import rootpy.plotting as rp

# local imports
import plot_configs.variable_binning as binning
import plot_configs.plotting_styles as ps
ps.init_plot_style()

# current integrated lumi
lumi = 41.5

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

if "naf" in socket.gethostname():
    workpath = "/nfs/dust/cms/user/vdlinden/DRACO-MLfoy/workdir/"
else:
    workpath = "/ceph/vanderlinden/DRACO-MLfoy/workdir/"
data_dir = workpath+"/AachenDNN_files/"
plot_dir = workpath+"/AachenDNN_files/plots/"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)


backgrounds = { "ttbb": data_dir + "/ttbb.h5",
                "tt2b": data_dir + "/tt2b.h5",
                "ttb":  data_dir + "/ttb.h5",
                "ttcc": data_dir + "/ttcc.h5",
                "ttlf": data_dir + "/ttlf.h5"}
signals = { "ttH": data_dir + "/ttHbb.h5" }



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

add_vars = [
    "GenAdd_BB_inacceptance_jet",
    "GenAdd_B_inacceptance_jet",
    "GenHiggs_BB_inacceptance_jet",
    "GenHiggs_B_inacceptance_jet",
    "GenTopHad_B_inacceptance_jet",
    "GenTopHad_QQ_inacceptance_jet",
    "GenTopHad_Q_inacceptance_jet",
    "GenTopLep_B_inacceptance_jet",

    "GenAdd_BB_inacceptance_part",
    "GenAdd_B_inacceptance_part",
    "GenHiggs_BB_inacceptance_part",
    "GenHiggs_B_inacceptance_part",
    "GenTopHad_B_inacceptance_part",
    "GenTopHad_QQ_inacceptance_part",
    "GenTopHad_Q_inacceptance_part",
    "GenTopLep_B_inacceptance_part",
    #"Weight_XS",
    #"Weight_CSV",
    #"Weight_GEN_nom"
    ]
# loop over categories and get list of variables
for cat in categories:
    print("starting with category "+str(cat))
    category_cut = cat
    category_vars = categories[cat]+add_vars

    cut_sig_dfs = {}
    cut_bkg_dfs = {}    

    cat_dir = plot_dir + "/shapes_"+str(category_names[cat])+"/"
    if not os.path.exists(cat_dir):
        os.makedirs(cat_dir)

    for key in bkg_dfs:
        cut_bkg_dfs[key] = bkg_dfs[key].query(category_cut)[category_vars+["weight"]]

    for key in sig_dfs:
        cut_sig_dfs[key] = sig_dfs[key].query(category_cut)[category_vars+["weight"]]

    for variable in category_vars:
        print(variable)
        plot_name = cat_dir + "/{}.pdf".format(variable)
        plot_name = plot_name.replace("[","_").replace("]","")

        
        hist_variable(variable, plot_name, cut_bkg_dfs, cut_sig_dfs, plt_title = category_names[cat], log = False)
        hist_variable(variable, plot_name, cut_bkg_dfs, cut_sig_dfs, plt_title = category_names[cat], log = True)
    



    
    
