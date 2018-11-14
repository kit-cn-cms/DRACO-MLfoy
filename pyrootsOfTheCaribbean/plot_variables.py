import rootpy
import pandas
import matplotlib.pyplot as plt
import variable_info
import variable_binning
import glob
import os
#import ROOT
import numpy as np
import rootpy.plotting as rp
import rootpy.plotting.root2matplotlib as rplt

style = rp.style.get_style("ATLAS")
style.SetEndErrorSize(3)
rp.style.set_style(style)


def get_colors():
    return {
        "ttH": "royalblue",
        "ttlf":  "salmon",
        "ttcc":  "tomato",
        "ttbb":  "brown",
        "tt2b":  "darkred",
        "ttb":   "red"
        }


categories = {
    "(N_Jets == 6 and N_BTagsM >= 3)": variable_info.variables_4j_3b,
    "(N_Jets == 5 and N_BTagsM >= 3)": variable_info.variables_5j_3b,
    "(N_Jets == 4 and N_BTagsM >= 3)": variable_info.variables_6j_3b,
    }

category_names = {
    "(N_Jets == 6 and N_BTagsM >= 3)": "6j_ge3t",
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

    color_dict = get_colors()
    ordered_bkgs = ["ttbb", "tt2b", "ttb", "ttcc", "ttlf"]

    bins = variable_binning.binning[variable]["nbins"]
    bin_range = variable_binning.binning[variable]["bin_range"]

    #r = bin_range[1]-bin_range[0]
    #bin_range[0]-= 0.5*r
    #bin_range[1]+= 0.5*r

    bkg_hists = []
    weight_integral = 0
    for key in ordered_bkgs:
        weight_integral += sum(bkgs[key]["weight"].values)
        hist = rp.Hist( bins, *bin_range, title = key, markersize = 0, legendstyle = "F" )
        hist.Sumw2()
        hist.fill_array(
            bkgs[key][variable].values, 
            bkgs[key]["weight"].values )
        hist.fillstyle = "solid"
        hist.fillcolor = color_dict[key]
        hist.linecolor = "black"
        hist.linewidth = 1
        bkg_hists.append( hist )

    bkg_stack = rp.HistStack( bkg_hists, stacked = True , drawstyle ="HIST E1 X0")
    bkg_stack.SetMinimum(1e-4)
    # plot ttH    
    sig_key = "ttH"

    values = sigs[sig_key][variable].values
    weights = sigs[sig_key]["weight"].values
    weight_sum = sum(weights)
    scale_factor = 1.*weight_integral/weight_sum
    weights = [w*scale_factor for w in weights]

    sig_title = sig_key + "*{:.3f}".format(scale_factor)
    sig_hist = rp.Hist( bins, *bin_range, title = sig_title, markersize = 0, drawstyle = "shape", legendstyle = "L" )
    sig_hist.Sumw2()
    #sig_hist.markersize = 1
    sig_hist.fillstyle = "hollow"
    sig_hist.linestyle = "solid"
    sig_hist.linecolor = color_dict[sig_key]
    sig_hist.linewidth = 2
    sig_hist.fill_array(values, weights)

    # create canvas
    canvas = rp.Canvas(width = 1024, height = 768)
    canvas.SetTopMargin(0.07)
    canvas.SetBottomMargin(0.15)
    canvas.SetRightMargin(0.05)
    canvas.SetLeftMargin(0.15)
    canvas.SetTicks(1,1)
    #rp.utils.draw([bkg_stack, sig_stack], xtitle = variable, ytitle = "Events", pad = canvas)
    rp.utils.draw([bkg_stack, sig_hist], xtitle = variable, ytitle = "Events", pad = canvas)
    
    if log: canvas.cd().SetLogy()

    legend = rp.Legend(bkg_hists+[sig_hist], entryheight = 0.03)#, rightmargin = 0.05, margin = 0.3)
    legend.SetX1NDC(0.90)
    legend.SetX2NDC(0.99)
    legend.SetY1NDC(0.90)
    legend.SetY2NDC(0.99)
    legend.SetBorderSize(0)
    legend.SetLineStyle(0)
    legend.SetTextSize(0.04)
    legend.SetTextFont(42)
    legend.SetFillStyle(0)
    legend.Draw()
    
    canvas.Modified()
    canvas.Update()

    if log: plot_name = plot_name.replace(".pdf","_log.pdf")
    canvas.SaveAs(plot_name)

    canvas.Clear()



# load dataframes
bkg_dfs = {}
sig_dfs = {}
for key in signals:
    print("loading signal "+str(signals[key]))
    with pandas.HDFStore( signals[key], mode = "r" ) as store:
        df = store.select("data")
        sig_dfs[key] = df.assign(weight = lambda x: x.Weight_XS*x.Weight_CSV*x.Weight_GEN_nom)
        
for key in backgrounds:
    print("loading backgroud "+str(backgrounds[key]))
    with pandas.HDFStore( backgrounds[key], mode = "r" ) as store:
        df = store.select("data")
        bkg_dfs[key] = df.assign(weight = lambda x: x.Weight_XS*x.Weight_CSV*x.Weight_GEN_nom)

add_vars = [
    "GenAdd_BB_inacceptance",
    "GenAdd_B_inacceptance",
    "GenHiggs_BB_inacceptance",
    "GenHiggs_B_inacceptance",
    "GenTopHad_B_inacceptance",
    "GenTopHad_QQ_inacceptance",
    "GenTopHad_Q_inacceptance",
    "GenTopLep_B_inacceptance",
    "Weight_XS",
    "Weight_CSV",
    "Weight_GEN_nom"]

# loop over categories and get list of variables
for cat in categories:
    print("starting with category "+str(cat))
    category_cut = cat
    category_vars = categories[cat]+add_vars

    cut_sig_dfs = {}
    cut_bkg_dfs = {}    

    for key in bkg_dfs:
        cut_bkg_dfs[key] = bkg_dfs[key].query(category_cut)[category_vars+["weight"]]

    for key in sig_dfs:
        cut_sig_dfs[key] = sig_dfs[key].query(category_cut)[category_vars+["weight"]]

    for variable in category_vars:
        print(variable)
        plot_name = plot_dir + "/{}_{}.pdf".format(category_names[cat], variable)
        plot_name = plot_name.replace("[","_").replace("]","")

        
        hist_variable(variable, plot_name, cut_bkg_dfs, cut_sig_dfs, plt_title = category_names[cat], log = False)
        hist_variable(variable, plot_name, cut_bkg_dfs, cut_sig_dfs, plt_title = category_names[cat], log = True)
    



    
    
