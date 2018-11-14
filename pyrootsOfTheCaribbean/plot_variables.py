import pandas
import matplotlib.pyplot as plt
import variable_info
import variable_binning
import glob
import os

def get_colors():
    return {
        "ttH": "blue",
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
    plt.figure(figsize = [15,10])

    color_dict = get_colors()
    ordered_bkgs = ["ttbb", "tt2b", "ttb", "ttcc", "ttlf"]

    bins = variable_binning.binning[variable]["nbins"]
    bin_range = variable_binning.binning[variable]["bin_range"]

    bkg_names = []
    bkg_lists = []
    bkg_weights = []
    bkg_colors = []
    for key in ordered_bkgs:
        bkg_names.append(key)
        bkg_lists.append( bkgs[key][variable].values )
        bkg_weights.append( bkgs[key]["weight"].values )
        bkg_colors.append( color_dict[key] )

    n, bins, _ = plt.hist( bkg_lists, bins = bins, range = bin_range, log = log, histtype = "stepfilled", stacked = True, weights = bkg_weights, label = bkg_names, color = bkg_colors )

    weight_integral = sum([sum(l) for l in bkg_weights])
    for key in sigs:
        values = sigs[key][variable].values
        weights = sigs[key]["weight"].values
        weight_sum = sum(weights)
        scale_factor = 1.*weight_integral/weight_sum
        weights = [w*scale_factor for w in weights]

        plt.hist( values, bins = bins, range = bin_range, log = log, histtype = "step", weights = weights, label = key, color = color_dict[key], lw = 2 )

    plt.xlabel( variable )
    plt.ylabel( "entries per bin" )
    plt.xlim(bin_range)
    plt.legend()
    plt.grid()
    plt.title(plt_title, loc = "left")    

    if log: plot_name = plot_name.replace(".pdf","_log.pdf")

    plt.savefig( plot_name )
    print("saved plot at "+str(plot_name))
    plt.clf()





# load dataframes
bkg_dfs = {}
sig_dfs = {}
for key in signals:
    print("loading signal "+str(signals[key]))
    with pandas.HDFStore( signals[key], mode = "r" ) as store:
        df = store.select("data")
        sig_dfs[key] = df.assign(weight = lambda x: x.Weight_XS)
        
for key in backgrounds:
    print("loading backgroud "+str(backgrounds[key]))
    with pandas.HDFStore( backgrounds[key], mode = "r" ) as store:
        df = store.select("data")
        bkg_dfs[key] = df.assign(weight = lambda x: x.Weight_XS)

add_vars = [
    "GenAdd_BB_inacceptance",
    "GenAdd_B_inacceptance",
    "GenHiggs_BB_inacceptance",
    "GenHiggs_B_inacceptance",
    "GenTopHad_B_inacceptance",
    "GenTopHad_QQ_inacceptance",
    "GenTopHad_Q_inacceptance",
    "GenTopLep_B_inacceptance"]

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
    



    
    
