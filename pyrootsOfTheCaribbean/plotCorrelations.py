import pandas
import glob
import os
import sys
import socket
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# local imports
import variable_info
import plot_configs.variable_binning as binning
import plot_configs.plotting_styles as ps
ps.init_plot_style()

# give 1 as first argument if all correlations should be plotted as scatter plots
if len(sys.argv) > 1:   plot_correlations = sys.argv[1]
else:                   plot_correlations = 0

# lumi
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

data_dir = workpath+"/AachenDNN_files"
plot_dir = workpath+"/AachenDNN_files/plots/"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

backgrounds = { "ttbb": data_dir + "/ttbb.h5",
                "tt2b": data_dir + "/tt2b.h5",
                "ttb":  data_dir + "/ttb.h5",
                "ttcc": data_dir + "/ttcc.h5",
                "ttlf": data_dir + "/ttlf.h5"}
signals = { "ttH": data_dir + "/ttHbb.h5" }


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

def plot_correlation(var1, var1_name, var2, var2_name, plot_dir, cat):
    plt.hist2d(var1, var2,
        bins = [min(binning.binning[var1_name]["nbins"],20),min(binning.binning[var2_name]["nbins"],20)])
    plt.colorbar()

    plt.title(cat, loc = "left")
    plt.xlabel(var1_name)
    plt.ylabel(var2_name)

    cat_dir = plot_dir + "/" + cat
    if not os.path.exists(cat_dir):
        os.makedirs(cat_dir)
    out_path = cat_dir + "/scatter_"+ \
        str(var1_name.replace("[","_").replace("]",""))+"_vs_"+ \
        str(var2_name.replace("[","_").replace("]",""))+".pdf"
    plt.savefig(out_path)
    print("saved plot at "+out_path)
    plt.clf()


# loop over categories and get list of variables
for cat in categories:
    print("starting with category "+str(cat))
    category_cut = cat
    category_vars = sorted(categories[cat])


    matrix = []
    cut_df = grand_df.query(category_cut)[category_vars]
    for v1 in category_vars:
        line = []
        for v2 in category_vars:
            correlation = np.corrcoef( cut_df[v1].values, cut_df[v2].values )[0][1]
            if plot_correlations:
                plot_correlation(cut_df[v1].values, v1, cut_df[v2].values, v2, plot_dir, category_names[cat])
            line.append(correlation)
            #print("correlation of {} and {}: {}".format(
            #    v1, v2, correlation))       

        matrix.append(line)

    n_variables = len(category_vars)
    plt.figure(figsize = [10,10])
    
    y = np.arange(0., n_variables+1, 1)
    x = np.arange(0., n_variables+1, 1)
    
    xn, yn = np.meshgrid(x,y)

    plt.pcolormesh(xn, yn, matrix, cmap = "RdBu", vmin = -1, vmax = 1)

    cb = plt.colorbar()
    cb.set_label("correlation")
   
    plt.xlim(0, n_variables)
    plt.ylim(0, n_variables)
    plt.title(cat, loc = "left")

    plt_axis = plt.gca()
    plt_axis.set_xticks(np.arange( (x.shape[0] -1)) + 0.5, minor = False )
    plt_axis.set_yticks(np.arange( (y.shape[0] -1)) + 0.5, minor = False )    

    plt_axis.set_xticklabels(category_vars, rotation = 90)
    plt_axis.set_yticklabels(category_vars)

    plt.tick_params(axis = "both", which = "major", labelsize = 7)

    plt_axis.set_aspect("equal")
    plt.tight_layout()


    save_path = plot_dir + "/correlation_matrix_"+str(category_names[cat])+".pdf"
    plt.savefig(save_path)
    print("saved correlation matrix at "+str(save_path))
    plt.clf()
    
    
