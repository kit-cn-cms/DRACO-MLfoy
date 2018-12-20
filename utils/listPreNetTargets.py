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
import generateJTcut as JTcut
import variable_sets.top_10_variables as variable_set

# current integrated lumi
lumi = 41.5

data_dir = "/ceph/vanderlinden/MLFoyTrainData/DNN/"
plot_dir = data_dir + "/plots/"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

categories = ["4j_ge3t", "5j_ge3t", "ge6j_ge3t"]
classes = ["ttH", "ttbb", "tt2b", "ttb", "ttcc", "ttlf"]

backgrounds = { "ttbb": data_dir + "/ttbb_dnn.h5",
                "tt2b": data_dir + "/tt2b_dnn.h5",
                "ttb":  data_dir + "/ttb_dnn.h5",
                "ttcc": data_dir + "/ttcc_dnn.h5",
                "ttlf": data_dir + "/ttlf_dnn.h5"}
signals = { "ttH": data_dir + "/ttHbb_dnn.h5" }




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

prenet_targets = [
    "GenAdd_BB_inacceptance",
    "GenAdd_B_inacceptance",
    "GenHiggs_BB_inacceptance",
    "GenHiggs_B_inacceptance",
    "GenTopHad_B_inacceptance",
    "GenTopHad_QQ_inacceptance",
    "GenTopHad_Q_inacceptance",
    "GenTopLep_B_inacceptance",
    ]

prenet_targets = [t+"_part" for t in prenet_targets]

def get_true_frac(df, var, weighted = True):
    if not weighted:
        df["weight"] = pandas.Series([1.]*df.shape[0], index = df.index)

    true_df = df.query(str(var)+" == 1")
    false_df = df.query(str(var)+" == 0")

    true_sum = true_df["weight"].sum()
    false_sum = false_df["weight"].sum()
    
    true_rate = true_sum/(true_sum+false_sum)
    return true_rate


for cat in categories:
    print("\n\n\n{:32s}".format("")+str(cat))
    print("{:30s} | ttH    | ttbb   | tt2b   | ttb    | ttcc   | ttlf   |".format(""))
    category_cut = JTcut.getJTstring(cat)

    cut_sig_dfs = {}
    cut_bkg_dfs = {}    

    for key in bkg_dfs:
        cut_bkg_dfs[key] = bkg_dfs[key].query(category_cut)[prenet_targets+["weight"]]

    for key in sig_dfs:
        cut_sig_dfs[key] = sig_dfs[key].query(category_cut)[prenet_targets+["weight"]]

    for variable in prenet_targets:

        true_fracs = {}
        for key in cut_sig_dfs:
            true_frac = get_true_frac( cut_sig_dfs[key], variable)           
            true_fracs[key] = true_frac

        for key in cut_bkg_dfs:
            true_frac = get_true_frac( cut_bkg_dfs[key], variable)
            true_fracs[key] = true_frac

        print_str = "{:30s} | ".format(variable)
        for c in classes:
            print_str+= "{:05.2f}% | ".format(true_fracs[c]*100)
        print(print_str)

    
