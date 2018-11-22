import pandas
import socket
import variable_info
import glob
import os
import numpy as np

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
    workpath = "/storage/c/vanderlinden/DRACO-MLfoy/workdir/"
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

def get_true_frac(df, var, weighted = True):
    if not weighted:
        df["weight"] = pandas.Series([1.]*df.shape[0], index = df.index)

    true_df = df.query(str(var)+" == 1")
    false_df = df.query(str(var)+" == 0")

    true_sum = true_df["weight"].sum()
    false_sum = false_df["weight"].sum()
    
    true_rate = true_sum/(true_sum+false_sum)
    return true_rate

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
    #"GenAdd_BB_inacceptance_jet",
    #"GenAdd_B_inacceptance_jet",
    #"GenHiggs_BB_inacceptance_jet",
    #"GenHiggs_B_inacceptance_jet",
    #"GenTopHad_B_inacceptance_jet",
    #"GenTopHad_QQ_inacceptance_jet",
    #"GenTopHad_Q_inacceptance_jet",
    #"GenTopLep_B_inacceptance_jet",
    "GenAdd_BB_inacceptance_part",
    "GenAdd_B_inacceptance_part",
    "GenHiggs_BB_inacceptance_part",
    "GenHiggs_B_inacceptance_part",
    "GenTopHad_B_inacceptance_part",
    "GenTopHad_QQ_inacceptance_part",
    "GenTopHad_Q_inacceptance_part",
    "GenTopLep_B_inacceptance_part",
    ]

classes = ["ttH", "ttbb", "tt2b", "ttb", "ttcc", "ttlf"]

for cat in categories:
    print("\n\n\n{:32s}".format("")+str(cat))
    print("{:30s} | ttH    | ttbb   | tt2b   | ttb    | ttcc   | ttlf   |".format(""))
    category_cut = cat

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

    
