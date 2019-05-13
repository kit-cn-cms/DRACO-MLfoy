import pandas
import glob
import os
import sys
import socket
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import optparse

# local imports
import variable_info

# lumi
lumi = 41.3


filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

"""
USE: python evaluation.py -i
"""
usage="usage=%prog [options] \n"
usage+="USE: python train_template.py -i "

parser = optparse.OptionParser(usage=usage)

parser.add_option("-i", "--inputdirectory", dest="inputDir",default="cate3",
        help="DIR of data", metavar="inputDir")

(options, args) = parser.parse_args()


print("basedir")
print(basedir)

if not os.path.isabs(options.inputDir):
    inPath = basedir+"/Inputs/"+options.inputDir
elif os.path.exists(options.inputDir):
    inPath=options.inputDir
else:
    sys.exit("ERROR: Input Directory does not exist!")
print(inPath)

input_categories = {
    "cate8" : "(N_Jets >= 4 and N_BTagsM >= 4)",
    "cate9" : "(N_Jets >= 4 and N_BTagsM >= 3)",
    "cate7" : "(N_Jets >= 4 and N_BTagsM = 3)",
    "cate6" : "(N_Jets >= 4 and N_BTagsM = 2)",
    "cate4" : "(N_Jets = 3 and N_BTagsM = 3)",
    "cate3" : "(N_Jets = 3 and N_BTagsM = 2)",
    }


categories = {
    "(N_Jets >= 4 and N_BTagsM >= 4)": variable_info.variables_ge4j_ge4t,
    "(N_Jets >= 4 and N_BTagsM >= 3)": variable_info.variables_ge4j_ge3t,
    "(N_Jets >= 4 and N_BTagsM = 3)": variable_info.variables_ge4j_3t,
    "(N_Jets >= 4 and N_BTagsM = 2)": variable_info.variables_ge4j_2t,
    "(N_Jets = 3 and N_BTagsM = 2)": variable_info.variables_3j_2t,
    "(N_Jets = 3 and N_BTagsM = 3)": variable_info.variables_3j_3t,
    }

category_names = {
    "(N_Jets >= 4 and N_BTagsM >= 4)": "ge4j_ge4t",
    "(N_Jets >= 4 and N_BTagsM >= 3)": "ge4j_ge3t",
    "(N_Jets >= 4 and N_BTagsM = 3)": "ge4j_3t",
    "(N_Jets >= 4 and N_BTagsM = 2)": "ge4j_2t",
    "(N_Jets = 3 and N_BTagsM = 2)": "3j_2t",
    "(N_Jets = 3 and N_BTagsM = 3)": "3j_3t",
    }

if "naf" in socket.gethostname():
    workpath = "/nfs/dust/cms/user/vdlinden/DRACO-MLfoy/workdir/"
else:
    workpath = "/storage/c/vanderlinden/DRACO-MLfoy/workdir/"

plot_dir = "/nfs/dust/cms/user/angirald/workspace/DRACO-MLfoy/correlationplots/"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

backgrounds = { "ttbb": inPath + "/ttbb_dnn.h5",
                "tt2b": inPath + "/tt2b_dnn.h5",
                "ttb":  inPath + "/ttb_dnn.h5",
                "ttcc": inPath + "/ttcc_dnn.h5",
                "ttlf": inPath + "/ttlf_dnn.h5"}
signals = { "ttH": inPath + "/ttHbb_dnn.h5" }


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
cat = input_categories[options.inputDir]
print("starting with category "+str(cat))
category_cut = cat
category_vars = sorted(categories[cat])
print(category_vars)

matrix = []
cut_df = grand_df[category_vars]
for v1 in category_vars:
    line = []
    for v2 in category_vars:
        correlation = np.corrcoef( cut_df[v1].values, cut_df[v2].values )[0][1]
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


save_path = plot_dir + "/correlation_matrix_"+str(category_names[cat])+".png"
plt.savefig(save_path)
print("saved correlation matrix at "+str(save_path))
plt.clf()
