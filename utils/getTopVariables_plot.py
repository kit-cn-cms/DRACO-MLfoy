# global imports
import optparse
import os
import sys
import pandas as pd
import glob
from collections import Counter
import operator
import matplotlib.pyplot as plt
import numpy as np
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

# parse options
parser = optparse.OptionParser(usage="%prog [options] file")
parser.add_option("-d", "--directory", dest = "input_dir", metavar = "INPUTDIR",
    help = "name of input directory which contains the weight files at '...workdir/INPUTDIR_CAT/run*/absolute_weight_sum.csv'")
parser.add_option("-o", "--output", dest = "output_dir", metavar = "OUTDIR", default = filedir,
    help = "directory for output plots")
(opts, args) = parser.parse_args()

jtcategories = ["4j_ge3t", "5j_ge3t", "ge6j_ge3t"]
for cat in jtcategories:
    print("\n\n"+cat+"\n\n")
    file_dir = basedir+"/workdir/"+opts.input_dir+"_"+cat+"/run*/absolute_weight_sum.csv"
    rankings = glob.glob(file_dir)
    variables = {}
    for ranking in rankings:
        csv = pd.read_csv(ranking, header = 0, sep = ",")
        sum_of_weights = csv["weight_sum"].sum()
        for row in csv.iterrows():
            if not row[1][0] in variables: variables[row[1][0]] = []
            variables[row[1][0]].append(row[1][1]/sum_of_weights)


    mean_dict = {}
    for v in variables: mean_dict[v] = np.median(variables[v])
    var = []
    val = []
    mean = []
    std = []
    i = 0
    for v, m in sorted(mean_dict.iteritems(), key = lambda (k, vl): (vl, k)):
        i += 1
        val.append(i)
        var.append(v)
        mean.append(m)
        std.append( np.std(variables[v]) )
        print(v,m)

    plt.figure(figsize = [10,10])
    plt.errorbar(mean, val, xerr = std, fmt = "o")
    plt.xlim([0.,0.07])
    plt.grid()
    plt.yticks(val, var)
    plt.title(cat)
    plt.xlabel("mean of sum of input weights")
    plt.tight_layout()
    plot_dir = opts.output_dir+"/"+opts.input_dir+"_weight_sums_"+cat+".pdf"
    plt.savefig(plot_dir)
    print("saved plot at {}".format(plot_dir))
    plt.clf() 
