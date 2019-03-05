# global imports
import os
import sys
import pandas as pd
import glob
from collections import Counter
import operator
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)


input_dir = sys.argv[1]
n_variables = int(sys.argv[2])

jtcategories = ["4j_ge3t", "5j_ge3t", "ge6j_ge3t"]

print("variables = {}")

for cat in jtcategories:
    file_dir = basedir+"/workdir/"+input_dir+"_"+cat+"/run*/absolute_weight_sum.csv"
    rankings = glob.glob(file_dir)
    top_vars = []
    for ranking in rankings:
        csv = pd.read_csv(ranking, header = 0, sep = ",").tail(int(n_variables))
        variables = list(csv["variable"].values)
        top_vars += variables
    ranked = dict(Counter(top_vars))
    ranked = sorted(ranked.items(), key = operator.itemgetter(1))
    # print lists for mattermost
    print("\n## top {} variables {}\n".format(n_variables, cat))
    for r in ranked[-n_variables:]: print(r[0])
    print("\n"+"-"*40+"\n")

    # print variable set
    #print("variables[\""+cat+"\"] = [")
    #for v in variables: print("\t\""+v+"\",")
    #print("\t]\n\n\n")

print("all_variables = set( [v for key in variables for v in variables[key] ] )")
