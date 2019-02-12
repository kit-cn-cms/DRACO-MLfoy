# global imports
import os
import sys
import pandas as pd

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)


input_dir = sys.argv[1]
n_variables = int(sys.argv[2])

jtcategories = ["4j_ge3t", "5j_ge3t", "ge6j_ge3t"]

print("variables = {}")

for cat in jtcategories:
    file_dir = basedir+"/workdir/"+input_dir+"_"+cat+"/variable_ranking.csv"

    csv = pd.read_csv(file_dir, header = 0, sep = ",").tail(n_variables)
    variables = csv["variable"].values

    # print lists for mattermost
    #print("\n## top {} variables {}\n".format(n_variables, cat))
    #for v in variables: print(v)
    #print("\n"+"-"*40+"\n")

    # print variable set
    print("variables[\""+cat+"\"] = [")
    for v in variables: print("\t\""+v+"\",")
    print("\t]\n\n\n")

print("all_variables = set( [v for key in variables for v in variables[key] ] )")
