# global imports
import os
import sys
import pandas as pd
import glob
from collections import Counter
import operator
from csv import DictWriter, DictReader #me

# so that matplotlib can be used over ssh
import matplotlib #me
matplotlib.use('Agg') #me
import matplotlib.pyplot as plt #me


import numpy as np
import optparse
import pylab
import csv

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

#cmd: python getTopVariables.py -w /home/ycung/Desktop/DRACO-MLfoy_thesis/workdir -i training_ANN_v2/ANN_training_*_JTSTRING -o /home/ycung/Desktop/DRACO-MLfoy_thesis/workdir/variable_ranking_ANN_v2 -p --nplot 25 --nvset 25 --std -v -t first_layer ge4j_ge3t
#     python getTopVariables.py -w /home/ycung/Desktop/DRACO-MLfoy_thesis/workdir -i 16-04-2020/BNN_training_JTSTRING -o /home/ycung/Desktop/DRACO-MLfoy_thesis/workdir/variable_ranking_04-05-2020 --std -p -t first_layer ge4j_ge3t

usage = "python getTopVariables.py [options] [jtCategories]"
parser = optparse.OptionParser(usage=usage)
parser.add_option("-w","--workdir",dest="workdir",default=basedir+"/workdir",
    help = "path to working directory where trained DNN files are placed")
parser.add_option("-i","--input",dest="inputdir",default="test_training",
    help = "path to DNN directories relative to WORKDIR. add JTSTRING as placeholder\
for jet-tag categories and escaped wildcards ('\*') for multiple DNN runs.")
parser.add_option("-o","--output",dest="outdir",default="./",
    help = "path to output directory e.g. for plots or variable sets")
parser.add_option("--filename",dest="filename",default="",
    help = "filename") #me
parser.add_option("--std",dest="no_std",default=True,action="store_false",
    help = "set for deactivating plot of std") #me
parser.add_option("-p","--plot",dest="plot",default=False,action="store_true",
    help = "generate plots of variable rankings")
parser.add_option("--nplot",dest="nplot",default=30,type=int,
    help = "number of variables to be plotted (-1 for all).")
parser.add_option("-v","--variableset",dest="generate_variableset",default=False,action="store_true",
    help = "generate new variable set from variable rankings")
parser.add_option("--nvset",dest="nvset",default=30,type=int,
    help = "number of variables written to variable set (-1 for all).")
parser.add_option("-t", "--type", dest = "weight_type",default="absolute",
    help = "type of variable ranking (i.e. name of weight file TYPE_weight_sum.csv), e.g. 'absolute', 'propagated'")
parser.add_option("--taylor", dest = "taylor_expansion",default=False,action="store_true",
    help = "evaluate taylor expansion 1D ranking")
parser.add_option("--nodes", dest = "nodes",default="ttH",
    help = "comma separated list of processes to be considered in taylor ranking")
parser.add_option("--count", dest = "count", default=False, action="store_true",
    help = "for only counting the frequency of the variables in top_*.csv")

(opts, args) = parser.parse_args()

make_nice_labels_dict = {}
with open("/home/ycung/Desktop/DRACO-MLfoy_thesis/utils/make_labels_nice.csv") as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        make_nice_labels_dict[row[0]] = row[1]

if not opts.count:
    inputdir = opts.workdir+"/"+opts.inputdir
    if not "JTSTRING" in inputdir:
        inputdir+="_JTSTRING/"
    print("using input directory {}".format(inputdir))

    if not os.path.exists(opts.outdir):
        os.makedirs(opts.outdir)


    sorted_variables = {}
    for jtcat in args:
        print("\n\nhandling category {}\n\n".format(jtcat))
        jtpath = inputdir.replace("JTSTRING",jtcat)
        
        if opts.taylor_expansion:
            jtpath+= "/checkpoints/keras_taylor_1D.csv"
        
        elif "BNN" in opts.inputdir:
            jtpath+= "/"+opts.weight_type+"_weights.csv"

        else:
            # collect weight sum files
            jtpath+= "/"+opts.weight_type+"_weight_sums.csv"

        rankings = glob.glob(jtpath)
        print("found {} variable ranking files".format(len(rankings)))

        if opts.taylor_expansion:
            node_dict_means = {}
            node_dict_stds = {}
            node_dict_ranking = {}
            for node in opts.nodes.split(","):
                print("="*20)
                print("generating ranking for node {}".format(node))
                variables = {}
                for ranking in rankings:
                    csv = pd.read_csv(ranking, sep = ",").set_index("variable")[node]
                    for var, val in csv.items():
                        if not var in variables: variables[var] = []
                        variables[var].append(val)
                mean_dict = {}
                std_dict = {}
                for v in variables: mean_dict[v] = np.median(variables[v])
                for v in variables: std_dict[v] = np.std(variables[v])
                node_dict_means[node] = mean_dict
                node_dict_stds[node] = std_dict

                var = []
                val = []
                mean = []
                std = []
                i = 0
                maxvalue = 0
                for v, m in sorted(mean_dict.iteritems(), key = lambda (k, vl): (vl, k)):
                    i+=1
                    val.append(i)
                    var.append(v)
                    mean.append(m)
                    std.append( np.std(variables[v]) )
                    print(v,m)
                    if mean[-1]+std[-1] > maxvalue: maxvalue = mean[-1]+std[-1]
                
                node_dict_ranking[node] = var[-opts.nvset :]
            top_variables = []
            for node in node_dict_ranking:
                top_variables+= node_dict_ranking[node]
            top_variables = list(set(top_variables))
            print("using {} variables for this jt region".format(len(top_variables)))

            sorted_variables[jtcat] = top_variables       

        else:
            if "BNN" in opts.inputdir:
                # collect variables and their relative importance
                variables = {}
                variables_std = {}

                for ranking in rankings:
                    csv = pd.read_csv(ranking, header = 0, sep = ",", names = ["variable", "weight_mean_sum", "weight_std_sum"])
                    sum_of_weights = csv["weight_mean_sum"].sum()

                    for row in csv.iterrows():
                        if not row[1][0] in variables: 
                            variables[row[1][0]] = []
                            if len(rankings) == 1: variables_std[row[1][0]] = []

                        variables[row[1][0]].append(row[1][1]/sum_of_weights)
                        if len(rankings) == 1: variables_std[row[1][0]].append(row[1][2]/sum_of_weights)


                # collect mean values of variables
                mean_dict = {}
                for v in variables: mean_dict[v] = np.median(variables[v])

                # generate lists sorted by mean variable importance
                var_latex = []
                var = []
                val = []
                mean = []
                std = []
                i = 0
                maxvalue = 0
                for v, m in sorted(mean_dict.iteritems(), key = lambda (k, vl): (vl, k)):
                    i += 1
                    val.append(i)
                    var_latex.append(make_nice_labels_dict[v])
                    var.append(v)
                    mean.append(m)
                    # if len(rankings) == 1: 
                    #     std.append(variables_std[v][0])
                    # else: 
                    #     std.append(np.std(variables[v]))
                    print(v,m)
                    if opts.no_std is False:
                        if mean[-1] > maxvalue: maxvalue = mean[-1]
                    else:
                        if mean[-1]+std[-1] > maxvalue: maxvalue = mean[-1]+std[-1]
                
                if not opts.nplot == -1:
                    min_value = mean[len(mean)-opts.nplot]
                else:
                    min_value = mean[0] 
                    

            else:           
                # collect variables and their relative importance
                variables = {}
                for ranking in rankings:
                    csv = pd.read_csv(ranking, header = 0, sep = ",", names = ["variable", "weight_sum"])
                    sum_of_weights = csv["weight_sum"].sum()
                    for row in csv.iterrows():
                        if not row[1][0] in variables: variables[row[1][0]] = []
                        variables[row[1][0]].append(row[1][1]/sum_of_weights)


                # collect mean values of variables
                mean_dict = {}
                for v in variables: mean_dict[v] = np.median(variables[v])

                # generate lists sorted by mean variable importance
                var_latex = []
                var = []
                val = []
                mean = []
                std = []
                i = 0
                maxvalue = 0
                for v, m in sorted(mean_dict.iteritems(), key = lambda (k, vl): (vl, k)):
                    i += 1
                    val.append(i)
                    var_latex.append(make_nice_labels_dict[v])
                    var.append(v)
                    mean.append(m)
                    std.append( np.std(variables[v]) )
                    print(v,m)
                    if opts.no_std is False:
                        if mean[-1] > maxvalue: 
                            maxvalue = mean[-1]
                        min_value = mean[0] 
                    else: 
                        if mean[-1]+std[-1] > maxvalue: 
                            maxvalue = mean[-1]+std[-1]
                
                if opts.nplot != -1:
                    min_value = mean[len(mean)-opts.nplot] 
                else:
                    min_value = mean[0] 

            sorted_variables[jtcat] = var

            if opts.plot:
                if not opts.nplot == -1:
                    mean = mean[-opts.nplot :]
                    val = val[-opts.nplot :]
                    std = std[-opts.nplot :]
                    var_latex = var_latex[-opts.nplot :]
                    var = var[-opts.nplot :]


                nvariables = len(var_latex)
                plt.rc('xtick',labelsize=20)
                plt.rc('ytick',labelsize=20)
                pylab.rcParams['ytick.major.pad']='10'
                plt.figure(figsize = [13,nvariables/2.])
                
                
                if opts.no_std is False:
                    plt.plot(mean, val, "o")
                else:
                    plt.errorbar(mean, val, xerr = std, fmt = "o")

                plt.xlim([min_value/1.03,1.03*maxvalue])
                
                #plt.grid()
                plt.yticks(val, var_latex)
                plt.xlabel("Mittelwert der Summe der Eingangsgewichte (in Prozent)", fontsize=24)
                #plt.xlabel("mean of sum of input weights (in percent)")
                plt.title(r"$\mathrm{\mathbf{CMS\ private\ work}}$", loc = "left", fontsize=24)
                plt.title(r"$\geq$ 4 jets, $\geq$ 3b - tags", loc = "right", fontsize=24)
                plt.tight_layout()
                outfile = opts.outdir+"/"+opts.filename+opts.weight_type+"_weight_sums"+opts.outdir.split("variable_ranking")[1]+".pdf"
                plt.savefig(outfile, bbox_inches='tight')
                plt.clf() 
                print("saved plot to {}".format(outfile))

                # comparison of the top nplot variables
                filename = opts.outdir+"/top_"+ str(opts.nplot)+ "_variables.csv"
                with open(filename, "w") as f:
                    headers = ["project_name", "variable_name"]
                    csv_writer = DictWriter(f,delimiter=',', lineterminator='\n',fieldnames=headers)
                    csv_writer.writeheader()
                    if not opts.nplot == -1:
                        for x in var[-opts.nplot :]:
                            csv_writer.writerow({"project_name": opts.filename, "variable_name": x})
                    else:
                        for x in var:
                            csv_writer.writerow({"project_name": opts.filename, "variable_name": x})
                print("saved top_"+ str(opts.nplot)+ "_variables.csv to" + str(filename))


    if opts.generate_variableset:
        string = "variables = {}\n"
        for jt in sorted_variables:
            string += "\nvariables[\"{}\"] = [\n".format(jt)

            variables = sorted_variables[jt]
            if opts.nvset!=-1 and not opts.taylor_expansion:
                variables = variables[-opts.nvset:]

            for v in variables:
                string += "    \"{}\",\n".format(v)

            string += "    ]\n\n"

        string += "all_variables = list(set( [v for key in variables for v in variables[key] ] ))\n"

        if opts.taylor_expansion:
            outfile = opts.outdir+"/autogenerated_taylor_expansion_1D_variableset.py"
        else:
            outfile = opts.outdir+"/autogenerated_"+opts.weight_type+"_variableset.py"

        with open(outfile,"w") as f:
            f.write(string)
        print("wrote variable set to {}".format(outfile))

else:
    filename =  opts.outdir+"/top_"+ str(opts.nplot)+ "_variables.csv"
    with open(filename, "r") as f:
        csv_reader = DictReader(f, delimiter=",")
        column_names = csv_reader.fieldnames
        variables = {}
        for row in csv_reader: 
            if row[column_names[1]] not in variables.keys():
                variables[row[column_names[1]]] = 1
            
            else:
                variables[row[column_names[1]]] += 1 
        
        for k in sorted(variables, key=variables.get, reverse=True):
            print k, variables[k]