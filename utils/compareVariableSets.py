import sys
import os
import optparse
import pandas as pd

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)
import generateJTcut
nameConfig = basedir+"/pyrootsOfTheCaribbean/plot_configs/variableConfig.csv"
translationFile = pd.read_csv(nameConfig, sep = ",").set_index("variablename", drop = True)

parser = optparse.OptionParser()
parser.add_option("-a",dest="variable_set1",
    help = "path to first variable set")
parser.add_option("-b",dest="variable_set2",
    help = "path to second variable set")
parser.add_option("-o",dest="outpath", default="./",
    help = "output path relative to current directory, name of outfile is set automatically")
parser.add_option("-l",dest="latex", action="store_true", default=False,
    help = "toggle to generate latex table, outpath needed")
(opts, args) = parser.parse_args()

sys.path.append(os.path.dirname(opts.variable_set1))
set1 = __import__(os.path.basename(opts.variable_set1).replace(".py",""))
set2 = __import__(os.path.basename(opts.variable_set2).replace(".py",""))
set1_name = str(opts.variable_set1)
set2_name = str(opts.variable_set2)


# specify the compared variable sets
#import variable_sets.allVars_comb_S01 as set1
#import variable_sets.topVariables_L as set1
#set2_name = "pre christmas tight cut variables"
#import variable_sets.topVariables_T as set2
#import variable_sets.topVariables_validated_decorr as set2

vars1 = set1.variables
vars2 = set2.variables

for key in vars1:
    print("\n\n\n"+"-"*40)
    print("CATEGORY: {}".format(key))
    
    variables = list(set(vars1[key] + vars2[key]))
    common = []
    only_set1 = []
    only_set2 = []

    for v in variables:
        if v not in vars2[key]:
            only_set1.append(v)
        elif v not in vars1[key]:
            only_set2.append(v)
        else:
            common.append(v)

    print("\n\tvariables only in {}".format(set1_name))
    for v in only_set1: print(v)

    print("\n\tvariables only in {}".format(set2_name))
    for v in only_set2: print(v)

    print("\n\tcommon variables")
    for v in common: print(v)



if opts.latex:
    jtRegions = list(sorted(args))

    for jt in jtRegions:
        table = "\\begin{tabular}{l|"+"r"*2+"}\n"
        table+= "\\toprule\n"
        table+= generateJTcut.getJTlabel(jt).replace("\\geq","$\\geq$").replace("\\leq","$\\leq$")+" & "+set1_name + " & " +set2_name+"\\\\\n"
        table+= " \midrule"

        # join all variables
        tableVariables = {}
        allVariables = []
        

        variables1 = vars1[jt]
        #tableVariables[jt] = variables
        variables2 = vars2[jt]
        allVariables = list(sorted(set(variables1+variables2)))

        for v in allVariables:
            try:
                table += translationFile.loc[v,"texname"]
            except:
                table += v.replace("_","\_")
            for sets in [variables1, variables2]:
                table += " & "
                if v in sets:
                    table += " \\checkmark "
                else:
                    table += " --- "
            table += "\\\\\n"
        table += "\\midrule\n"
        table += "\\end{tabular}"
        table = table.replace("\DeltaR","\Delta R")
        #outfile = opts.variable_set.replace(".py",".txt")
        outfile = opts.outpath+"comparison_table_"+str(jt)+".txt"
        with open(outfile, "w") as f:
            f.write(table)
        print("wrote latex table to {}".format(outfile))

