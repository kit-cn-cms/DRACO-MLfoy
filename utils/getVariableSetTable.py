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
parser.add_option("-v",dest="variable_set",
    help = "path to variable set")
(opts, args) = parser.parse_args()

sys.path.append(os.path.dirname(opts.variable_set))
varset = __import__(os.path.basename(opts.variable_set).replace(".py",""))


jtRegions = list(sorted(args))
table = "\\begin{tabular}{l|"+"r"*len(jtRegions)+"}\n"
table+= "\\midrule\n"
table+= "       & "+" & ".join([generateJTcut.getJTlabel(jt).replace("\\geq","$\\geq$").replace("\\leq","$\\leq$") for jt in jtRegions])+"\\\\\n"

# join all variables
tableVariables = {}
allVariables = []
for jt in jtRegions:
    variables = varset.variables[jt]
    tableVariables[jt] = variables
    allVariables+=tableVariables[jt]
allVariables = list(sorted(set(allVariables)))

for v in allVariables:
    table += translationFile.loc[v,"texname"]
    for jt in jtRegions:
        table += " & "
        if v in tableVariables[jt]:
            table += " \\checkmark "
        else:
            table += " --- "
    table += "\\\\\n"
table += "\\midrule\n"
table += "\\end{tabular}"
outfile = opts.variable_set.replace(".py",".txt")
with open(outfile, "w") as f:
    f.write(table)
print("wrote latex table to {}".format(outfile))

