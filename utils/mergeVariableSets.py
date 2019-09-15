import sys
import os
import optparse
import imp

parser = optparse.OptionParser()
parser.add_option("--jt", dest="jtregions",
    help = "comma separated list of jet tag regions",
    default = "4j_ge3t,5j_ge3t,ge6j_ge3t")
parser.add_option("-o", dest = "output",        
    help = "output path of new variable set",
    default = "merged_variableset.py")
(opts, args) = parser.parse_args()


varsets = []
for i, vset in enumerate(args):
    varsets.append(imp.load_source("module{}".format(i),vset))


string = "variables = {}\n"
for jt in opts.jtregions.split(","):
    string += "\nvariables[\"{}\"] = [\n".format(jt)

    variables = []
    for vset in varsets:
        variables += vset.variables[jt]

    variables = sorted(list(set(variables)))

    for v in variables:
        string += "    \"{}\",\n".format(v)

    string += "    ]\n\n"

string += "all_variables = list(set( [v for key in variables for v in variables[key] ] ))\n"

with open(opts.output,"w") as f:
    f.write(string)
print("wrote variable set to {}".format(opts.output))

