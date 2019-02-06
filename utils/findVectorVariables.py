import optparse
import os
import sys
import uproot

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

import variable_sets.ntuplesVariables as variable_set

parser = optparse.OptionParser(usage="%prog [options]")
parser.add_option("-f","--file",dest="file",metavar="FILE",
    help="ntuple file to check variables")
parser.add_option("-n",dest="number_of_indices",metavar="NINDICES",default=4,
    help="determine how many indices should be considered")
parser.add_option("--generateSet",action="store_true",dest="generateSet",default=False,
    help="generate printout for generating a variable_set.py file")
parser.add_option("-o",dest="outputfile",
    help="specify output file for variable_set.py file")
(opts, args) = parser.parse_args()

if not opts.file:
    parser.error("need to specify file")

# get all variables
variables = variable_set.all_variables

new_set_of_variables = []
# open sample file
with uproot.open(opts.file) as f:
    tree = f["MVATree"]

    for v in variables:
        print("looking at variable: {}".format(v))
        df = tree.pandas.df([v])
        if "subentry" in df.index.names:
            new_set_of_variables += [v+"[{}]".format(i) for i in range(opts.number_of_indices)]
        else:
            new_set_of_variables += [v]

if opts.generateSet:
    out = "variables = {}\n"
    for jt in ["4j_ge3t", "5j_ge3t", "ge6j_ge3t"]:
        out += "variables[\""+jt+"\"] = [\n"
        for v in new_set_of_variables:
            out += "    '"+str(v)+"',\n"
        out += "    ]\n\n\n"

    out += "all_variables = set( [v for key in variables for v in variables[key] ] )"
    if opts.outputfile:
        with open(opts.outputfile,"w") as of:
            of.write(out)
        print("output generated at {}".format(opts.outputfile))
    else:
        print("-"*50)
        print("variable set printout:")
        print(out)
        print("-"*50)




