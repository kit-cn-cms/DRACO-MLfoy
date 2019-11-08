import optparse
import os
import sys
import ROOT
import uproot

parser = optparse.OptionParser(usage="%prog [options]")
parser.add_option("-f","--file",dest="file",metavar="FILE",
    help="ntuple file to check variables")
parser.add_option("-n",dest="number_of_indices",metavar="NINDICES",default=4,
    help="determine how many indices should be considered")
parser.add_option("--generateSet",action="store_true",dest="generateSet",default=False,
    help="generate printout for generating a variable_set.py file")
parser.add_option("-o",dest="outputfile",
    help="specify output file for variable_set.py file")
parser.add_option("-j",dest="jtregions",help="jet-tag regions (comma separated)",default="ge4j_ge3t")
(opts, args) = parser.parse_args()

if not opts.file:
    parser.error("need to specify file")
opts.number_of_indices = int(opts.number_of_indices)



def getAllVariables(infile):
    f = ROOT.TFile(infile)
    t = f.Get("MVATree")
    variables = []
    vetos = ["Weight", "HTXS", "Gen", "Prescale", 
            "Electron", "Muon", "AK8", 
            "Triggered", "DeepFlavour", 
            "common5_output", "Untagged", 
            "CSV_DNN", "PileUp", "Evt_ID", 
            "Evt_Lumi", "Evt_Odd", "Evt_Run",
            "PartonFlav", "GoodTags",]
    for b in list(t.GetListOfBranches()):
        ignore = False
        for veto in vetos:
            if veto in b.GetName(): ignore = True
        if not ignore: variables.append(b.GetName())

    return variables


def figureOutVectors(variables, infile):
    new_variables = []
    with uproot.open(infile) as f:
        tree = f["MVATree"]

        for v in variables:
            print("looking at variable: {}".format(v))
            df = tree.pandas.df([v])
            if "subentry" in df.index.names:
                if "LooseLepton" in v:
                    new_variables += [v+"[0]"]
                elif "Jet" in v or "CSV" in v:
                    new_variables += [v+"[{}]".format(i) for i in range(opts.number_of_indices)]
                else:   
                    print("vector variable {} did not match name query".format(v))
            else:
                new_variables += [v]

    return new_variables


def generateVariableSet(variables, categories, path):
    out = "variables = {}\n"
    for cat in categories:
        out += "variables[\"{}\"] = [\n".format(cat)
        for v in variables:
            out += "    '{}',\n".format(v)
        out += "    ]\n\n"

    out += "all_variables = list(set( [v for key in variables for v in variables[key] ] ))"
    
    with open(path,"w") as f:
        f.write(out)
    print("generated new variable set at {}".format(path))






variables = getAllVariables(opts.file)
new_variables = figureOutVectors(variables, opts.file)
if opts.generateSet:
    generateVariableSet(new_variables, opts.jtregions.split(","), opts.outputfile)
else:
    print("variables:")
    for v in variables: print(v)





