# global imports
import ROOT
import os
import sys
import pandas as pd
import glob
from collections import Counter
import operator
import numpy as np
import optparse


# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)
import generateJTcut
nameConfig = basedir+"/pyrootsOfTheCaribbean/plot_configs/variableConfig.csv"
translationFile = pd.read_csv(nameConfig, sep = ",").set_index("variablename", drop = True)

def makeLatexCompatible(text):
    text = text.replace("#Delta","$\\Delta$")
    text = text.replace("#eta","$\\eta$")
    text = text.replace("#phi","$\\phi$")
    text = text.replace("p_{T}","$p_{\\text{T}}$")
    text = text.replace("H_{T}","$H_{\\text{T}}$")
    text = text.replace("#chi^{2}","$\\chi^{2}$")
    text = text.replace("M_{2}","$M_{2}$")
    text = text.replace("t_{had}","$\\text{t}_{\\text{had}}$")
    text = text.replace("b_{had}","$\\text{b}_{\\text{had}}$")
    text = text.replace("W_{had}","$\\text{W}_{\\text{had}}$")
    text = text.replace("t_{lep}","$\\text{t}_{\\text{lep}}$")
    text = text.replace("b_{lep}","$\\text{b}_{\\text{lep}}$")
    text = text.replace("W_{lep}","$\\text{W}_{\\text{lep}}$")

    text = text.replace("$$","")
    return text



usage = "python getTopVariables.py [options] [jtCategories]"
parser = optparse.OptionParser(usage=usage)
parser.add_option("-w","--workdir",dest="workdir",default=basedir+"/workdir/",
    help = "path to working directory where trained DNN files are placed")
parser.add_option("-i","--input",dest="inputdir",default="test_training",
    help = "path to DNN directories relative to WORKDIR. add JTSTRING as placeholder\
for jet-tag categories and escaped wildcards ('\*') for multiple DNN runs.")
parser.add_option("-o","--output",dest="outdir",default="./",
    help = "path to output directory e.g. for plots or variable sets")
parser.add_option("-p","--plot",dest="plot",default=False,action="store_true",
    help = "generate plots of variable rankings")
parser.add_option("--nplot",dest="nplot",default=30,type=int,
    help = "number of variables to be plotted (-1 for all).")
parser.add_option("-v","--variableset",dest="generate_variableset",default=False,action="store_true",
    help = "generate new variable set from variable rankings")
parser.add_option("--nvset",dest="nvset",default=30,type=int,
    help = "number of variables written to variable set (-1 for all).")
parser.add_option("-l","--latex",dest="latex",default=False,action="store_true",
    help = "generate latex table of new variable sets")
parser.add_option("-t", "--type", dest = "weight_type",default="absolute",
    help = "type of variable ranking (i.e. name of weight file TYPE_weight_sum.csv), e.g. 'absolute', 'propagated', 'first_layer'")

(opts, args) = parser.parse_args()
inputdir = opts.workdir+"/"+opts.inputdir
if not "JTSTRING" in inputdir:
    inputdir+="_JTSTRING/"
print("using input directory {}".format(inputdir))

if not os.path.exists(opts.outdir):
    os.makedirs(opts.outdir)


sorted_varNames = {}
for jtcat in args:
    print("\n\nhandling category {}\n\n".format(jtcat))
    jtpath = inputdir.replace("JTSTRING",jtcat)
    
    # collect weight sum files
    jtpath+= "/"+opts.weight_type+"_weight_sums.csv"
    rankings = glob.glob(jtpath)
    print("found {} variable ranking files".format(len(rankings)))

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
    var = []
    varNames = []
    mean = []
    std = []
    maxvalue = 0
    for v, m in sorted(mean_dict.iteritems(), key = lambda (k, vl): (vl, k)):
        varNames.append(v)
        var.append(translationFile.loc[v,"displayname"])
        mean.append(m)
        std.append( np.std(variables[v]) )
        print(v,m)
        if mean[-1]+std[-1] > maxvalue: maxvalue = mean[-1]+std[-1]

    sorted_varNames[jtcat] = varNames

    if opts.plot:
        if not opts.nplot == -1:
            mean = mean[-opts.nplot :]
            std = std[-opts.nplot :]
            var = var[-opts.nplot :]

        nvariables = len(var)

        canvas = ROOT.TCanvas("","",nvariables*50, 1500)
        canvas.SetBottomMargin(canvas.GetBottomMargin()*5)
        canvas.SetLeftMargin(canvas.GetLeftMargin()*2)
        graph = ROOT.TH1F("","",nvariables+2,1,nvariables+3)
        graph.SetName("variableRanking")
        graph.SetStats(False)
        for i in range(nvariables):
            graph.SetBinContent(nvariables-i+2, mean[i])
            graph.SetBinError(nvariables-i+2, std[i]+0.001)
            graph.GetXaxis().SetBinLabel(nvariables-i+2, var[i])
        graph.GetYaxis().SetTitle("mean of sum of input weights")
        graph.LabelsOption("v")
        graph.SetTitle("")
        graph.SetMarkerStyle(20)
        graph.SetMarkerSize(2)
        graph.SetMarkerColor(ROOT.kAzure-3)
        graph.SetLineColor(ROOT.kAzure-3)
        graph.Draw("PEX0")
        canvas.SetGridx(1)
        canvas.RedrawAxis()
        canvas.RedrawAxis("g")        

        outfile = opts.outdir+"/"+opts.weight_type+"_weight_sums_"+jtcat+".pdf"
        canvas.SaveAs(outfile)
        print("saved plot to {}".format(outfile))

if opts.generate_variableset:
    string = "variables = {}\n"
    for jt in sorted_varNames:
        string += "\nvariables[\"{}\"] = [\n".format(jt)

        variables = sorted_varNames[jt]
        if not opts.nvset==-1:
            variables = variables[-opts.nvset:]

        for v in variables:
            string += "    \"{}\",\n".format(v)

        string += "    ]\n\n"

    string += "all_variables = list(set( [v for key in variables for v in variables[key] ] ))\n"

    outfile = opts.outdir+"/autogenerated_"+opts.weight_type+"_variableset.py"
    with open(outfile,"w") as f:
        f.write(string)
    print("wrote variable set to {}".format(outfile))

    if opts.latex:
        jtRegions = list(sorted([jt for jt in sorted_varNames]))
        table = "\\begin{tabular}{l|"+"r"*len(sorted_varNames)+"}\n"
        table+= "\\midrule\n"
        table+= "       & "+" & ".join([generateJTcut.getJTlabel(jt).replace("\\geq","$\\geq$").replace("\\leq","$\\leq$") for jt in jtRegions])+"\\\\\n"
        
        # join all variables
        tableVariables = {}
        allVariables = []
        for jt in jtRegions:
            variables = sorted_varNames[jt]
            if not opts.nvset==-1:
                tableVariables[jt] = variables[-opts.nvset:]
            else:
                tableVariables[jt] = variables
            allVariables+=tableVariables[jt]
        allVariables = list(sorted(set(allVariables)))

        for v in allVariables:
            table += makeLatexCompatible(translationFile.loc[v,"displayname"])
            for jt in jtRegions:
                table += " & "
                if v in tableVariables[jt]:
                    table += " \\checkmark "
                else:
                    table += " --- "
            table += "\\\\\n"
        table += "\\midrule\n"
        table += "\\end{tabular}"
        outfile = opts.outdir+"/variableSetTable.txt"
        with open(outfile, "w") as f:
            f.write(table)
        print("wrote latex table to {}".format(outfile))



