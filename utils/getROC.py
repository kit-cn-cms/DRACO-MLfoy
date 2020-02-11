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
from ROOT import TCanvas, TMultiGraph, TGraph, TLegend, TH1F, THStack, gStyle, TH2D


# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)
import generateJTcut
#from DRACO_Frameworks.DNN.DNN import loadDNN
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

#function used to convert the JTstrings to sth that's easier to process in latex
def replace_all(text, dic):
    for i, j in dic.iteritems():
        text = text.replace(i, j)
    return text


usage = "python getROC.py [options] [jtCategories], e. g. python getROC.py -w ../workdir/  -v 'allVars','allVars_noReco','RecoVarsOnly' -n sp_v* 4j_ge3t 5j_ge3t ge6j_ge3t;\nalso I have no fucking clue why the initialization process is so slow"
parser = optparse.OptionParser(usage=usage)
parser.add_option("-w","--workdir",dest="workdir",default=basedir+"/workdir/",
    help = "path to working directory where trained DNN files are placed")
parser.add_option("-n","--naming",dest="naming",default="sp_",
    help = "naming of directories containing metrics files in subdir from workdir. add JTSTRING as placeholder\
for jet-tag categories and escaped wildcards ('\*') for multiple DNN runs. this is equal to what was given as -i for training; overall search path is workdir/{variablesets}/{naming}")
parser.add_option("-o","--output",dest="outdir",default="./",
    help = "path to output directory e.g. for plots or tables")
parser.add_option("-v","--variableset",dest="varsets",default=False,
    help = "comma seperated list of trainings/variable sets to compare the ROC-values of")
parser.add_option("-l","--latex",dest="ToLatex",default=False, action = "store_true",
    help = "renders resulting datafram to latex table")
parser.add_option("-R","--ROC",dest="GetRoc",default=False, action = "store_true",
    help = "true or false to extract ROC")
parser.add_option("-C","--ConfStats",dest="GetConfStats",default=False, action = "store_true",
    help = "true or false to extract confusion matrix stats")
parser.add_option("-p","--DoPlot",dest="DoPlot",default=False, action = "store_true",
    help = "true or false to enable plot")

#parser.add_option("-l","--latex",dest="latex",default=False,action="store_true",
    #help = "generate latex table of new variable sets")
#parser.add_option("-t", "--type", dest = "weight_type",default="absolute",
    #help = "type of variable ranking (i.e. name of weight file TYPE_weight_sum.csv), e.g. 'absolute', 'propagated', 'first_layer'")

(opts, args) = parser.parse_args()


varsets = opts.varsets.split(",")
if not os.path.exists(opts.outdir):
    os.makedirs(opts.outdir)
print(varsets)

sorted_varNames = {}
varsetNames = {}
MeanVarnames = []
initarray = np.zeros(shape=(len(args),len(varsets)))
RocValues = pd.DataFrame(data=initarray, columns = varsets)#, index=dict(list(enumerate(args))))
RocValues.rename(index=dict(list(enumerate(args))), inplace=True)

ConfStats = pd.DataFrame(data=initarray, columns = varsets)#, index=dict(list(enumerate(args))))
ConfStats.rename(index=dict(list(enumerate(args))), inplace=True)


'''
for varset in varsets:
    inputdir = opts.workdir+"/"+varset+"/"+opts.naming
    if not "JTSTRING" in inputdir:
        inputdir+="_JTSTRING/"
    print("using input directory {}".format(inputdir))
    for jtcat in args:
        print("handling category {}".format(jtcat))
        jtpath = inputdir.replace("JTSTRING",jtcat)
        netpaths = glob.glob(jtpath)
        for path in netpaths:
            print("loading DNN to produce ConfusionStatsFile afterwards")
            network = loadDNN(path, path)
            network.plot_confusionMatrix(True,True,True,True)
            del network
        
print("done")
'''
measurestringConf = "TruePositiveMean"
measurestringROC = "ROC-AUC score"
def addmethodlabel(varset):
    methodlabel=""
    if "S01" in varset:
        methodlabel += "(S)"
    elif "Merge" in varset:
        methodlabel += "(M)"
    elif "RecoVarsOnly" in varset or "allVars" in varset:
        methodlabel += "(T)"
    else: 
        pass
    return methodlabel
    
    #if "allVars" in varset:
        #binlabel+="A"
    #if "RecoVarsOnly" in varset:
        #binlabel+="B"
    #if "noReco" in varset:
        #binlabel+="C"
    #binlabel=numbering[-len(numbering)-1+nsets-yit]
    

for varset in varsets:
    inputdir = opts.workdir+"/"+varset+"/"+opts.naming
    if not "JTSTRING" in inputdir:
        inputdir+="_JTSTRING/"
    print("using input directory {}".format(inputdir))
    
    for jtcat in args:
        print("handling category {}".format(jtcat))
        jtpath = inputdir.replace("JTSTRING",jtcat)
        
        # collect weight sum files
        jtpath+= "/eval_metrics.csv"
        metrics = glob.glob(jtpath)
        print("found {} eval metrics files".format(len(metrics)))
        
        # collect weight sum files
        jtpath = inputdir.replace("JTSTRING",jtcat)
        jtpath+= "/ConfStats.csv"
        Confusions = glob.glob(jtpath)
        print("found {} confusion matrix files".format(len(Confusions)))
        
        
        if opts.GetRoc:
            # collect roc values
            measure = {}
            for metric in metrics:
                csv = pd.read_csv(metric, header = None, sep = ",")#, names = ["measure", "value"])
                for row in csv.iterrows():
                    if not row[1][0] in measure: measure[row[1][0]] = []
                    measure[row[1][0]].append(row[1][1])
        
            # collect mean values of roc
            mean_dict_Roc = {}
            for v in measure: 
                if len(measure[v]) > 1 and varset not in MeanVarnames: 
                    MeanVarnames.append(varset)
                mean_dict_Roc[v] = np.median(measure[v])
            print(mean_dict_Roc)
            #print("\n")
            try: 
                RocValues.at[jtcat,varset] = mean_dict_Roc[measurestringROC]
            except:
                RocValues.at[jtcat,varset] = "0"
                print("exception done")
        
        if opts.GetConfStats:
            measure = {}
            #print(Confusions)
            for stats in Confusions:
                #here header is 2 because there is information about confusion matrix, that is not handled properly
                #and is likely to crash the script when trying to calculate the mean values
                csv = pd.read_csv(stats, header = 1, sep = ",")#, names = ["measure", "value"])
                for row in csv.iterrows():
                    if not row[1][0] in measure: measure[row[1][0]] = []
                    measure[row[1][0]].append(row[1][1])
        
            # collect mean values of roc
            mean_dict_Conf = {}
            #print(measure)
            for v in measure: 
                if len(measure[v]) > 1 and varset not in MeanVarnames:
                    MeanVarnames.append(varset)
                mean_dict_Conf[v] = np.median(measure[v])
            print(mean_dict_Conf)
            #print("\n")
            try: 
                ConfStats.at[jtcat,varset] = mean_dict_Conf[measurestringConf]
            except:
                ConfStats.at[jtcat,varset] = "0"
                print("exception done")
        #if len(measure[v]) > 1: MeanVarnames.append(varset)
            
if not opts.GetRoc and not opts.GetConfStats:
    print("set flags -R oder -C to receive any results apart from checking for files")
if opts.GetRoc:
    print("\noverall results of mean of ROC-AUC:\n")
    print(RocValues)
#RocValues.to_hdf("tmp_out.hdf5", "a")
if opts.GetConfStats:
    print("\noverall results of mean of TruePositiveMean along diagonal of confusion matrix:\n")
    print(ConfStats)
    

if opts.ToLatex:
    #builtin function is nice idea but only partially suited for formatting:
    #table = RocValues.to_latex(float_format="\\num{{ {:0.3f} }}".format)
    
    
    ################# this here for JT categories in rows #########################
    #Hence: old-fashioned way
    #SubDict = {"4j":"FourJ", "5j":"FiveJ", "6j":"SixJ", "3t":"ThreeT", "4t":"FourT", "_":" " }
    #size = len(RocValues.columns)
    #table = "begin{tabular}{l" +str(size*"r")+"}} \n\\toprule \n{}"
    #for varset in varsets:
        #table += " & {} ".format(varset)
    
    #table += " \\ \n\midrule \n"
    #for jtcat in args:
        #table += "\\"+replace_all(jtcat,SubDict)
        #for numbers in RocValues.loc[jtcat]:
            #table += " & \\num{{ {:0.3f} }} ".format(numbers)
        #table += " \\\ \n"
    ##table += "& \\num{{ {} }} ".format(varset)
    #table += "\\bottomrule \n\\end{tabular}"
    #table.replace("0.000", "---")
    
    if opts.GetRoc and not opts.GetConfStats:
        Measure = RocValues
    if not opts.GetRoc and opts.GetConfStats:
        Measure = ConfStats
    if opts.GetRoc and opts.GetConfStats:
        sys.exit("Run either -C or -R to receive the latex table")
        
        
    nsets = len(Measure.columns)
    numbering='ABCDEFGHIJKLMN'
    
    
    Coding={}
    binlabel=[]
    for i in range(nsets):
        #Coding[numbering[i]] = Measure.columns[i]
        Coding[Measure.columns[i]] = numbering[i]+addmethodlabel(Measure.columns[i])
        binlabel.append(Coding[Measure.columns[i]])
    
    SubDict = {"4j":"FourJ", "5j":"FiveJ", "6j":"SixJ", "3t":"ThreeT", "4t":"FourT", "_":"" }
    size = len(Measure.index)
    table = "begin{tabular}{l |" +str(size*"r")+"} \n\\toprule \n{}"
    for jtcat in args:
        table += " & \\{} ".format(replace_all(jtcat,SubDict))
    
    table += " \\\ \n\midrule \n"
    for varset in varsets:
        table += "{}".format(Coding[varset])
        for numbers in Measure[varset]:
            table += " & \\num{{ {:0.3f} }}".format(numbers)
        table += " \\\ \n"
    #table += "& \\num{{ {} }} ".format(varset)
    table += "\\bottomrule \n\\end{tabular}"
    table.replace("0.000", "---")
    
    
    #print(table)

    with open(opts.outdir+"/ROCtable.tex","w+") as f:
        f.write(table)
    ## generate lists sorted by mean variable importance
    #var = []
    #varNames = []
    #mean = []
    #std = []
    #maxvalue = 0
        #for v, m in sorted(mean_dict.iteritems(), key = lambda (k, vl): (vl, k)):
            #varNames.append(v)
            #try:
                #var.append(translationFile.loc[v,"displayname"])
            #except:
                #var.append(v)
            #mean.append(m)
            #std.append( np.std(variables[v]) )
            #print(v,m)
            #if mean[-1]+std[-1] > maxvalue: maxvalue = mean[-1]+std[-1]

        #sorted_varNames[jtcat] = varNames

#if opts.plot:
    #if not opts.nplot == -1:
        #mean = mean[-opts.nplot :]
        #std = std[-opts.nplot :]
        #var = var[-opts.nplot :]

    #nvariables = len(var)

    #canvas = ROOT.TCanvas("","",nvariables*80, 1500)
    #canvas.SetBottomMargin(canvas.GetBottomMargin()*4)
    #canvas.SetLeftMargin(canvas.GetLeftMargin()*2)
    #graph = ROOT.TH1F("","",nvariables+1,1,nvariables+2)
    #graph.SetName("variableRanking")
    #graph.SetStats(False)
    #for i in range(nvariables):
        #graph.SetBinContent(nvariables-i+1, mean[i])
        #graph.SetBinError(nvariables-i+1, std[i])
        #graph.GetXaxis().SetBinLabel(nvariables-i+1, var[i])
    #graph.GetYaxis().SetTitle("importance measure")
    #graph.LabelsOption("v")
    #graph.SetTitle("")
    #graph.SetMarkerStyle(20)
    #graph.SetMarkerSize(2)
    #graph.SetMarkerColor(ROOT.kAzure-3)
    #graph.SetLineColor(ROOT.kAzure-3)
    #graph.SetLineWidth(2)
    #graph.GetXaxis().SetLabelSize(graph.GetXaxis().GetLabelSize()*20/nvariables)
    #graph.Draw("PEX0")
    #canvas.SetGridx(1)
    #canvas.RedrawAxis()
    #canvas.RedrawAxis("g")        

    #outfile = opts.outdir+"/"+opts.weight_type+"_weight_sums_"+jtcat+".pdf"
    #canvas.SaveAs(outfile)
    #print("saved plot to {}".format(outfile))
    
if opts.DoPlot:
    
    if opts.GetRoc and not opts.GetConfStats:
        Measure = RocValues
    if not opts.GetRoc and opts.GetConfStats:
        Measure = ConfStats
    if opts.GetRoc and opts.GetConfStats:
        sys.exit("Run either -C or -R to receive the matrix plot")
        
    print(MeanVarnames)
  
    root_subdict = {"4j":"4 j", "5j":"5 j", "6j":"6 j", "3t":"3 t", "4t":"4 t", "_":", ", "ge":"#geq"}
    #c = TCanvas("c","c",800, 600)
    #c.SetBottomMargin(c.GetBottomMargin()*1.5)
    #c.SetLeftMargin(c.GetLeftMargin()*1.4)

    #hs = THStack("histstack", "")
    ncats = len(args)

    #markerstyles = np.arange(0,10) + 20
    #markercolors = [ROOT.kAzure-3, ROOT.kOrange+10, ROOT.kSpring-6, ROOT.kViolet+2, ROOT.kOrange-7]
    #for j in range(len(varsets)):
        #varset=varsets[j]
        #graph = ROOT.TH1F(str(varsets),str(varsets),ncats,1,ncats+1)
        #graph.SetName(str(varset))
        #graph.SetStats(False)
        #for i in range(ncats):
            ##print(i)
            #graph.SetBinContent(i+1, RocValues[varset][i])
            ##graph.SetBinError(nvariables-i+1, std[i])
            #graph.GetXaxis().SetBinLabel(i+1, replace_all(RocValues.index[i],root_subdict))
        #graph.LabelsOption("v")
        #graph.SetTitle("")
        #graph.SetMarkerStyle(markerstyles[j])
        #graph.SetMarkerSize(3)
        #graph.SetMarkerColor(markercolors[j%5])
        #graph.SetLineColor(markercolors[j%5])
        #graph.SetLineWidth(0)
        #graph.GetXaxis().SetLabelSize(10)
        #hs.Add(graph)

    #hs.Draw()
    #hs.GetXaxis().SetLabelSize(.05)

    ##hs.Set
    ##hs.GetYAxis().SetRangeUser(0.5,0.8);
    #hs.Draw("nostack PR")

    ##mg.Draw("AP RX RY");
    #c.BuildLegend();
    #c.Update()
    ##gr1.Draw()

    ##c.Print("multigraphleg.gif");

    
        # init histogram
        
    nsets = len(Measure.columns)
    ytitle = "variable sets"
    xtitle = "categories"
    canvasName = "MatrixCanvas"
    matrix = np.array([Measure[Measure>0].min(), Measure[Measure>0].max()])
    privateWork=True
    numbering='ABCDEFGHIJKLMN'
    cm = ROOT.TH2D("Measure Matrix", "", ncats, 0, ncats, nsets, 0, nsets)
    cm.SetStats(False)
    ROOT.gStyle.SetPaintTextFormat(".3f")
    #print(ncats, nsets)
    #print(cm.GetNbinsX(), cm.GetNbinsY())
    a = Measure.transpose()
    for xit in range(cm.GetNbinsX()):
        for yit in range(cm.GetNbinsY()):
            cm.SetBinContent(xit+1,yit+1, a.iat[-yit-1, xit])
            #if has_errors:
                #cm.SetBinError(xit+1,yit+1, errors[xit, yit])

    cm.GetXaxis().SetTitle(xtitle)
    cm.GetYaxis().SetTitle(ytitle)

    cm.SetMarkerColor(ROOT.kWhite)

    minimum = np.min(matrix)
    maximum = np.max(matrix)

    cm.GetZaxis().SetRangeUser(minimum, maximum)
    
    
    Coding={}
    binlabel=[]
    for i in range(nsets):
        #Coding[numbering[i]] = Measure.columns[i]
        Coding[Measure.columns[i]] = numbering[i]+addmethodlabel(Measure.columns[i])
        binlabel.append(Coding[Measure.columns[i]])
    
    for xit in range(ncats):
        cm.GetXaxis().SetBinLabel(xit+1, replace_all(Measure.index[xit],root_subdict))
    for yit in range(nsets):
        #print(Measure.columns[yit])        
        
        #if Measure.columns[yit] in MeanVarnames:
        #    cm.GetYaxis().SetBinLabel(yit+1, "#color["+str(ROOT.kRed+1)+"]{"+binlabel[-yit-1]+"}")
        #else:
        cm.GetYaxis().SetBinLabel(yit+1, binlabel[-yit-1])#RocValues.columns[yit])
            

    cm.GetXaxis().SetLabelSize(0.05)
    cm.GetYaxis().SetLabelSize(0.05)
    cm.SetMarkerSize(2.)
    if cm.GetNbinsX()>6:
        cm.SetMarkerSize(1.5)
    if cm.GetNbinsX()>8:
        cm.SetMarkerSize(1.)

    canvas = ROOT.TCanvas(canvasName, canvasName, int(1024./5*ncats), 2400)
    canvas.SetTopMargin(0.15)
    canvas.SetBottomMargin(0.18)
    canvas.SetRightMargin(0.15)
    canvas.SetLeftMargin(0.15)
    canvas.SetTicks(1,1)

    # draw histogram
    ROOT.gStyle.SetPalette(ROOT.kViridis)
    draw_option = "colz text1"
    #if ROCerr: draw_option += "e"
    cm.DrawCopy(draw_option)

    # setup TLatex
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextColor(ROOT.kBlack)
    latex.SetTextSize(0.03)

    l = canvas.GetLeftMargin()
    r = canvas.GetRightMargin()
    t = canvas.GetTopMargin()

    # add category label
    #latex.DrawLatex(l,1.-t+0.01, catLabel)

    if privateWork:
        latex.DrawLatex(l, 1.-t+0.01, "CMS private work")
    canvas.Update()
    if opts.GetRoc:
        #latex.SetBBoxX2(1.-r*0.05)
        measurestringROC = "ROC-AUC"
        latex.SetTextAlign(31)
        latex.DrawLatex(1.-r*1.05, 1.-t+0.01, measurestringROC)
        canvas.SaveAs(opts.outdir+"/combination_"+measurestringROC.replace(" ","")+".png")
        canvas.SaveAs(opts.outdir+"/combination_"+measurestringROC.replace(" ","")+".pdf")
    if opts.GetConfStats:
        measurestringConf="STD of True Positives"
        latex.SetTextAlign(31)
        latex.DrawLatex(1.-r*1.05, 1.-t+0.01, measurestringConf)
        canvas.SaveAs(opts.outdir+"/combination_"+measurestringConf.replace(" ","")+".png")
        canvas.SaveAs(opts.outdir+"/combination_"+measurestringConf.replace(" ","")+".pdf")
    # add ROC score if activated
    
    

    print(Coding)
    print(MeanVarnames)

    raw_input("press <ent>")
'''

#if opts.generate_variableset:
    #string = "variables = {}\n"
    #for jt in sorted_varNames:
        #string += "\nvariables[\"{}\"] = [\n".format(jt)

        #variables = sorted_varNames[jt]
        #if not opts.nvset==-1:
            #variables = variables[-opts.nvset:]

        #for v in variables:
            #string += "    \"{}\",\n".format(v)

        #string += "    ]\n\n"

    #string += "all_variables = list(set( [v for key in variables for v in variables[key] ] ))\n"

    #outfile = opts.outdir+"/autogenerated_"+opts.weight_type+"_variableset.py"
    #with open(outfile,"w") as f:
        #f.write(string)
    #print("wrote variable set to {}".format(outfile))

    #if opts.latex:
        #jtRegions = list(sorted([jt for jt in sorted_varNames]))
        #table = "\\begin{tabular}{l|"+"r"*len(sorted_varNames)+"}\n"
        #table+= "\\midrule\n"
        #table+= "       & "+" & ".join([generateJTcut.getJTlabel(jt).replace("\\geq","$\\geq$").replace("\\leq","$\\leq$") for jt in jtRegions])+"\\\\\n"
        
        ## join all variables
        #tableVariables = {}
        #allVariables = []
        #for jt in jtRegions:
            #variables = sorted_varNames[jt]
            #if not opts.nvset==-1:
                #tableVariables[jt] = variables[-opts.nvset:]
            #else:
                #tableVariables[jt] = variables
            #allVariables+=tableVariables[jt]
        #allVariables = list(sorted(set(allVariables)))
        
        #foundNoLatex = False
        #for v in allVariables:
            #try:
                #table += makeLatexCompatible(translationFile.loc[v,"displayname"])
            #except:
                #table += str(v) + "NOLATEX"
                #foundNoLatex = True
            #for jt in jtRegions:
                #table += " & "
                #if v in tableVariables[jt]:
                    #table += " \\checkmark "
                #else:
                    #table += " --- "
            #table += "\\\\\n"
        #table += "\\midrule\n"
        #table += "\\end{tabular}"
        #outfile = opts.outdir+"/variableSetTable.txt"
        #with open(outfile, "w") as f:
            #f.write(table)
        #print("wrote latex table to {}".format(outfile))
        #if foundNoLatex:
            #print("there were variables with no explicist definition for latex code")
'''

