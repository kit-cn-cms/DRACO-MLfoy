import ROOT
import os
import sys
import pandas
import numpy as np
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(os.path.dirname(filedir))
sys.path.append(basedir)

import utils.generateJTcut as JTcut
import plot_configs.variableConfig as binning
import plot_configs.setupPlots as setup

class Sample:
    def __init__(self, sampleName, sampleFile, signalSample = False, filled = None, XSscaling = 1., plotColor = None, apply_cut = True, maxEntries = None):
        self.sampleName = sampleName
        self.sampleFile = sampleFile
        self.isSignal   = signalSample
        self.applyCut   = apply_cut
        self.XSScale    = XSscaling
        self.filled     = filled
        self.stop       = None if not maxEntries else int(maxEntries)

        self.plotColor  = plotColor
        if self.plotColor == None:
            try:
                self.plotColor = setup.GetPlotColor(self.sampleName)
            except:
                print("no color for sample chosen + sample not in color dictionary")
                self.plotColor = 1

        if self.filled == None:
            self.filled = not signalSample

        self.load()
        self.cut_data   = {}

    def load(self):
        with pandas.HDFStore(self.sampleFile, mode = "r") as store:
            self.data = store.select("data", stop = self.stop)
        print("\tnevents: {}".format(self.data.shape[0]))
        # hack
        self.data["Weight_XS"] = self.data["Weight_XS"].astype(float)

    def cutData(self, cut, variables, lumi_scale):
        # if lumi scale was set to zero set scale to 1
        scale = lumi_scale
        if lumi_scale == 0:
            scale = 1.

        if not self.applyCut or cut in ["inclusive", "SL"]:
            self.cut_data[cut] = self.data
            self.cut_data[cut] = self.cut_data[cut].assign(weight = lambda x: x.Weight_XS*x.Weight_GEN_nom*scale)
            return

        # cut events according to JT category
        category_cut = JTcut.getJTstring(cut)

        # only save variables that are needed
        variables += ["Weight_XS", "Weight_GEN_nom"]
        self.cut_data[cut] = self.data.query(category_cut)
        self.cut_data[cut] = self.cut_data[cut][list(set(variables))]

        # add weight entry for scaling
        self.cut_data[cut] = self.cut_data[cut].assign(weight = lambda x: x.Weight_XS*x.Weight_GEN_nom*scale*self.XSScale)
            
        

class variablePlotter:
    def __init__(self, output_dir, variable_set, add_vars = [], ignored_vars = [], max_entries = None, plotOptions = {}):
        self.output_dir     = output_dir
        self.variable_set   = variable_set
        self.add_vars       = list(add_vars)
        self.ignored_vars   = list(ignored_vars)
        self.max_entries    = max_entries

        self.samples        = {}
        self.ordered_stack  = []
        self.categories     = []

        # handle options
        defaultOptions = {
            "ratio":        False,
            "ratioTitle":   None,
            "logscale":     False,
            "scaleSignal":  -1,
            "privateWork":  False,
            "KSscore":      False}

        for key in plotOptions:
            defaultOptions[key] = plotOptions[key]
        self.options = defaultOptions
        

    def addSample(self, **kwargs):
        print("adding sample: "+str(kwargs["sampleName"]))
        kwargs["maxEntries"] = self.max_entries
        self.samples[kwargs["sampleName"]] = Sample(**kwargs)
        if not self.samples[kwargs["sampleName"]].isSignal:
            self.ordered_stack.append(kwargs["sampleName"])

    def addCategory(self, category):
        print("adding category: {}".format(category))
        self.categories.append(category)

    def getAllVariables(self):
        variables = []
        for key in self.samples:
            variables = list(self.samples[key].data.columns.values)
        variables = list(set(variables+self.add_vars))
        return variables

    def plot(self, saveKSValues = False):
        # loop over categories and get list of variables
        for cat in self.categories:
            print("starting with category {}".format(cat))

            cat_dir = self.output_dir+"/"+cat+"/"
            if not os.path.exists(cat_dir):
                os.makedirs(cat_dir)
        
            if saveKSValues:
                ks_file = self.output_dir+"/"+cat+"_KSvalues.csv"
                ks_dict = {}

            # if no variable_set is given, plot all variables in samples
            if self.variable_set == None:
                variables = self.getAllVariables()
            # load list of variables from variable set
            elif cat in self.variable_set.variables:
                variables = self.variable_set.variables[cat] + self.add_vars
            else:
                variables = self.variable_set.all_variables + self.add_vars

            # filter events according to JT category
            for key in self.samples:
                self.samples[key].cutData(cat, variables, self.options["lumiScale"])

            # loop over all variables and perform plot each time
            for variable in variables:
                if variable in self.ignored_vars: continue
                print("plotting variable: {}".format(variable))

                # generate plot output name
                plot_name = cat_dir + "/{}.pdf".format(variable)
                plot_name = plot_name.replace("[","_").replace("]","")
                    
                # generate plot
                histInfo = self.histVariable(
                    variable    = variable,
                    plot_name   = plot_name,
                    cat         = cat)

                if saveKSValues:
                    ks_dict[variable] = histInfo["KSScore"]

            if saveKSValues:
                with open(ks_file, "w") as f:
                    for key, value in sorted(ks_dict.iteritems(), key = lambda (k,v): (v,k)):
                        f.write("{},{}\n".format(key, value))


    def histVariable(self, variable, plot_name, cat):
        histInfo = {}

        # get number of bins and binrange from config file
        bins = binning.getNbins(variable)
        bin_range = binning.getBinrange(variable)

        # check if bin_range was found
        if not bin_range:
            maxValue = max([max(self.samples[sample].cut_data[cat][variable].values) for sample in self.samples])
            minValue = min([min(self.samples[sample].cut_data[cat][variable].values) for sample in self.samples])
            config_string = "variables[\""+variable+"\"]\t\t\t= Variable(bin_range = [{},{}])\n".format(minValue, maxValue)
            with open("new_variable_configs.txt", "a") as f:
                f.write(config_string)
            bin_range = [minValue, maxValue]

        histInfo["nbins"] = bins
        histInfo["range"] = bin_range

        bkgHists = []
        bkgLabels = []
        weightIntegral = 0

        # loop over backgrounds and fill hists
        for sampleName in self.ordered_stack:
            sample = self.samples[sampleName]

            # get weights
            weights = sample.cut_data[cat]["weight"].values
            # get values
            values = sample.cut_data[cat][variable].values

            #weights = [weights[i] for i in range(len(weights)) if not np.isnan(values[i])]
            #values =  [values[i]  for i in range(len(values))  if not np.isnan(values[i])]

            weightIntegral += sum(weights)
        
            # setup histogram
            hist = setup.setupHistogram(
                values      = values,
                weights     = weights,
                nbins       = bins,
                bin_range   = bin_range,
                color       = sample.plotColor,
                xtitle      = cat+"_"+sample.sampleName+"_"+variable,
                ytitle      = setup.GetyTitle(self.options["lumiScale"]),
                filled      = sample.filled)

            bkgHists.append(hist)
            bkgLabels.append(sample.sampleName)

        sigHists = []
        sigLabels = []
        sigScales = []
        
        # if not background was added, the weight integral is equal to 0
        if weightIntegral == 0:
            self.options["scaleSignal"] = 0   
        histInfo["bkgYield"] = weightIntegral

        # scale stack to one if lumiScale is set to zero
        if self.options["lumiScale"] == 0:
            for hist in bkgHists:
                hist.Scale(1./weightIntegral)
            weightIntegral = 1.

        # loop over signals and fill hists
        for key in self.samples:
            sample = self.samples[key]
            if not sample.isSignal: continue

            # get weights
            weights = sample.cut_data[cat]["weight"].values
            # determine scale factor
            if self.options["scaleSignal"] == -1:
                scaleFactor = weightIntegral/(sum(weights)+1e-9)
            elif self.options["scaleSignal"] == 0:
                scaleFactor = (1./(sum(weights)+1e-9))
            else:
                scaleFactor = float(self.options["scaleSignal"])

            # setup histogram
            hist = setup.setupHistogram(
                values      = sample.cut_data[cat][variable].values,
                weights     = weights,
                nbins       = bins,
                bin_range   = bin_range,
                color       = sample.plotColor,
                xtitle      = cat+"_"+sample.sampleName+"_"+variable,
                ytitle      = setup.GetyTitle(),
                filled      = sample.filled)

            hist.Scale(scaleFactor)

            sigHists.append(hist)
            sigLabels.append(sample.sampleName)
            sigScales.append(scaleFactor)

        # init canvas
        canvas = setup.drawHistsOnCanvas(
            sigHists, bkgHists, self.options,   
            canvasName = variable)

        # setup legend
        legend = setup.getLegend()
        # add signal entriesa
        for iSig in range(len(sigHists)):
            labelstring = sigLabels[iSig]       
            if not self.options["lumiScale"] == 0.:
                labelstring = sigLabels[iSig]+" x {:4.0f}".format(sigScales[iSig])

            # add KS score to label if activated
            if self.options["KSscore"]:
                KSscore = setup.calculateKSscore(bkgHists[0],sigHists[iSig])
                labelstring="#splitline{"+labelstring+"}{KSscore = %.3f}"%(KSscore)
                histInfo["KSScore"] = KSscore
                
            legend.AddEntry(sigHists[iSig], labelstring, "L")

        # add background entries
        for iBkg in range(len(bkgHists)):
            legend.AddEntry(bkgHists[iBkg], bkgLabels[iBkg], "F")

        # draw loegend
        legend.Draw("same")

        # add lumi and category to plot
        setup.printLumi(canvas, lumi = self.options["lumiScale"], ratio = self.options["ratio"])
        setup.printCategoryLabel(canvas, JTcut.getJTlabel(cat), ratio = self.options["ratio"])
        if self.options["privateWork"]: 
            setup.printPrivateWork(canvas, ratio = self.options["ratio"])

        # save canvas
        setup.saveCanvas(canvas, plot_name)
    
        return histInfo


