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
    def __init__(self, sampleName, sampleFile, signalSample = False, plotColor = None, apply_cut = True):
        self.sampleName = sampleName
        self.sampleFile = sampleFile
        self.isSignal   = signalSample
        self.applyCut   = apply_cut

        self.plotColor  = plotColor
        if self.plotColor == None:
            try:
                self.plotColor = setup.GetPlotColor(self.sampleName)
            except:
                print("no color for sample chosen + sample not in color dictionary")
                self.plotColor = 1

        self.load()
        self.cut_data   = {}

    def load(self):
        with pandas.HDFStore(self.sampleFile, mode = "r") as store:
            self.data = store.select("data", stop = 1000000)
        print("\tnevents: {}".format(self.data.shape[0]))

    def cutData(self, cut, variables, lumi_scale):
        if not self.applyCut or cut in ["inclusive", "SL"]:
            self.cut_data[cut] = self.data
            self.cut_data[cut] = self.cut_data[cut].assign(weight = lambda x: x.Weight_XS*x.Weight_GEN_nom*lumi_scale)
            return

        # cut events according to JT category
        category_cut = JTcut.getJTstring(cut)

        # only save variables that are needed
        self.cut_data[cut] = self.data.query(category_cut)[list(set(variables+["Weight_XS", "Weight_GEN_nom"]))]

        # add weight entry for scaling
        self.cut_data[cut] = self.cut_data[cut].assign(weight = lambda x: x.Weight_XS*x.Weight_GEN_nom*lumi_scale)
            
        

class variablePlotter:
    def __init__(self, output_dir, variable_set, add_vars, plotOptions = {}):
        self.output_dir     = output_dir
        self.variable_set   = variable_set
        self.add_vars       = list(add_vars)        

        self.samples        = {}
        self.ordered_stack  = []
        self.categories     = []

        # handle options
        defaultOptions = {
            "ratio":        False,
            "ratioTitle":   None,
            "logscale":     False,
            "scaleSignal":  -1,
            "KSscore":      False}

        for key in plotOptions:
            defaultOptions[key] = plotOptions[key]
        self.options = defaultOptions
        

    def addSample(self, **kwargs):
        print("adding sample: "+str(kwargs["sampleName"]))
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

    def plot(self):
        # loop over categories and get list of variables
        for cat in self.categories:
            print("starting with category {}".format(cat))

            cat_dir = self.output_dir+"/"+cat+"/"
            if not os.path.exists(cat_dir):
                os.makedirs(cat_dir)

            # if no variable_set is given, plot all variables in samples
            if self.variable_set == None:
                variables = self.getAllVariables()
            # load list of variables from variable set
            elif cat in self.variable_set.variables:
                variables = list(set(self.variable_set.variables[cat] + self.add_vars))
            else:
                variables = list(set(self.variable_set.variables + self.add_vars))

            # filter events according to JT category
            for key in self.samples:
                self.samples[key].cutData(cat, variables, self.options["lumiScale"])

            # loop over all variables and perform plot each time
            for variable in variables:
                print("plotting variable: {}".format(variable))

                # generate plot output name
                plot_name = cat_dir + "/{}.pdf".format(variable)
                plot_name = plot_name.replace("[","_").replace("]","")
                    
                # generate plot
                self.histVariable(
                    variable    = variable,
                    plot_name   = plot_name,
                    cat         = cat)

    def histVariable(self, variable, plot_name, cat):
        # get number of bins and binrange from config file
        bins = binning.getNbins(variable)
        bin_range = binning.getBinrange(variable)

        # check if bin_range was found
        if not bin_range:
            maxValue = -999
            minValue = 999
            for key in self.samples:
                maxValue = max(maxValue, max(self.samples[key].cut_data[cat][variable].values))
                minValue = min(minValue, min(self.samples[key].cut_data[cat][variable].values))
            config_string = "variables[\""+variable+"\"]\t\t\t= Variable(bin_range = [{},{}])\n".format(minValue, maxValue)
            with open("new_variable_configs.txt", "a") as f:
                f.write(config_string)
            bin_range = [minValue, maxValue]

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
                ytitle      = setup.GetyTitle(),
                filled      = True)

            bkgHists.append(hist)
            bkgLabels.append(sample.sampleName)

        sigHists = []
        sigLabels = []
        sigScales = []
        
        # if not background was added, the weight integral is equal to 0
        if weightIntegral == 0:
            self.options["scaleSignal"] = 0   

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
                filled      = False)

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
            labelstring = sigLabels[iSig]+" x {:4.0f}".format(sigScales[iSig])

            # add KS score to label if activated
            if self.options["KSscore"]:
                labelstring="#splitline{"+labelstring+"}{KSscore = %.3f}"%(setup.calculateKSscore(bkgHists[0],sigHists[iSig]))

            legend.AddEntry(sigHists[iSig], labelstring, "L")

        # add background entries
        for iBkg in range(len(bkgHists)):
            legend.AddEntry(bkgHists[iBkg], bkgLabels[iBkg], "F")

        # draw loegend
        legend.Draw("same")

        # add lumi and category to plot
        setup.printLumi(canvas, lumi = self.options["lumiScale"], ratio = self.options["ratio"])
        setup.printCategoryLabel(canvas, JTcut.getJTlabel(cat), ratio = self.options["ratio"])

        # save canvas
        setup.saveCanvas(canvas, plot_name)




                        


class variablePlotter2D:
    def __init__(self, output_dir, variable_set, add_vars, plotOptions = {}):
        self.output_dir     = output_dir
        self.variable_set   = variable_set
        self.add_vars       = list(add_vars)        

        self.samples        = {}
        self.ordered_stack  = []
        self.categories     = []

        # handle options
        defaultOptions = {
            "logscale":     False,
            "lumiScale":    1,
            }

        for key in plotOptions:
            defaultOptions[key] = plotOptions[key]
        self.options = defaultOptions
        

    def addSample(self, **kwargs):
        print("adding sample: "+str(kwargs["sampleName"]))
        self.samples[kwargs["sampleName"]] = Sample(**kwargs)
        if not self.samples[kwargs["sampleName"]].isSignal:
            self.ordered_stack.append(kwargs["sampleName"])

    def addCategory(self, category):
        print("adding category: {}".format(category))
        self.categories.append(category)

    def getAllVariables(self, cat):
        single_variables = self.variable_set.variables[cat]
        variable_pairs = self.add_vars
        for i, v1 in enumerate(single_variables):
            for j, v2 in enumerate(single_variables):
                if j<=i: continue
                variable_pairs.append([v1,v2])
        return variable_pairs

    def plot(self):
        # loop over categories and get list of variables
        for cat in self.categories:
            print("starting with category {}".format(cat))

            cat_dir = self.output_dir+"/"+cat+"/"
            if not os.path.exists(cat_dir):
                os.makedirs(cat_dir)
            # if no variable_set is given, plot all variables in samples
            if self.variable_set == None:
                variable_pairs = self.add_vars
            else:
                # load list of variables from variable set
                variable_pairs = self.getAllVariables(cat)

            variables = list(set([v[0] for v in variable_pairs]+[v[1] for v in variable_pairs]))
            # filter events according to JT category
            for key in self.samples:
                self.samples[key].cutData(cat, variables, self.options["lumiScale"])

            # loop over all variables and perform plot each time
            for variables in variable_pairs:
                print("plotting variables: {} vs {}".format(variables[0],variables[1]))

                for key in self.samples:
                    # generate plot output name
                    plot_name = cat_dir + "/{}_vs_{}_{}.pdf".format(
                        variables[0],
                        variables[1],
                        key.replace("(","").replace(")",""),
                        )
                    plot_name = plot_name.replace("[","_").replace("]","")
                        
                    # generate plot
                    self.histVariables2D(
                        vX          = variables[0],
                        vY          = variables[1],
                        plot_name   = plot_name,
                        sample      = key,
                        cat         = cat)

    def histVariables2D(self, vX, vY, plot_name, sample, cat):

        # get number of bins and binrange from config file
        binsX = binning.getNbins(vX)
        binsY = binning.getNbins(vY)
        rangeX = binning.getBinrange(vX)
        rangeY = binning.getBinrange(vY)

        # check if bin_range was found
        if not rangeX:
            maxValue = max(self.samples[sample].cut_data[cat][vX].values)
            minValue = min(self.samples[sample].cut_data[cat][vX].values)
            config_string = "variables[\""+vX+"\"]\t\t\t= Variable(bin_range = [{},{}])\n".format(minValue, maxValue)
            with open("new_variable_configs.txt", "a") as f:
                f.write(config_string)
            rangeX = [minValue, maxValue]

        if not rangeY:
            maxValue = max(self.samples[sample].cut_data[cat][vY].values)
            minValue = min(self.samples[sample].cut_data[cat][vY].values)
            config_string = "variables[\""+vY+"\"]\t\t\t= Variable(bin_range = [{},{}])\n".format(minValue, maxValue)
            with open("new_variable_configs.txt", "a") as f:
                f.write(config_string)
            rangeY = [minValue, maxValue]


        # fill hist
        weights = self.samples[sample].cut_data[cat]["weight"].values
        valuesX = self.samples[sample].cut_data[cat][vX].values
        valuesY = self.samples[sample].cut_data[cat][vY].values

        hist = setup.setupHistogram2D(
            valuesX     = valuesX,
            valuesY     = valuesY,
            weights     = weights,
            binsX       = binsX,
            binsY       = binsY,
            rangeX      = rangeX,
            rangeY      = rangeY,
            titleX      = vX,
            titleY      = vY)

        canvas = setup.drawHistOnCanvas2D(
            hist        = hist,
            canvasName  = vX+"_vs_"+vY,
            catLabel    = JTcut.getJTlabel(cat),
            sampleName  = sample)

        # add lumi and category to plot
        setup.printLumi(canvas, lumi = self.options["lumiScale"], twoDim = True)

        # save canvas
        setup.saveCanvas(canvas, plot_name)




                        

