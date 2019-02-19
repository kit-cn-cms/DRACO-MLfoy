import os
import sys
import pandas
import numpy as np
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(os.path.dirname(filedir))
sys.path.append(basedir)

import utils.generateJTcut as JTcut
import pyrootsOfTheCaribbean.plot_configs.variableConfig as binning
import pyrootsOfTheCaribbean.plot_configs.setupPlots as setup

class Sample:
    def __init__(self, sampleName, sampleFile, apply_cut = True):
        self.sampleName = sampleName
        self.sampleFile = sampleFile
        self.applyCut   = apply_cut

        self.load()
        self.cut_data   = {}

    def load(self):
        with pandas.HDFStore(self.sampleFile, mode = "r") as store:
            self.data = store.select("data", stop = 1000000)
        print("\tnevents: {}".format(self.data.shape[0]))

    def cutData(self, cut, variables):
        if not self.applyCut or cut in ["inclusive", "SL"]:
            self.cut_data[cut] = self.data
            self.cut_data[cut] = self.cut_data[cut].assign(weight = lambda x: x.Weight_XS*x.Weight_GEN_nom)
            return

        # cut events according to JT category
        category_cut = JTcut.getJTstring(cut)

        # only save variables that are needed
        self.cut_data[cut] = self.data.query(category_cut)[list(set(variables+["Weight_XS", "Weight_GEN_nom"]))]

        # add weight entry for scaling
        self.cut_data[cut] = self.cut_data[cut].assign(weight = lambda x: x.Weight_XS*x.Weight_GEN_nom)
            
        

class variableAnalyzer:
    def __init__(self, output_dir, variable_set, add_vars, numberOfBins = 100):
        self.output_dir         = output_dir
        self.variable_set       = variable_set
        self.add_vars           = list(add_vars)
        self.numberOfBins       = numberOfBins

        self.samples            = {}
        self.sampleNames        = []
        self.categories         = []

    def addSample(self, **kwargs):
        print("adding sample: "+str(kwargs["sampleName"]))
        self.samples[kwargs["sampleName"]] = Sample(**kwargs)
        self.sampleNames.append(kwargs["sampleName"])

    def addCategory(self, category):
        print("adding category: {}".format(category))
        self.categories.append(category)


    def perform1Danalysis(self, metric = "KS"):
        # loop over categories and get list of variables
        for cat in self.categories:
            print("starting with category {}".format(cat))

            cat_dir = self.output_dir+"/"+cat+"/"
            if not os.path.exists(cat_dir):
                os.makedirs(cat_dir)
            output_csv = self.output_dir+"/"+cat+"_1Ddistances_"+metric+".csv"
            good_variables_file = self.output_dir+"/"+cat+"_good_vars_1D.txt"    
    
            # load list of variables from variable set
            if cat in self.variable_set.variables:
                variables = self.variable_set.variables[cat] + self.add_vars
            else:
                variables = self.variable_set.all_variables + self.add_vars

            # filter events according to JT category
            
            for key in self.sampleNames:
                self.samples[key].cutData(cat, variables)

            # loop over all variables and perform plot each time
            variable_info = {}
            good_variables = []
            for variable in variables:
                print("analyzing variable: {}".format(variable))

                # generate plot output name
                plot_name = cat_dir + "/{}.pdf".format(variable)
                plot_name = plot_name.replace("[","_").replace("]","")
                    
                distanceDictionary = self.calculateAllDistances(
                    variable    = variable,
                    cat         = cat,
                    metric      = metric)
                
                variable_info[variable] = distanceDictionary
                max_pvalue = distanceDictionary[max(distanceDictionary, key = lambda k: distanceDictionary[k])]
                if max_pvalue < 0.05:
                    good_variables.append(variable)
        
                distanceMatrix = self.generateMatrix(distanceDictionary)
                m = setup.setup2DHistogram(
                    matrix      = distanceMatrix,
                    ncls        = len(self.sampleNames),
                    xtitle      = setup.generateLatexLabel(variable),
                    ytitle      = "",
                    binlabel    = self.sampleNames)
    
                canvas = setup.draw2DHistOnCanvas(m,"KSpvalues"+cat+variable,JTcut.getJTlabel(cat))
                setup.saveCanvas(canvas, plot_name)

            # generate dataframe info
            df = pandas.DataFrame(variable_info)
            df.to_csv(output_csv)
            with open(good_variables_file, "w") as f:
                f.write("variables[\"{}\"] = [\n".format(cat))
                for v in good_variables: f.write("    \"{}\",\n".format(v))
                f.write("    ]\n\n")
            print("saving distances in csv file {}".format(output_csv))
                
                
                
                
    def generateMatrix(self, distances):
        samples = self.sampleNames
        n_samples = len(samples)

        matrix = np.empty( (n_samples, n_samples), dtype = np.float64)

        for i, x in enumerate(samples):
            for j, y in enumerate(samples):
                if i == j:
                    matrix[i,j] = 1.
                else:
                    matrix[i,j] = distances["{}_vs_{}".format(x,y)]
        return matrix                    
                    
    
    def calculateAllDistances(self, variable, cat, metric = "KS"):
        bins = self.numberOfBins
        
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

        histograms = {}
        for sampleName in self.samples:
            sample = self.samples[sampleName]

            weights = sample.cut_data[cat]["weight"].values
            values  = sample.cut_data[cat][variable].values

            integral = sum(weights)

            hist = setup.setupHistogram(
                values          = values,
                weights         = weights,
                nbins           = bins,
                bin_range       = bin_range,
                xtitle          = cat+"_"+sample.sampleName+"_"+variable,
                ytitle          = "")

            hist.Scale(1./(sum(weights)+1e-9))

            histograms[sample.sampleName] = hist

        # now loop over all histograms to calculate the proximity value for each combination
        distances = {}
        for ix, xhist in enumerate(histograms):
            for iy, yhist in enumerate(histograms):
                if ix == iy: continue
                distance = self.calculateDistance(histograms[xhist], histograms[yhist], metric = metric)
                distances["{}_vs_{}".format(xhist,yhist)] = distance
        
        return distances


    def calculateDistance(self, hx, hy, metric = "KS"):
        if metric == "KS":
            KSscore = hx.KolmogorovTest(hy)
            return KSscore
        if metric == "Chi2":
            chi2 = hx.Chi2Test(hy, "WW")

