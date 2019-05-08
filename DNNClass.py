import pandas as pd
import os
import sys
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)

# imports with keras
import keras
from keras.utils import to_categorical
import keras.models as models
from keras import backend as K

# Limit gpu usage
import tensorflow as tf

import pyrootsOfTheCaribbean.plot_configs.setupPlots as setup


def getJTstring(cat):
    # jtstring is '(ge)[NJets]j_(ge)[NTags]t'
    # output format is '(N_Jets (>=/==) [NJets] and N_BTagsM (>=/==) [NTags])'

    string_parts = cat.split("_")

    cutstring = "("
    for part in string_parts:
        if part.endswith("l"):
            cutstring += "N_LooseLeptons"
        elif part.endswith("j"):
            cutstring += "N_jets"
        elif part.endswith("t"):
            cutstring += "N_btags"
        else:
            print("invalid format of category substring '{}' - IGNORING".format(part))
            continue

        if part.startswith("ge"):
            cutstring += " >= "+part[2:-1]
        elif part.startswith("le"):
            cutstring += " <= "+part[2:-1]
        else:
            cutstring += " == "+part[:-1]
        if not part == string_parts[-1]:
            cutstring += " and "

    cutstring += ")"

    return cutstring

def getJTlabel(cat):
    # jtstring is '(ge)[NJets]j_(ge)[NTags]t'
    # output format is '1 lepton, (\geq) 6 jets, (\geq) 3 b-tags'

    # special labels:
    if cat == "inclusive":  return "inclusive"
    if cat == "DL":         return "dileptonic t#bar{t}"

    string_parts = cat.split("_")

    cutstring = ""
    for part in string_parts:
        partstring = ""
        if part.startswith("ge"):
            n = part[2:-1]
            partstring += "\geq "
        elif part.startswith("le"):
            n = part[2:-1]
            partstring += "\leq "
        else:
            n = part[:-1]
            partstring += ""
        partstring += n


        if part.endswith("l"):
            partstring += " lepton"
        elif part.endswith("j"):
            partstring += " jet"
        elif part.endswith("t"):
            partstring += " b-tag"
        else:
            print("invalid format of category substring '{}' - IGNORING".format(part))
            continue

        # plural
        if int(n)>1: partstring += "s"

        if not part == string_parts[-1]:
            partstring += ", "
        cutstring += partstring

    return cutstring



class Sample:
    def __init__(self, path, label, normalization_weight = 1.):
        self.path = path
        self.label = label
        self.normalization_weight = normalization_weight


    def load_dataframe(self, event_category, lumi):
        print("-"*50)
        print("loading sample file "+str(self.path))
        with pd.HDFStore( self.path, mode = "r" ) as store:
            df = store.select("data")
            print("number of events: "+str(df.shape[0]))

        # add event weight
        df = df.assign(total_weight = lambda x: x.weight)

        # add lumi weight
        df = df.assign(lumi_weight = lambda x: x.weight * lumi * self.normalization_weight)

        self.data = df
        print("-"*50)


class InputSamples:
    def __init__(self, input_path):
        self.input_path = input_path
        self.samples = []

    def addSample(self, sample_path, label, normalization_weight = 1.):
        if not os.path.isabs(sample_path):
            sample_path = self.input_path + "/" + sample_path
        self.samples.append( Sample(sample_path, label, normalization_weight) )





class DNN():
    def __init__(self,
            save_path,
            input_samples,
            event_category,
            variables
            ):

        # save some information
        # list of samples to load into dataframe
        self.input_samples = input_samples

        # output directory for results
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs( self.save_path )

        # name of event category (usually nJet/nTag category)
        self.JTstring       = event_category
        self.event_category = getJTstring(event_category)
        self.categoryLabel  = getJTlabel(event_category)

        # list of input variables
        self.variables = variables

        # load data set
        self.data = self._load_datasets()
        self.event_classes = self.data.output_classes

        # make plotdir
        self.plot_path = self.save_path+"/plots/"
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)


    def _load_datasets(self):
        ''' load data set '''
        return DataFrame(
            input_samples       = self.input_samples,
            event_category      = self.event_category,
            variables     = self.variables,)


    def load_trained_model(self, inputDirectory):
        ''' load an already trained model '''
        checkpoint_path = inputDirectory+"/checkpoints/trained_model.h5py"
        weight_path = inputDirectory+"/checkpoints/trained_model_weights.h5"

        # get the model
        self.model = keras.models.load_model(checkpoint_path)
        self. model.compile(
              loss        = "categorical_crossentropy",
              optimizer   = keras.optimizers.Adam(1e-4),
             )
        self.model.summary()

        # save predicitons
        self.model_prediction_vector = self.model.predict(
            self.data.get_full_df()[self.variables].values )
        print(self.data.get_full_df()[self.variables].values)
        print(self.model_prediction_vector)

        # save predicted classes with argmax
        self.predicted_classes = np.argmax( self.model_prediction_vector, axis = 1)
        print(self.predicted_classes)

    def predict_event_query(self ):
        events = self.data.get_full_df()[self.variables]
        print(str(events.shape[0]) + " events.")

        for index, row in events.iterrows():
            print("========== DNN output ==========")
            print("Event: "+str(index))
            print(row)
            print("-------------------->")
            print(row.values)
            print("-------------------->")
            output = self.model.predict( np.array([list(row.values)]))[0]
            for i, node in enumerate(self.event_classes):
                print(str(node)+" node: "+str(output[i]))
            print("-------------------->")


    def plot_outputNodes(self, log = False, privateWork = False,
                        nbins = 20, bin_range = [0.,1.]):

        ''' plot distribution in outputNodes '''
        plotNodes = plotOutputNodes(
            data                = self.data,
            prediction_vector   = self.model_prediction_vector,
            event_classes       = self.event_classes,
            nbins               = nbins,
            bin_range           = bin_range,
            event_category      = self.categoryLabel,
            plotdir             = self.plot_path,
            logscale            = log)

        plotNodes.plot(ratio = False,  privateWork = privateWork)




class DataFrame(object):
    ''' takes a path to a folder where one h5 per class is located
        the events are cut according to the event_category
        variables in variables are used as input variables
        for better training, the variables can be normed to std(1) and mu(0) '''

    def __init__(self,
                input_samples,
                event_category,
                variables,
                lumi = 41.5):

        self.event_category = event_category
        self.lumi = lumi
        self.variables = variables


        # loop over all input samples and load dataframe
        train_samples = []
        for sample in input_samples.samples:
            sample.load_dataframe(self.event_category, self.lumi)
            train_samples.append(sample.data)

        # concatenating all dataframes
        df = pd.concat( train_samples )
        del train_samples

        # add class_label translation
        index = 0
        self.class_translation = {}
        self.classes = []
        for sample in input_samples.samples:
            self.class_translation[sample.label] = index
            self.classes.append(sample.label)
            index += 1
        self.index_classes = [self.class_translation[c] for c in self.classes]

        self.output_classes = self.classes
        print(self.output_classes)


        # add index labelling to dataframe
        df["index_label"] = pd.Series( [self.class_translation[c] for c in df["class_label"].values], index = df.index )

        unnormed_df = df.copy()

        norm_csv = pd.DataFrame(index=variables, columns=["mu", "std"])
        for v in variables:
            norm_csv["mu"][v] = unnormed_df[v].mean()
            norm_csv["std"][v] = unnormed_df[v].std()
        df[variables] = (df[variables] - df[variables].mean())/df[variables].std()
        self.norm_csv = norm_csv

        self.unsplit_df = df.copy()


        self.unsplit_df = df.copy()


    def get_labels(self, as_categorical = True):
        if as_categorical: return to_categorical( self.unsplit_df["index_label"].values )
        else:              return self.unsplit_df["index_label"].values


    # full sample ----------------------------------
    def get_full_df(self):
        return self.unsplit_df

    def get_class_flag(self, class_label):
        return pd.Series( [1 if c==class_label else 0 for c in self.unsplit_df["class_label"].values], index = self.unsplit_df.index ).values


    def get_lumi_weights(self):
        return self.unsplit_df["lumi_weight"].values


class plotOutputNodes:
    def __init__(self, data, prediction_vector, event_classes, nbins, bin_range,event_category, plotdir, logscale = False):
        self.data              = data
        self.prediction_vector = prediction_vector
        self.event_classes     = event_classes
        self.nbins             = nbins
        self.bin_range         = bin_range
        self.event_category    = event_category
        self.plotdir           = plotdir
        self.logscale          = logscale
        self.signalIndex       = []
        self.signalFlag        = []

        self.privateWork = False

    def plot(self, ratio = False, privateWork = False):
        self.privateWork = privateWork

        # generate one plot per output node
        for i, node_cls in enumerate(self.event_classes):
            # get output values of this node
            out_values = self.prediction_vector[:,i]

            nodeIndex = self.data.class_translation[node_cls]

            signalIndex = [nodeIndex]
            signalFlag  = [self.data.get_class_flag(node_cls)]
            print(i)
            print(nodeIndex)
            print(signalIndex)
            print(signalFlag)

            # fill lists according to class
            bkgHists  = []
            bkgLabels = []
            weightIntegral = 0

            # loop over all classes to fill hists according to truth level class
            for j, truth_cls in enumerate(self.event_classes):
                classIndex = self.data.class_translation[truth_cls]
                print(classIndex)

                # filter values per event class
                filtered_values = [ out_values[k] for k in range(len(out_values)) \
                    if self.data.get_labels(as_categorical = False)[k] == classIndex]

                filtered_weights = [ self.data.get_lumi_weights()[k] for k in range(len(out_values)) \
                    if self.data.get_labels(as_categorical = False)[k] == classIndex]

                if j in signalIndex:
                    # signal histogram
                    sig_values  = filtered_values
                    sig_label   = str(truth_cls)
                    sig_weights = filtered_weights
                else:
                    # background histograms
                    weightIntegral += sum(filtered_weights)

                    histogram = setup.setupHistogram(
                        values    = filtered_values,
                        weights   = filtered_weights,
                        nbins     = self.nbins,
                        bin_range = self.bin_range,
                        color     = setup.GetPlotColor(truth_cls),
                        xtitle    = str(truth_cls)+" at "+str(node_cls)+" node",
                        ytitle    = setup.GetyTitle(self.privateWork),
                        filled    = True)

                    bkgHists.append( histogram )
                    bkgLabels.append( truth_cls )

            # setup signal histogram
            sigHist = setup.setupHistogram(
                values    = sig_values,
                weights   = sig_weights,
                nbins     = self.nbins,
                bin_range = self.bin_range,
                color     = setup.GetPlotColor(sig_label),
                xtitle    = str(sig_label)+" at "+str(node_cls)+" node",
                ytitle    = setup.GetyTitle(self.privateWork),
                filled    = False)

            # set signal histogram linewidth
            sigHist.SetLineWidth(3)

            # set scalefactor
            scaleFactor = weightIntegral/(sum(sig_weights)+1e-9)
            sigHist.Scale(scaleFactor)

            # rescale histograms if privateWork enabled
            if privateWork:
                sigHist.Scale(1./sigHist.Integral())
                for bHist in bkgHists:
                    bHist.Scale(1./weightIntegral)

            plotOptions = {
                "ratio":      ratio,
                "ratioTitle": "#frac{scaled Signal}{Background}",
                "logscale":   self.logscale}

            # initialize canvas
            canvas = setup.drawHistsOnCanvas(
                sigHist, bkgHists, plotOptions,
                canvasName = node_cls+" node")

            # setup legend
            legend = setup.getLegend()

            # add signal entry
            legend.AddEntry(sigHist, sig_label+" x {:4.0f}".format(scaleFactor), "L")

            # add background entries
            for i, h in enumerate(bkgHists):
                legend.AddEntry(h, bkgLabels[i], "F")

            # draw legend
            legend.Draw("same")

            # add lumi or private work label to plot
            if self.privateWork:
                setup.printPrivateWork(canvas, plotOptions["ratio"], nodePlot = True)
            else:
                setup.printLumi(canvas, ratio = plotOptions["ratio"])

            # add category label
            setup.printCategoryLabel(canvas, self.event_category, ratio = plotOptions["ratio"])

            out_path = self.plotdir + "/outputNode_{}.pdf".format(node_cls)
            setup.saveCanvas(canvas, out_path)

        # add the histograms together
        workdir = os.path.dirname(self.plotdir[:-1])
        cmd = "pdfunite "+str(self.plotdir)+"/outputNode_*.pdf "+str(workdir)+"/outputNodes.pdf"
        print(cmd)
        os.system(cmd)
