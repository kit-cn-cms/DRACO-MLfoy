import os
import sys
import numpy as np
import ROOT
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# local imports
filedir  = os.path.dirname(os.path.realpath(__file__))
pyrootdir = os.path.dirname(filedir)
basedir  = os.path.dirname(pyrootdir)
sys.path.append(pyrootdir)
sys.path.append(basedir)

import plot_configs.setupPlots as setup


class plotDiscriminators:
    def __init__(self, data, prediction_vector, event_classes, nbins, bin_range, signal_class, event_category, plotdir, logscale = False):
        self.data              = data
        self.prediction_vector = prediction_vector
        self.predicted_classes = np.argmax( self.prediction_vector, axis = 1)

        self.event_classes     = event_classes
        self.nbins             = nbins
        self.bin_range         = bin_range
        self.signal_class      = signal_class
        self.event_category    = event_category
        self.plotdir           = plotdir
        self.logscale          = logscale

        self.signalIndex       = self.data.class_translation[self.signal_class]
        self.signalFlag        = self.data.get_class_flag(self.signal_class)

        # default settings
        self.printROCScore = False

    def set_printROCScore(self, printROCScore):
        self.printROCScore = printROCScore

    def plot(self, ratio = False):
        # generate one plot per output node
        for i, node_cls in enumerate(self.event_classes):
            nodeIndex = self.data.class_translation[node_cls]

            # get output values of this node
            out_values = self.prediction_vector[:,i]

            if self.printROCScore:
                # calculate ROC value for specific node
                nodeROC = roc_auc_score(self.signalFlag, out_values)

            # fill lists according to class
            bkgHists  = []
            bkgLabels = []
            weightIntegral = 0

            # loop over all classes to fill hists according to truth level class
            for j, truth_cls in enumerate(self.event_classes):
                classIndex = self.data.class_translation[truth_cls]

                # filter values per event class
                filtered_values = [ out_values[k] for k in range(len(out_values)) \
                    if self.data.get_test_labels(as_categorical = False)[k] == classIndex \
                    and self.predicted_classes[k] == nodeIndex]

                filtered_weights = [ self.data.get_lumi_weights()[k] for k in range(len(out_values)) \
                    if self.data.get_test_labels(as_categorical = False)[k] == classIndex \
                    and self.predicted_classes[k] == nodeIndex]

                if j == self.signalIndex:
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
                        ytitle    = setup.GetyTitle(),
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
                ytitle    = setup.GetyTitle(),
                filled    = False)
            # set signal histogram linewidth
            sigHist.SetLineWidth(3)

            # set scalefactor
            scaleFactor = weightIntegral/(sum(sig_weights)+1e-9)
            sigHist.Scale(scaleFactor)

            plotOptions = {
                "ratio":      ratio,
                "ratioTitle": "#frac{scaled Signal}{Background}",
                "logscale":   self.logscale}
            canvas = setup.drawHistsOnCanvas(
                sigHist, bkgHists, plotOptions, 
                canvasName = node_cls+" final discriminator")

            # setup legend
            legend = setup.getLegend()

            # add signal entry
            legend.AddEntry(sigHist, sig_label+" x {:4.0f}".format(scaleFactor), "L")

            # add background entries
            for i, h in enumerate(bkgHists):
                legend.AddEntry(h, bkgLabels[i], "F")

            # draw legend
            legend.Draw("same")

            # add ROC score if activated
            if self.printROCScore:
                setup.printROCScore(canvas, nodeROC, plotOptions["ratio"])

            # add lumi and category to plot
            setup.printLumi(canvas, ratio = plotOptions["ratio"])
            setup.printCategoryLabel(canvas, self.event_category, ratio = plotOptions["ratio"])

            out_path = self.plotdir + "/finaldiscr_{}.pdf".format(node_cls)
            setup.saveCanvas(canvas, out_path)

        # add the histograms together
        workdir = os.path.dirname(self.plotdir[:-1])
        cmd = "pdfunite "+str(self.plotdir)+"/finaldiscr_*.pdf "+str(workdir)+"/discriminators.pdf"
        print(cmd)
        os.system(cmd)


class plotDiscriminatorsComparison_ttbb:
    def __init__(self, data, prediction_vector, event_classes, nbins, bin_range, signal_class, event_category, plotdir, logscale = False, ttbb_dataset=None, ttbb_prediction_vector=None):
        self.data              = data
        self.prediction_vector = prediction_vector
        self.predicted_classes = np.argmax( self.prediction_vector, axis = 1)

        if ttbb_dataset:
            self.ttbb_data = ttbb_dataset # Array of ttbb Datasets
        else:
            self.ttbb_data = None

        if ttbb_prediction_vector:
            self.ttbb_prediction_vector = ttbb_prediction_vector # Array of ttbb vector


        self.event_classes     = event_classes # List of nodes for plots
        self.nbins             = nbins
        self.bin_range         = bin_range
        self.signal_class      = signal_class # ttbb in this case
        self.event_category    = event_category
        self.plotdir           = plotdir
        self.logscale          = logscale

        self.signalIndex       = self.data.class_translation[self.signal_class]
        self.signalFlag        = self.data.get_class_flag(self.signal_class)

        # default settings
        self.printROCScore = False



    def set_printROCScore(self, printROCScore):
        self.printROCScore = printROCScore

    def plot(self, ratio = False):

        # Loop over all nodes
        for i, node_cls in enumerate(self.event_classes):
            nodeIndex = self.data.class_translation[node_cls]

            out_values_base = self.prediction_vector[:,i]
            out_values_additional_ttbb = []
            predictions_ttbb = []

            for ttbb_vector in self.ttbb_prediction_vector:
                out_values_additional_ttbb.append(ttbb_vector[:,i])
                predictions_ttbb.append(np.argmax(ttbb_vector, axis = 1))
            data_list_conc = [self.data]+self.ttbb_data
            pred_conc = [self.predicted_classes]+predictions_ttbb
            bkgHists  = []
            bkgLabels = []
            weightIntegral = 0
            # loop over all classes to fill hists according to truth level class
            for j,values_vector in enumerate([out_values_base]+out_values_additional_ttbb):
                # filter values per event class
                Filter_Flags = data_list_conc[j].get_class_flag("ttbb")
                filtered_values = [ values_vector[k] for k in range(len(values_vector)) \
                    if pred_conc[j][k] == nodeIndex and 
                    Filter_Flags[k]==True]

                filtered_weights = [ data_list_conc[j].get_lumi_weights()[k] for k in range(len(values_vector)) \
                    if pred_conc[j][k] == nodeIndex and 
                    Filter_Flags[k]==True]

                if j == 0:
                    # signal histogram
                    sig_values  = filtered_values
                    sig_label   = str("ttbb_powheg")
                    sig_weights = filtered_weights
                    scaleFactor= sum(sig_weights)
                else:
                    # background histograms
                    weightIntegral += sum(filtered_weights)
                    
                    histogram = setup.setupHistogram(
                        values    = filtered_values,
                        weights   = filtered_weights,
                        nbins     = self.nbins,
                        bin_range = self.bin_range,
                        color     = j,
                        xtitle    = str("ttbb_whatever")+" at "+str(node_cls)+" node",
                        ytitle    = setup.GetyTitle(),
                        filled    = False)
                    histogram.Scale(scaleFactor/sum(filtered_weights))
                    TMP_LIST= ["Helac","Openloops","amcatnlo"]
                    bkgHists.append( histogram )
                    bkgLabels.append( TMP_LIST[j-1] )
            # setup signal histogram
            sigHist = setup.setupHistogram(
                values    = sig_values,
                weights   = sig_weights,
                nbins     = self.nbins,
                bin_range = self.bin_range,
                color     = ROOT.kBlue,
                xtitle    = str(sig_label)+" at "+str(node_cls)+" node",
                ytitle    = setup.GetyTitle(),
                filled    = False)
            # set signal histogram linewidth
            sigHist.SetLineWidth(2)

            # set scalefactor
            #scaleFactor = weightIntegral/(sum(sig_weights)+1e-9)
            #scaleFactor = 1
            #sigHist.Scale(scaleFactor)

            plotOptions = {
                "ratio":      ratio,
                "ratioTitle": "#frac{scaled Signal}{Background}",
                "logscale":   self.logscale}
            canvas = setup.drawHistsOnCanvas(
                sigHist, bkgHists, plotOptions, 
                canvasName = node_cls+" final discriminator")

            # setup legend
            legend = setup.getLegend()

            # add signal entry
            legend.AddEntry(sigHist, sig_label, "L")

            # add background entries
            for i, h in enumerate(bkgHists):
                legend.AddEntry(h, bkgLabels[i], "F")

            # draw legend
            legend.Draw("same")

            # add ROC score if activated
            print(self.printROCScore)
            self.printROCScore=False
            if self.printROCScore:
                setup.printROCScore(canvas, nodeROC, plotOptions["ratio"])

            # add lumi and category to plot
            setup.printLumi(canvas, ratio = plotOptions["ratio"])
            setup.printCategoryLabel(canvas, self.event_category, ratio = plotOptions["ratio"])

            out_path = self.plotdir + "/finaldiscr_{}.pdf".format(node_cls)
            setup.saveCanvas(canvas, out_path)

        # add the histograms together
        workdir = os.path.dirname(self.plotdir[:-1])
        cmd = "pdfunite "+str(self.plotdir)+"/finaldiscr_*.pdf "+str(workdir)+"/discriminators.pdf"
        print(cmd)
        os.system(cmd)





class plotOutputNodes:
    def __init__(self, data, prediction_vector, event_classes, nbins, bin_range, signal_class, event_category, plotdir, logscale = False):
        self.data              = data
        self.prediction_vector = prediction_vector
        self.event_classes     = event_classes
        self.nbins             = nbins
        self.bin_range         = bin_range
        self.signal_class      = signal_class
        self.event_category    = event_category
        self.plotdir           = plotdir
        self.logscale          = logscale

        self.signalIndex       = self.data.class_translation[self.signal_class]
        self.signalFlag        = self.data.get_class_flag(self.signal_class)

        # default settings
        self.printROCScore = False
        self.cutVariable = False
        self.eventInCut = np.array([True for _ in range(len(self.prediction_vector))])

    def set_cutVariable(self, cutClass, cutValue):
        cutVariableIndex = self.data.class_translation[cutClass]
        predictions_cutVariable = self.prediction_vector[:, cutVariableIndex]
        self.eventInCut = [predictions_cutVariable[i] <= cutValue for i in len(self.prediction_vector)]

        self.cutVariable = True

    def set_printROCScore(self, printROCScore):
        self.printROCScore = printROCScore

    def plot(self, ratio = False):
        # generate one plot per output node
        for i, node_cls in enumerate(self.event_classes):
            # get output values of this node
            out_values = self.prediction_vector[:,i]

            if self.printROCScore:
                # calculate ROC value for specific node
                nodeROC = roc_auc_score(self.signalFlag, out_values)

            # fill lists according to class
            bkgHists  = []
            bkgLabels = []
            weightIntegral = 0

            # loop over all classes to fill hists according to truth level class
            for j, truth_cls in enumerate(self.event_classes):
                classIndex = self.data.class_translation[truth_cls]

                # filter values per event class
                filtered_values = [ out_values[k] for k in range(len(out_values)) \
                    if self.data.get_test_labels(as_categorical = False)[k] == classIndex \
                    and self.eventInCut[k]]

                filtered_weights = [ self.data.get_lumi_weights()[k] for k in range(len(out_values)) \
                    if self.data.get_test_labels(as_categorical = False)[k] == classIndex \
                    and self.eventInCut[k]]

                if j == self.signalIndex:
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
                        ytitle    = setup.GetyTitle(),
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
                ytitle    = setup.GetyTitle(),
                filled    = False)
            # set signal histogram linewidth
            sigHist.SetLineWidth(3)

            # set scalefactor
            scaleFactor = weightIntegral/(sum(sig_weights)+1e-9)
            sigHist.Scale(scaleFactor)

            plotOptions = {
                "ratio":      ratio,
                "ratioTitle": "#frac{scaled Signal}{Background}",
                "logscale":   self.logscale}
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

            ## scale signal Histogram
            #sigHist.Scale( scaleFactor )

            # draw legend
            legend.Draw("same")

            # add ROC score if activated
            if self.printROCScore:
                setup.printROCScore(canvas, nodeROC, plotOptions["ratio"])

            # add lumi and category to plot
            setup.printLumi(canvas, ratio = plotOptions["ratio"])
            setup.printCategoryLabel(canvas, self.event_category, ratio = plotOptions["ratio"])

            out_path = self.plotdir + "/outputNode_{}.pdf".format(node_cls)
            setup.saveCanvas(canvas, out_path)

        # add the histograms together
        workdir = os.path.dirname(self.plotdir[:-1])
        cmd = "pdfunite "+str(self.plotdir)+"/outputNode_*.pdf "+str(workdir)+"/outputNodes.pdf"
        print(cmd)
        os.system(cmd)


class plotConfusionMatrix:
    def __init__(self, data, prediction_vector, event_classes, event_category, plotdir):
        self.data              = data
        self.prediction_vector = prediction_vector
        self.predicted_classes = np.argmax(self.prediction_vector, axis = 1)

        self.event_classes     = event_classes
        self.n_classes         = len(self.event_classes)

        self.event_category    = event_category
        self.plotdir           = plotdir

        self.confusion_matrix = confusion_matrix(
            self.data.get_test_labels(as_categorical = False), self.predicted_classes)

        # default settings
        self.printROCScore = False
        self.ROCScore = None
    
    def set_printROCScore(self, printROCScore):
        self.printROCScore = printROCScore
        self.ROCScore = roc_auc_score(
            self.data.get_test_labels(), self.prediction_vector)

    def plot(self, norm_matrix = True):
        
        # norm confusion matrix if activated
        if norm_matrix:
            new_matrix = np.empty( (self.n_classes, self.n_classes), dtype = np.float64)
            for yit in range(self.n_classes):
                evt_sum = float(sum(self.confusion_matrix[yit,:]))
                for xit in range(self.n_classes):
                    new_matrix[yit,xit] = self.confusion_matrix[yit,xit]/(evt_sum+1e-9)

            self.confusion_matrix = new_matrix
        

        # initialize Histogram
        cm = setup.setup2DHistogram(
            matrix      = self.confusion_matrix.T,
            ncls        = self.n_classes,
            xtitle      = "predicted class",
            ytitle      = "true class",
            binlabel    = self.event_classes)

        canvas = setup.draw2DHistOnCanvas(cm, "confusion matrix", self.event_category, self.ROCScore)
        setup.saveCanvas(canvas, self.plotdir+"/confusionMatrix.pdf")