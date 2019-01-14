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
    def __init__(self, data, prediction_vector, event_classes, nbins, bin_range, signal_class, event_category, plotdir, logscale = False, plot_nonTrainData = False):
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
        self.plot_nonTrainData = plot_nonTrainData

        self.signalIndex       = self.data.class_translation[self.signal_class]
        self.signalFlag        = self.data.get_class_flag(self.signal_class)

        # default settings
        self.printROCScore = False

    def set_printROCScore(self, printROCScore):
        self.printROCScore = printROCScore

    def plot(self, ratio = False):
        # generate one plot per output node
        for i, node_cls in enumerate(self.event_classes):
            print("\nPLOTTING OUTPUT NODE '"+str(node_cls))+"'"
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

            sig_values = []
            sig_labels = []
            sig_weights = []

            # if non-train data plotting is enabled, add the histograms here
            if self.plot_nonTrainData:
                for sample in self.data.non_train_samples:
                    values = sample.prediction_vector[:,i]
                    filtered_values = [ values[k] for k in range(len(values)) \
                        if sample.predicted_classes[k] == nodeIndex]
                    filtered_weights = [ sample.lumi_weights[k] for k in range(len(values)) \
                        if sample.predicted_classes[k] == nodeIndex]
                    print("{} events in discriminator: {}\t(Integral: {})".format(sample.label, len(filtered_values), sum(filtered_weights)))


                    if sample.signalSample:
                        sig_values.append(filtered_values)
                        sig_labels.append(sample.label)
                        sig_weights.append(filtered_weights)
                    else:
                        histogram = setup.setupHistogram(
                            values    = filtered_values,
                            weights   = filtered_weights,
                            nbins     = self.nbins,
                            bin_range = self.bin_range,
                            color     = setup.GetPlotColor(sample.label),
                            xtitle    = str(sample.label)+" at "+str(node_cls)+" node",
                            ytitle    = setup.GetyTitle(),
                            filled    = True)

                        bkgHists.append(histogram)
                        bkgLabels.append(sample.label)


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

                print("{} events in discriminator: {}\t(Integral: {})".format(truth_cls, len(filtered_values), sum(filtered_weights)))

                if j == self.signalIndex:
                    # signal histogram
                    sig_values.append(filtered_values)
                    sig_labels.append(str(truth_cls))
                    sig_weights.append(filtered_weights)
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
    
            sigHists = []
            scaleFactors = []
            for iSig in range(len(sig_labels)):
                # setup signal histogram
                sigHist = setup.setupHistogram(
                    values    = sig_values[iSig],
                    weights   = sig_weights[iSig],
                    nbins     = self.nbins,
                    bin_range = self.bin_range,
                    color     = setup.GetPlotColor(sig_labels[iSig]),
                    xtitle    = str(sig_labels[iSig])+" at "+str(node_cls)+" node",
                    ytitle    = setup.GetyTitle(),
                    filled    = False)

                # set signal histogram linewidth
                sigHist.SetLineWidth(3)

                # set scalefactor
                scaleFactor = weightIntegral/(sum(sig_weights[iSig])+1e-9)
                sigHist.Scale(scaleFactor)
                sigHists.append(sigHist)
                scaleFactors.append(scaleFactor)

            plotOptions = {
                "ratio":      ratio,
                "ratioTitle": "#frac{scaled Signal}{Background}",
                "logscale":   self.logscale}
            canvas = setup.drawHistsOnCanvas(
                sigHists, bkgHists, plotOptions, 
                canvasName = node_cls+" final discriminator")

            # setup legend
            legend = setup.getLegend()

            # add signal entry
            for i, h in enumerate(sigHists):
                legend.AddEntry(h, sig_labels[i]+" x {:4.0f}".format(scaleFactors[i]), "L")

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















class plotOutputNodes:
    def __init__(self, data, prediction_vector, event_classes, nbins, bin_range, signal_class, event_category, plotdir, logscale = False, plot_nonTrainData = False):
        df = pd.concat(samples)
        self.data              = data
        self.prediction_vector = prediction_vector
        self.event_classes     = event_classes
        self.nbins             = nbins
        self.bin_range         = bin_range
        self.signal_class      = signal_class
        self.event_category    = event_category
        self.plotdir           = plotdir
        self.logscale          = logscale
        self.plot_nonTrainData = plot_nonTrainData

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
        









