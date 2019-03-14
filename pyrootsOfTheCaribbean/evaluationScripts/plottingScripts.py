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
                
                figure_of_merit = get_FOM(sum(sig_weights[iSig]), weightIntegral)
                print("figure of merit for {}: {}".format(sig_labels[iSig], figure_of_merit))

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


def get_FOM(S, B):
    return np.sqrt( 2.*( (S+B)*np.log(1.+1.*S/B)-S))

def get_FOM_with_uncert(name, S, B, sB = 0.2):
    sB*=B
    term1 = (S+B)*np.log(
        ( (S+B)*(B+sB**2) )/( B**2+(S+B)*sB**2 ) 
        )
    term2 = (B**2/sB**2)*np.log(
        1+( sB**2*S )/( B*(B+sB**2) )
        )
    return np.sqrt(2.*(term1-term2))








class plotOutputNodes:
    def __init__(self, data, prediction_vector, event_classes, nbins, bin_range, signal_class, event_category, plotdir, logscale = False, plot_nonTrainData = False):
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



class plotReconstruction:
    def __init__(self, sample, variables, nbins, bin_range, event_category, plotdir, logscale):
        self.sample             = sample
        self.variables          = variables
        self.nbins              = nbins
        self.bin_range          = bin_range
        self.event_category     = event_category
        self.plotdir            = plotdir
        self.logscale           = logscale

    def plot(self, ratio = False):
        for i, v in enumerate(self.variables):
            weights = self.sample.lumi_weights
            
            # define original distribution as background hist
            bkgHist = setup.setupHistogram(
                values          = self.sample.input_values[:,i],
                weights         = weights,
                nbins           = self.nbins,
                bin_range       = self.bin_range,
                color           = ROOT.kRed,
                xtitle          = str(v)+" input",
                ytitle          = setup.GetyTitle(),
                filled          = True)

            sigHist = setup.setupHistogram(
                values          = self.sample.prediction_vector[:,i],
                weights         = weights,
                nbins           = self.nbins,
                bin_range       = self.bin_range,
                color           = ROOT.kBlack,
                xtitle          = str(v)+" reco",
                ytitle          = setup.GetyTitle(),
                filled          = False)

            sigHist.SetLineWidth(3)
            plotOptions = {
                "ratio":      ratio,
                "ratioTitle": "#frac{reco}{original}",
                "logscale":   self.logscale}

            canvas = setup.drawHistsOnCanvas(
                sigHist, bkgHist, plotOptions,
                canvasName = str(v))

            # setup legend
            legend = setup.getLegend()

            legend.AddEntry(bkgHist, self.sample.label+" orig.", "F")
            legend.AddEntry(sigHist, self.sample.label+" reco.", "L")

            # draw legend
            legend.Draw("same")

            # add lumi and category to plot
            setup.printLumi(canvas, ratio = plotOptions["ratio"])
            setup.printCategoryLabel(canvas, self.event_category, ratio = plotOptions["ratio"])

            out_path = self.plotdir+"/reconstruction_"+str(v)+".pdf"
            setup.saveCanvas(canvas, out_path)

        workdir = os.path.dirname(self.plotdir[:-1])
        cmd = "pdfunite "+str(self.plotdir)+"/reconstruction_*.pdf "+str(workdir)+"/variableReconstructions.pdf"
        print(cmd)
        os.system(cmd)



class plotLoss:
    def __init__(self, train_sample, other_samples, loss_function, variables, nbins, bin_range, event_category, plotdir, logscale = False):
        self.train_sample      = train_sample
        self.other_samples     = other_samples
        self.loss_function     = loss_function
        self.variables         = variables
        self.nbins             = nbins
        self.bin_range         = bin_range
        self.event_category    = event_category
        self.plotdir           = plotdir
        self.logscale          = logscale

    def plot_nodes(self, ratio = False):
        for i, v in enumerate(self.variables):

            bkgHists = []
            bkgLabels = []
            weightIntegral = 0

            for sample in self.other_samples:
                weights = sample.lumi_weights
                values = sample.lossMatrix[:,i]

                hist = setup.setupHistogram(
                    values = values,
                    weights = weights,
                    nbins = self.nbins,
                    bin_range = self.bin_range,
                    color = setup.GetPlotColor(sample.label),
                    xtitle = str(v)+" "+str(sample.label)+" loss",
                    ytitle = setup.GetyTitle(),
                    filled = True)

                bkgHists.append(hist)
                bkgLabels.append(sample.label)
                weightIntegral += sum(weights)

            # get sing histogram
            sig_weights = self.train_sample.lumi_weights

            sigHist = setup.setupHistogram(
                values = self.train_sample.lossMatrix[:,i],
                weights = sig_weights,
                nbins = self.nbins,
                bin_range = self.bin_range,
                color = setup.GetPlotColor(self.train_sample.label),
                xtitle = str(v)+" "+str(self.train_sample.label)+" loss",
                ytitle = setup.GetyTitle(),
                filled = False)
            sigHist.SetLineWidth(3)
            scaleFactor = weightIntegral/(sum(sig_weights)+1e-9)
            sigHist.Scale(scaleFactor)

            plotOptions = {
                "ratio":      ratio,
                "ratioTitle": "#frac{scaled Signal}{Background}",
                "logscale":   self.logscale}
            
            canvas = setup.drawHistsOnCanvas(
                sigHist, bkgHists, plotOptions,
                canvasName = self.loss_function+" at "+str(v))

            # setup legend
            legend = setup.getLegend()

            for i, h in enumerate(bkgHists):
                legend.AddEntry(h, bkgLabels[i], "F")
            legend.AddEntry(sigHist, self.train_sample.label+" x {:4.0f}".format(scaleFactor), "L")

            # draw legend
            legend.Draw("same")

            # add lumi and category to plot
            setup.printLumi(canvas, ratio = plotOptions["ratio"])
            setup.printCategoryLabel(canvas, self.event_category, ratio = plotOptions["ratio"])

            out_path = self.plotdir+"/lossValues_"+str(v)+".pdf"
            setup.saveCanvas(canvas, out_path)

        workdir = os.path.dirname(self.plotdir[:-1])
        cmd = "pdfunite "+str(self.plotdir)+"/lossValues_*.pdf "+str(workdir)+"/lossDiscriminators.pdf"
        print(cmd)
        os.system(cmd)

    def plot_mean(self, ratio = False):

        bkgHists = []
        bkgLabels = []
        weightIntegral = 0
        
        for sample in self.other_samples:
            weights = sample.lumi_weights
            values = sample.lossVector

            hist = setup.setupHistogram(
                values    = values,
                weights   = weights,
                nbins     = self.nbins,
                bin_range = self.bin_range,
                color     = setup.GetPlotColor(sample.label),
                xtitle    = str(sample.label)+" loss",
                ytitle    = setup.GetyTitle(),
                filled    = True)

            # save hist
            bkgHists.append(hist)
            bkgLabels.append(sample.label)
            weightIntegral += sum(weights)

        # get sig histogram (trained sample)
        sig_weights = self.train_sample.lumi_weights        

        sigHist = setup.setupHistogram(
            values    = self.train_sample.lossVector,
            weights   = sig_weights,
            nbins     = self.nbins,
            bin_range = self.bin_range,
            color     = setup.GetPlotColor(self.train_sample.label),
            xtitle    = str(self.train_sample.label)+" loss",
            ytitle    = setup.GetyTitle(),
            filled    = False)
        sigHist.SetLineWidth(3)
        scaleFactor = weightIntegral/(sum(sig_weights)+1e-9)
        sigHist.Scale(scaleFactor)

        plotOptions = {
            "ratio":      ratio,
            "ratioTitle": "#frac{scaled Signal}{Background}",
            "logscale":   self.logscale}

        canvas = setup.drawHistsOnCanvas(
            sigHist, bkgHists, plotOptions, 
            canvasName = self.loss_function)

        # setup legend
        legend = setup.getLegend()

        for i, h in enumerate(bkgHists):
            legend.AddEntry(h, bkgLabels[i], "F")
        legend.AddEntry(sigHist, self.train_sample.label+" x {:4.0f}".format(scaleFactor), "L")

        # draw legend
        legend.Draw("same")

        # add lumi and category to plot
        setup.printLumi(canvas, ratio = plotOptions["ratio"])
        setup.printCategoryLabel(canvas, self.event_category, ratio = plotOptions["ratio"])

        workdir = os.path.dirname(self.plotdir[:-1])
        out_path = workdir+"/lossValues.pdf"
        setup.saveCanvas(canvas, out_path)









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
        if printROCScore:
            self.ROCScore = roc_auc_score(
                self.data.get_test_labels(), self.prediction_vector)

    def plot(self, norm_matrix = True, privateWork = False):
        
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

        canvas = setup.draw2DHistOnCanvas(cm, "confusion matrix", self.event_category, self.ROCScore, privateWork = privateWork)
        setup.saveCanvas(canvas, self.plotdir+"/confusionMatrix.pdf")
        









