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

        if self.signal_class:
            self.signalIndex   = self.data.class_translation[self.signal_class]
            self.signalFlag    = self.data.get_class_flag(self.signal_class)

        # default settings
        self.printROCScore = False
        self.privateWork = False

    def plot(self, ratio = False, printROC = False, privateWork = False):
        self.printROCScore = printROC
        self.privateWork = privateWork

        # generate one plot per output node
        for i, node_cls in enumerate(self.event_classes):
            print("\nPLOTTING OUTPUT NODE '"+str(node_cls))+"'"

            # get index of node
            nodeIndex = self.data.class_translation[node_cls]
            if self.signal_class:
                signalIndex = self.signalIndex
                signalFlag  = self.signalFlag
            else:
                signalIndex = nodeIndex
                signalFlag  = self.data.get_class_flag(node_cls)

            # get output values of this node
            out_values = self.prediction_vector[:,i]

            if self.printROCScore:
                # calculate ROC value for specific node
                nodeROC = roc_auc_score(signalFlag, out_values)

            # fill lists according to class
            bkgHists  = []
            bkgLabels = []
            weightIntegral = 0

            sig_values = []
            sig_labels = []
            sig_weights = []

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

                if j == signalIndex:
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
                        ytitle    = setup.GetyTitle(self.privateWork),
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
                    ytitle    = setup.GetyTitle(self.privateWork),
                    filled    = False)

                # set signal histogram linewidth
                sigHist.SetLineWidth(3)

                # set scalefactor
                scaleFactor = weightIntegral/(sum(sig_weights[iSig])+1e-9)
                sigHist.Scale(scaleFactor)
                sigHists.append(sigHist)
                scaleFactors.append(scaleFactor)

            # rescale histograms if privateWork is enabled    
            if privateWork:
                for sHist in sigHists:
                    sHist.Scale(1./sHist.Integral())
                for bHist in bkgHists:
                    bHist.Scale(1./weightIntegral)

            plotOptions = {
                "ratio":      ratio,
                "ratioTitle": "#frac{scaled Signal}{Background}",
                "logscale":   self.logscale}

            # initialize canvas
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

            # add lumi or private work label to plot
            if self.privateWork:
                setup.printPrivateWork(canvas, plotOptions["ratio"], nodePlot = True)
            else:
                setup.printLumi(canvas, ratio = plotOptions["ratio"])

            # add category label
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

        if self.signal_class:
            self.signalIndex   = self.data.class_translation[self.signal_class]
            self.signalFlag    = self.data.get_class_flag(self.signal_class)

        # default settings
        self.printROCScore = False
        self.privateWork = False

    def plot(self, ratio = False, printROC = False, privateWork = False):
        self.printROCScore = printROC
        self.privateWork = privateWork

        # generate one plot per output node
        for i, node_cls in enumerate(self.event_classes):
            # get output values of this node
            out_values = self.prediction_vector[:,i]
            
            nodeIndex = self.data.class_translation[node_cls]
            if self.signal_class:
                signalIndex = self.signalIndex
                signalFlag  = self.signalFlag
            else:
                signalIndex = nodeIndex
                signalFlag  = self.data.get_class_flag(node_cls)

            if self.printROCScore:
                # calculate ROC value for specific node
                nodeROC = roc_auc_score(signalFlag, out_values)

            # fill lists according to class
            bkgHists  = []
            bkgLabels = []
            weightIntegral = 0

            # loop over all classes to fill hists according to truth level class
            for j, truth_cls in enumerate(self.event_classes):
                classIndex = self.data.class_translation[truth_cls]

                # filter values per event class
                filtered_values = [ out_values[k] for k in range(len(out_values)) \
                    if self.data.get_test_labels(as_categorical = False)[k] == classIndex]

                filtered_weights = [ self.data.get_lumi_weights()[k] for k in range(len(out_values)) \
                    if self.data.get_test_labels(as_categorical = False)[k] == classIndex]

                if j == signalIndex:
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

            # add ROC score if activated
            if self.printROCScore:
                setup.printROCScore(canvas, nodeROC, plotOptions["ratio"])

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



class plotClosureTest:
    def __init__(self, data, test_prediction, train_prediction, event_classes, nbins, bin_range, signal_class, event_category, plotdir, logscale = False):
        self.data               = data
        self.test_prediction    = test_prediction
        self.train_prediction   = train_prediction

        self.pred_classes_test  = np.argmax(self.test_prediction, axis = 1)
        self.pred_classes_train = np.argmax(self.train_prediction, axis = 1)

        self.event_classes      = event_classes
        self.nbins              = nbins
        self.bin_range          = bin_range
        self.signal_class       = signal_class
        self.event_category     = event_category
        self.plotdir            = plotdir
        self.logscale           = logscale

        if self.signal_class:
            self.signalIndex = self.data.class_translation[self.signal_class]
            self.signalFlag  = self.data.get_class_flag(self.signal_class)

        # generate sub directory
        self.plotdir += "/ClosurePlots/"
        if not os.path.exists(self.plotdir):
            os.makedirs(self.plotdir)

        # default settings
        self.privateWork = False

    def plot(self, ratio = False, privateWork = False):
        self.privateWork = privateWork

        # loop over output nodes
        for i, node_cls in enumerate(self.event_classes):
            # get index of node
            nodeIndex = self.data.class_translation[node_cls]
            if self.signal_class:
                signalIndex = self.signalIndex
                signalClass = self.signal_class
            else:
                signalIndex = nodeIndex
                signalClass = node_cls

            # get output values of this node
            test_values = self.test_prediction[:,i]
            train_values = self.train_prediction[:,i]

            sig_test_values = [test_values[k] for k in range(len(test_values)) \
                if self.data.get_test_labels(as_categorical = False)[k] == signalIndex \
                and self.pred_classes_test[k] == nodeIndex]
            bkg_test_values = [test_values[k] for k in range(len(test_values)) \
                if not self.data.get_test_labels(as_categorical = False)[k] == signalIndex \
                and self.pred_classes_test[k] == nodeIndex]

            sig_train_values = [train_values[k] for k in range(len(train_values)) \
                if self.data.get_train_labels(as_categorical = False)[k] == signalIndex \
                and self.pred_classes_train[k] == nodeIndex]
            bkg_train_values = [train_values[k] for k in range(len(train_values)) \
                if not self.data.get_train_labels(as_categorical = False)[k] == signalIndex \
                and self.pred_classes_train[k] == nodeIndex]

            sig_test_weights = [self.data.get_lumi_weights()[k] for k in range(len(test_values)) \
                if self.data.get_test_labels(as_categorical = False)[k] == signalIndex \
                and self.pred_classes_test[k] == nodeIndex]
            bkg_test_weights = [self.data.get_lumi_weights()[k] for k in range(len(test_values)) \
                if not self.data.get_test_labels(as_categorical = False)[k] == signalIndex \
                and self.pred_classes_test[k] == nodeIndex]
        
            sig_train_weights = [self.data.get_train_lumi_weights()[k] for k in range(len(test_values)) \
                if self.data.get_train_labels(as_categorical = False)[k] == signalIndex \
                and self.pred_classes_train[k] == nodeIndex]
            bkg_train_weights = [self.data.get_train_lumi_weights()[k] for k in range(len(test_values)) \
                if self.data.get_train_labels(as_categorical = False)[k] == signalIndex \
                and self.pred_classes_train[k] == nodeIndex]

            # setup train histograms
            sig_train = setup.setupHistogram(
                values      = sig_train_values,
                weights     = sig_train_weights,
                nbins       = self.nbins,
                bin_range   = self.bin_range,
                color       = ROOT.kCyan,
                xtitle      = "signal train at "+str(node_cls)+" node",
                ytitle      = setup.GetyTitle(privateWork = True),
                filled      = False)
            sig_train.Scale(1./sig_train.Integral())
            sig_train.SetLineWidth(2)

            bkg_train = setup.setupHistogram(
                values      = bkg_train_values,
                weights     = bkg_train_weights,
                nbins       = self.nbins,
                bin_range   = self.bin_range,
                color       = ROOT.kOrange-3,
                xtitle      = "bkg train at "+str(node_cls)+" node",
                ytitle      = setup.GetyTitle(privateWork = True),
                filled      = False)
            bkg_train.Scale(1./bkg_train.Integral())
            bkg_train.SetLineWidth(2)

            # setup test histograms
            sig_test = setup.setupHistogram(
                values      = sig_test_values,
                weights     = sig_test_weights,
                nbins       = self.nbins,
                bin_range   = self.bin_range,
                color       = ROOT.kCyan+2,
                xtitle      = "signal test at "+str(node_cls)+" node",
                ytitle      = setup.GetyTitle(privateWork = True),
                filled      = False)
            sig_test.Scale(1./sig_test.Integral())
            sig_test.SetLineWidth(2)

            bkg_test = setup.setupHistogram(
                values      = bkg_test_values,
                weights     = bkg_test_weights,
                nbins       = self.nbins,
                bin_range   = self.bin_range,
                color       = ROOT.kOrange+7,
                xtitle      = "bkg test at "+str(node_cls)+" node",
                ytitle      = setup.GetyTitle(privateWork = True),
                filled      = False)
            bkg_test.Scale(1./bkg_test.Integral())
            bkg_test.SetLineWidth(2)

            plotOptions = {"logscale": self.logscale}

            # init canvas
            canvas = setup.drawClosureTestOnCanvas(
                sig_train, bkg_train, sig_test, bkg_test, plotOptions,
                canvasName = "closure test at {} node".format(node_cls))
                    
            # setup legend
            legend = setup.getLegend()

            # add entries
            legend.AddEntry(sig_train, "train {}".format(signalClass), "L")
            legend.AddEntry(bkg_train, "train bkg", "L")
            legend.AddEntry(sig_test,  "test {}".format(signalClass), "L")
            legend.AddEntry(bkg_test,  "test bkg", "L")

            # draw legend
            legend.Draw("same")

            # prit private work label if activated
            if self.privateWork:
                setup.printPrivateWork(canvas)
            # add category label
            setup.printCategoryLabel(canvas, self.event_category)

            


            # add private work label if activated
            if self.privateWork:
                setup.printPrivateWork(canvas, plotOptions["ratio"], nodePlot = True)

            out_path = self.plotdir+"/closureTest_at_{}_node.pdf".format(node_cls)
            setup.saveCanvas(canvas, out_path)

        # add the histograms together
        workdir = os.path.dirname(os.path.dirname(self.plotdir[:-1]))
        cmd = "pdfunite "+str(self.plotdir)+"/closureTest_*.pdf "+str(workdir)+"/closureTest.pdf"
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
        self.ROCScore = None
    
    def plot(self, norm_matrix = True, privateWork = False, printROC = False):
        if printROC:
            self.ROCScore = roc_auc_score(
                self.data.get_test_labels(), self.prediction_vector)

        # norm confusion matrix if activated
        if norm_matrix:
            new_matrix = np.empty( (self.n_classes, self.n_classes), dtype = np.float64)
            for yit in range(self.n_classes):
                evt_sum = float(sum(self.confusion_matrix[yit,:]))
                for xit in range(self.n_classes):
                    new_matrix[yit,xit] = self.confusion_matrix[yit,xit]/(evt_sum+1e-9)

            self.confusion_matrix = new_matrix
        

        # initialize Histogram
        cm = setup.setupConfusionMatrix(
            matrix      = self.confusion_matrix.T,
            ncls        = self.n_classes,
            xtitle      = "predicted class",
            ytitle      = "true class",
            binlabel    = self.event_classes)

        canvas = setup.drawConfusionMatrixOnCanvas(cm, "confusion matrix", self.event_category, self.ROCScore, privateWork = privateWork)
        setup.saveCanvas(canvas, self.plotdir+"/confusionMatrix.pdf")
        









