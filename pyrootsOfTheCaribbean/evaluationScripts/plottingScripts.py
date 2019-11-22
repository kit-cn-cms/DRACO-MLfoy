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
    def __init__(self, data, prediction_vector, event_classes, nbins, bin_range, signal_class, event_category, plotdir, logscale = False, sigScale = -1):
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
        self.sigScale          = sigScale
        self.signalIndex       = []
        self.signalFlag        = []

        if self.signal_class:
            for signal in signal_class:
                self.signalIndex.append(self.data.class_translation[signal])
                self.signalFlag.append(self.data.get_class_flag(signal))

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
                signalIndex = [nodeIndex]
                signalFlag  = [self.data.get_class_flag(node_cls)]

            # get output values of this node
            out_values = self.prediction_vector[:,i]

            if self.printROCScore and len(signalIndex)==1:
                # calculate ROC value for specific node
                nodeROC = roc_auc_score(signalFlag[0], out_values)

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

                if j in signalIndex:
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
                if self.sigScale == -1:
                    scaleFactor = weightIntegral/(sum(sig_weights[iSig])+1e-9)
                else:
                    scaleFactor = float(self.sigScale)

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
            if self.printROCScore and len(signalIndex)==1:
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

class getDiscriminators:
    def __init__(self, data, prediction_vector, event_classes, nbins, bin_range, signal_class, event_category, plotdir, binning=None,binflag=None):
        self.data              = data
        self.prediction_vector = prediction_vector
        self.predicted_classes = np.argmax( self.prediction_vector, axis = 1)

        self.event_classes     = event_classes
        self.nbins             = nbins
        self.bin_range         = bin_range
        self.signal_class      = signal_class
        self.event_category    = event_category
        self.plotdir           = plotdir
        self.signalIndex       = []
        self.signalFlag        = []
        self.binning           = binning
        self.binflag           = binflag

        if self.signal_class:
            print self.signal_class
            print self.data.class_translation
            self.signalIndex.append(self.data.class_translation[signal_class])
            self.signalFlag.append(self.data.get_class_flag(signal_class))


    def getHistograms(self,tag=None):
        Histograms = {}

        # generate one plot per output node
        for i, node_cls in enumerate(self.event_classes):
            print("\nGETTING OUTPUT NODE '"+str(node_cls))+"'"

            # get index of node
            nodeIndex = self.data.class_translation[node_cls]
            if self.signal_class:
                signalIndex = self.signalIndex
                signalFlag  = self.signalFlag
            else:
                signalIndex = [nodeIndex]
                signalFlag  = [self.data.get_class_flag(node_cls)]

            # get output values of this node
            out_values = self.prediction_vector[:,i]

            if not self.binning == None:
                self.bin_range=self.binning[node_cls]
                print self.bin_range
                self.nbins = self.bin_range.size - 1
                print self.nbins


            # fill lists according to class
            bkgHists  = []
            bkgLabels = []
            weightIntegral = 0

            Histograms[node_cls]={}

            sig_values = []
            sig_labels = []
            sig_weights = []

            # loop over all classes to fill hists according to truth level class
            for j, truth_cls in enumerate(self.event_classes):
                classIndex = self.data.class_translation[truth_cls]

                # filter values per event class
                filtered_values = [ out_values[k] for k in range(len(out_values)) \
                    if self.data.get_full_df_labels(as_categorical = False)[k] == classIndex \
                    and self.predicted_classes[k] == nodeIndex]

                filtered_weights = [ self.data.get_full_df_lumi_weights()[k] for k in range(len(out_values)) \
                    if self.data.get_full_df_labels(as_categorical = False)[k] == classIndex \
                    and self.predicted_classes[k] == nodeIndex]

                print("{} events in discriminator: {}\t(Integral: {})".format(truth_cls, len(filtered_values), sum(filtered_weights)))

                
                if j in signalIndex:
                    # signal histogram
                    sig_values.append(filtered_values)
                    sig_labels.append(str(truth_cls))
                    sig_weights.append(filtered_weights)
                else:
                    # background histograms
                    weightIntegral += sum(filtered_weights)
                    if tag == None:
                        xtitle = str(truth_cls)+" at "+str(node_cls)+" node"
                    else:
                        xtitle = str(truth_cls)+" at "+str(node_cls)+" node in "+str(tag)
                    histogram = setup.setupHistogram(
                        values    = filtered_values,
                        weights   = filtered_weights,
                        nbins     = self.nbins,
                        bin_range = self.bin_range,
                        color     = setup.GetPlotColor(truth_cls),
                        xtitle    = xtitle,
                        ytitle    = setup.GetyTitle(),
                        filled    = False,
                        binflag   = self.binflag)

                    Histograms[node_cls][truth_cls]=histogram

            sigHists = []
            scaleFactors = []
            for iSig in range(len(sig_labels)):
                if tag == None:
                    xtitle = str(sig_labels[iSig])+" at "+str(node_cls)+" node"
                else:
                    xtitle = str(sig_labels[iSig])+" at "+str(node_cls)+" node in "+str(tag)
                # setup signal histogram
                sigHist = setup.setupHistogram(
                    values    = sig_values[iSig],
                    weights   = sig_weights[iSig],
                    nbins     = self.nbins,
                    bin_range = self.bin_range,
                    color     = setup.GetPlotColor(sig_labels[iSig]),
                    xtitle    = xtitle,
                    ytitle    = setup.GetyTitle(),
                    filled    = False,
                    binflag   = self.binflag)

                # set signal histogram linewidth
                sigHist.SetLineWidth(3)
                Histograms[node_cls][self.signal_class]=sigHist
           
        return Histograms

class meanAndMeanError:
    def __init__(self, histograms, outputdir, categorylabel, DNNname):
        self.histograms = histograms
        self.outputdir = outputdir
        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)
            print("Directory '%s' created" %self.outputdir) 
        self.DNNname = DNNname
        self.categorylabel = categorylabel

    def calculate(self):


        for dataera in self.histograms:
            meanHistograms = {}
            stdHistograms = {}
            for model in self.histograms[dataera]:
                for node in self.histograms[dataera][model]:
                    meanHistograms[node] = {}
                    stdHistograms[node] = {}
                    for process in self.histograms[dataera][model][node]:
                        meanHistograms[node][process] = {}
                        stdHistograms[node][process] = {}

            print meanHistograms
            print stdHistograms

            for model in self.histograms[dataera]:
                for node in self.histograms[dataera][model]:
                    for process in self.histograms[dataera][model][node]:
                        hist_mean = self.histograms[dataera][model][node][process]["dnn0"].Clone()
                        hist_std = self.histograms[dataera][model][node][process]["dnn0"].Clone()
                        for n in range(self.histograms[dataera][model][node][process]["dnn0"].GetNbinsX()):
                            print "Bin {}".format(n)
                            dnnbincontent = []
                            for dnnname in self.histograms[dataera][model][node][process]:
                                dnn=self.histograms[dataera][model][node][process][dnnname]
                                dnnbincontent.append(dnn.GetBinContent(n))
                            temp = np.array(dnnbincontent)
                            temp_mean = np.mean(temp)
                            temp_std = np.std(temp)

                            hist_mean.SetBinContent(n,temp_mean)
                            hist_mean.SetBinError(n,temp_std)
                            hist_std.SetBinContent(n,temp_std)

                        meanHistograms[node][process][model] = hist_mean
                        stdHistograms[node][process][model] = hist_std

            print meanHistograms
            print stdHistograms
            outputdir=self.outputdir+"/Dataera_"+str(dataera)+"/"

            PSmean = plotShapes(histograms=meanHistograms, 
                                outputdir=outputdir+"/mean/",
                                categorylabel=self.categorylabel,DNNname="mean")
            PSmean.plot(error=True)

            PSstd = plotShapes(histograms=stdHistograms, 
                                outputdir=outputdir+"/standardDeviation/",
                                categorylabel=self.categorylabel,DNNname="standard deviation")
            PSstd.plot()





class plotShapes:   
    def __init__(self, histograms, outputdir, categorylabel, DNNname):
        self.histograms = histograms
        self.outputdir = outputdir
        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)
            print("Directory '%s' created" %self.outputdir) 
        self.DNNname = DNNname
        self.categorylabel = categorylabel

    def plot(self, error=None):
        for node in self.histograms:
            for process in self.histograms[node]:
                print("\nPLOTTING OUTPUT NODE '"+str(node)+"' PROCESS '"+str(process)+"'")
                canvasname=node+"_"+process
                canvas = self.getCanvas(canvasname)
                # setup legend
                legend = self.getLegend()

                #move Over and Underflow
                for dnn in self.histograms[node][process]:
                    histogram=self.histograms[node][process][dnn]
                    self.moveOverUnderFlow(histogram)

                yMax = 1e-9
                yMinMax = 1000.
                for i, dnn in enumerate(self.histograms[node][process]):
                    h=self.histograms[node][process][dnn]
                    yMax = max(h.GetBinContent(h.GetMaximumBin()), yMax)
                    if h.GetBinContent(h.GetMaximumBin()) > 0:
                        yMinMax = min(h.GetBinContent(h.GetMaximumBin()), yMinMax)

                if error==None:
                    option = "histo"
                else: option="histo E0"
                histtitle= self.DNNname+": "+process+" at "+node+" node"
                for i, dnn in enumerate(self.histograms[node][process]):
                    histogram=self.histograms[node][process][dnn]
                    if i==0:
                        histogram.SetLineColor(i+2)
                        histogram.GetXaxis().SetTitle(histtitle)
                        histogram.GetYaxis().SetRangeUser(0, yMax*1.5)
                        histogram.DrawCopy(option+"")
                    else:
                        histogram.SetLineColor(i+2)
                        histogram.DrawCopy(option+"same")
                    legend.AddEntry(histogram, dnn, "F")
                

                # draw legend
                legend.Draw("same")

                # add lumi or private work label to plot
                # if self.privateWork:
                #     setup.printPrivateWork(canvas, plotOptions["ratio"], nodePlot = True)
                # else:
                #     setup.printLumi(canvas, ratio = plotOptions["ratio"])

                # add category label
                
                self.printCategoryLabel(canvas,self.categorylabel)

                nodepath= self.outputdir+"/{}/".format(node)
                if not os.path.exists(nodepath):
                    os.makedirs(nodepath)
                    print("Directory '%s' created" %nodepath)

                out_path = nodepath + "/outputNode_{}_{}.pdf".format(node, process)
                
                self.saveCanvas(canvas, out_path)

        # # add the histograms together
        # workdir = os.path.dirname(self.plotdir[:-1])
        # cmd = "pdfunite "+str(self.plotdir)+"/outputNode_*.pdf "+str(workdir)+"/outputNodes.pdf"
        # print(cmd)
        # os.system(cmd)

    def getCanvas(self, name):
        canvas = ROOT.TCanvas(name, name, 1024, 768)
        canvas.SetTopMargin(0.07)
        canvas.SetBottomMargin(0.15)
        canvas.SetRightMargin(0.05)
        canvas.SetLeftMargin(0.15)
        canvas.SetTicks(1,1)
        return canvas

    def getLegend(self):
        legend=ROOT.TLegend(0.70,0.6,0.95,0.9)
        legend.SetBorderSize(0);
        legend.SetLineStyle(0);
        legend.SetTextFont(42);
        legend.SetTextSize(0.05);
        legend.SetFillStyle(0);
        return legend

    def saveCanvas(self, canvas, path):
        canvas.SaveAs(path)
        canvas.SaveAs(path.replace(".pdf",".png"))
        canvas.Clear()

    def printCategoryLabel(self, pad, catLabel):
        pad.cd(1)
        l = pad.GetLeftMargin()
        t = pad.GetTopMargin()
        r = pad.GetRightMargin()
        b = pad.GetBottomMargin()

        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextColor(ROOT.kBlack)

        latex.DrawLatex(l+0.02,1.-t-0.06, catLabel)

    def moveOverUnderFlow(self, hist):
        # move underflow
        hist.SetBinContent(1, hist.GetBinContent(0)+hist.GetBinContent(1))
        # move overflow
        hist.SetBinContent(hist.GetNbinsX(), hist.GetBinContent(hist.GetNbinsX()+1)+hist.GetBinContent(hist.GetNbinsX()))

        # set underflow error
        hist.SetBinError(1, ROOT.TMath.Sqrt(
            ROOT.TMath.Power(hist.GetBinError(0),2) + ROOT.TMath.Power(hist.GetBinError(1),2) ))
        # set overflow error
        hist.SetBinError(hist.GetNbinsX(), ROOT.TMath.Sqrt(
            ROOT.TMath.Power(hist.GetBinError(hist.GetNbinsX()),2) + 
            ROOT.TMath.Power(hist.GetBinError(hist.GetNbinsX()+1),2) ))
        #delete overflow and underflow after moving
        hist.SetBinContent(0,0)
        hist.SetBinError(0,0)
        hist.SetBinContent(hist.GetNbinsX()+1,0)
        hist.SetBinError(hist.GetNbinsX()+1,0)

    

class plotOutputNodes:
    def __init__(self, data, prediction_vector, event_classes, nbins, bin_range, signal_class, event_category, plotdir, logscale = False, sigScale = -1):
        self.data              = data
        self.prediction_vector = prediction_vector
        self.event_classes     = event_classes
        self.nbins             = nbins
        self.bin_range         = bin_range
        self.signal_class      = signal_class
        self.event_category    = event_category
        self.plotdir           = plotdir
        self.logscale          = logscale
        self.sigScale          = sigScale
        self.signalIndex       = []
        self.signalFlag        = []

        if self.signal_class:
            for signal in signal_class:
                self.signalIndex.append(self.data.class_translation[signal])
                self.signalFlag.append(self.data.get_class_flag(signal))

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
                signalIndex = [nodeIndex]
                signalFlag  = [self.data.get_class_flag(node_cls)]

            if self.printROCScore and len(signalIndex)==1:
                # calculate ROC value for specific node
                nodeROC = roc_auc_score(signalFlag[0], out_values)

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
            if self.sigScale == -1:
                scaleFactor = weightIntegral/(sum(sig_weights)+1e-9)
            else:
                scaleFactor = float(self.sigScale)
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
            if self.printROCScore and len(signalIndex)==1:
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
        self.signalIndex       = []

        if self.signal_class:
            for signal in signal_class:
                self.signalIndex.append(self.data.class_translation[signal])


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
                signalIndex = [nodeIndex]
                signalClass = node_cls

            # get output values of this node
            test_values = self.test_prediction[:,i]
            train_values = self.train_prediction[:,i]

            sig_test_values = [test_values[k] for k in range(len(test_values)) \
                if self.data.get_test_labels(as_categorical = False)[k] in signalIndex \
                and self.pred_classes_test[k] == nodeIndex]
            bkg_test_values = [test_values[k] for k in range(len(test_values)) \
                if not self.data.get_test_labels(as_categorical = False)[k] in signalIndex \
                and self.pred_classes_test[k] == nodeIndex]

            sig_train_values = [train_values[k] for k in range(len(train_values)) \
                if self.data.get_train_labels(as_categorical = False)[k] in signalIndex \
                and self.pred_classes_train[k] == nodeIndex]
            bkg_train_values = [train_values[k] for k in range(len(train_values)) \
                if not self.data.get_train_labels(as_categorical = False)[k] in signalIndex \
                and self.pred_classes_train[k] == nodeIndex]

            sig_test_weights = [self.data.get_lumi_weights()[k] for k in range(len(test_values)) \
                if self.data.get_test_labels(as_categorical = False)[k] in signalIndex \
                and self.pred_classes_test[k] == nodeIndex]
            bkg_test_weights = [self.data.get_lumi_weights()[k] for k in range(len(test_values)) \
                if not self.data.get_test_labels(as_categorical = False)[k] in signalIndex \
                and self.pred_classes_test[k] == nodeIndex]

            sig_train_weights = [self.data.get_train_lumi_weights()[k] for k in range(len(train_values)) \

                if self.data.get_train_labels(as_categorical = False)[k] in signalIndex \
                and self.pred_classes_train[k] == nodeIndex]
            bkg_train_weights = [self.data.get_train_lumi_weights()[k] for k in range(len(train_values)) \
                if self.data.get_train_labels(as_categorical = False)[k] in signalIndex \
                and self.pred_classes_train[k] == nodeIndex]

            # setup train histograms
            sig_train = setup.setupHistogram(
                values      = sig_train_values,
                weights     = sig_train_weights,
                nbins       = self.nbins,
                bin_range   = self.bin_range,
                color       = ROOT.kBlue,
                xtitle      = "signal train at "+str(node_cls)+" node",
                ytitle      = setup.GetyTitle(privateWork = True),
                filled      = True)
            sig_train.Scale(1./sig_train.Integral())
            sig_train.SetLineWidth(1)
            sig_train.SetFillColorAlpha(ROOT.kBlue, 0.5)

            bkg_train = setup.setupHistogram(
                values      = bkg_train_values,
                weights     = bkg_train_weights,
                nbins       = self.nbins,
                bin_range   = self.bin_range,
                color       = ROOT.kRed,
                xtitle      = "bkg train at "+str(node_cls)+" node",
                ytitle      = setup.GetyTitle(privateWork = True),
                filled      = True)
            bkg_train.Scale(1./bkg_train.Integral())
            bkg_train.SetLineWidth(1)
            bkg_train.SetFillColorAlpha(ROOT.kRed, 0.5)

            # setup test histograms
            sig_test = setup.setupHistogram(
                values      = sig_test_values,
                weights     = sig_test_weights,
                nbins       = self.nbins,
                bin_range   = self.bin_range,
                color       = ROOT.kBlue,
                xtitle      = "signal test at "+str(node_cls)+" node",
                ytitle      = setup.GetyTitle(privateWork = True),
                filled      = False)
            sig_test.Scale(1./sig_test.Integral())
            sig_test.SetLineWidth(1)
            sig_test.SetMarkerStyle(20)
            sig_test.SetMarkerSize(2)

            bkg_test = setup.setupHistogram(
                values      = bkg_test_values,
                weights     = bkg_test_weights,
                nbins       = self.nbins,
                bin_range   = self.bin_range,
                color       = ROOT.kRed,
                xtitle      = "bkg test at "+str(node_cls)+" node",
                ytitle      = setup.GetyTitle(privateWork = True),
                filled      = False)
            bkg_test.Scale(1./bkg_test.Integral())
            bkg_test.SetLineWidth(1)
            bkg_test.SetMarkerStyle(20)
            bkg_test.SetMarkerSize(2)

            plotOptions = {"logscale": self.logscale}

            # init canvas
            canvas = setup.drawClosureTestOnCanvas(
                sig_train, bkg_train, sig_test, bkg_test, plotOptions,
                canvasName = "closure test at {} node".format(node_cls))

            # setup legend
            legend = setup.getLegend()

            # add entries
            legend.AddEntry(sig_train, "train {}".format("+".join(signalClass)), "F")
            legend.AddEntry(bkg_train, "train bkg", "F")
            legend.AddEntry(sig_test,  "test {}".format("+".join(signalClass)), "L")
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





class plotEventYields:
    def __init__(self, data, prediction_vector, event_classes, event_category, signal_class, plotdir, logscale, sigScale = -1):
        self.data               = data
        self.prediction_vector  = prediction_vector
        self.predicted_classes  = np.argmax(self.prediction_vector, axis = 1)

        self.event_classes      = event_classes
        self.n_classes          = len(self.event_classes)
        self.signal_class       = signal_class
        self.signalIndex       = []

        if self.signal_class:
            for signal in signal_class:
                self.signalIndex.append(self.data.class_translation[signal])
        else:
            self.signalIndex = [self.data.class_translation["ttHbb"]]

        self.event_category     = event_category
        self.plotdir            = plotdir

        self.logscale           = logscale
        self.sigScale           = sigScale

        self.privateWork = False

    def plot(self, privateWork = False, ratio = False):
        self.privateWork = privateWork

        # loop over processes
        sigHists = []
        sigLabels = []
        bkgHists = []
        bkgLabels = []

        plotOptions = {
            "ratio":      ratio,
            "ratioTitle": "#frac{scaled Signal}{Background}",
            "logscale":   self.logscale}
        yTitle = "event Yield"
        if privateWork:
            yTitle = setup.GetyTitle(privateWork)

        totalBkgYield = 0

        # generate one plot per output node
        for i, truth_cls in enumerate(self.event_classes):
            classIndex = self.data.class_translation[truth_cls]

            class_yields = []

            # loop over output nodes
            for j, node_cls in enumerate(self.event_classes):

                # get output values of this node
                out_values = self.prediction_vector[:,i]

                nodeIndex = self.data.class_translation[node_cls]

                # get yields
                class_yield = sum([ self.data.get_lumi_weights()[k] for k in range(len(out_values)) \
                    if self.data.get_test_labels(as_categorical = False)[k] == classIndex \
                    and self.predicted_classes[k] == nodeIndex])
                class_yields.append(class_yield)



            if i in self.signalIndex:
                histogram = setup.setupYieldHistogram(
                    yields  = class_yields,
                    classes = self.event_classes,
                    xtitle  = str(truth_cls)+" event yield",
                    ytitle  = yTitle,
                    color   = setup.GetPlotColor(truth_cls),
                    filled  = False)

                # set signal histogram linewidth
                histogram.SetLineWidth(2)
                sigHists.append(histogram)
                sigLabels.append(truth_cls)


            else:
                histogram = setup.setupYieldHistogram(
                    yields  = class_yields,
                    classes = self.event_classes,
                    xtitle  = str(truth_cls)+" event yield",
                    ytitle  = yTitle,
                    color   = setup.GetPlotColor(truth_cls),
                    filled  = True)
                bkgHists.append(histogram)
                bkgLabels.append(truth_cls)

                totalBkgYield += sum(class_yields)



        # scale histograms according to options
        scaleFactors=[]
        for sig in sigHists:
            if self.sigScale == -1:
                scaleFactors.append(totalBkgYield/sig.Integral())
            else:
                scaleFactors.append(float(self.sigScale))
        if privateWork:
            for sig in sigHists:
                sig.Scale(1./sig.Integral())
            for h in bkgHists:
                h.Scale(1./totalBkgYield)
        else:
            for i,sig in enumerate(sigHists):
                sig.Scale(scaleFactors[i])

        # initialize canvas
        canvas = setup.drawHistsOnCanvas(
            sigHists, bkgHists, plotOptions,
            canvasName = "event yields per node")

        # setup legend
        legend = setup.getLegend()

        # add signal entry
        for i,sig in enumerate(sigHists):
            legend.AddEntry(sig, sigLabels[i]+" x {:4.0f}".format(scaleFactors[i]), "L")

        # add background entries
        for i, h in enumerate(bkgHists):
            legend.AddEntry(h, bkgLabels[i], "F")

        # draw legend
        legend.Draw("same")

        # add lumi
        setup.printLumi(canvas, ratio = plotOptions["ratio"])

        # add category label
        setup.printCategoryLabel(canvas, self.event_category, ratio = plotOptions["ratio"])

        out_path = self.plotdir + "/event_yields.pdf"
        setup.saveCanvas(canvas, out_path)


class plotBinaryOutput:
    def __init__(self, data, test_predictions, train_predictions, nbins, bin_range, event_category, plotdir, logscale = False, sigScale = -1):
        self.data               = data
        self.test_predictions   = test_predictions
        self.train_predictions  = train_predictions
        self.nbins              = nbins
        self.bin_range          = bin_range
        self.event_category     = event_category
        self.plotdir            = plotdir
        self.logscale           = logscale
        self.sigScale           = sigScale

        self.printROCScore = False
        self.privateWork = False

    def plot(self, ratio = False, printROC = False, privateWork = False, name = "binary discriminator"):
        self.printROCScore = printROC
        self.privateWork = privateWork


        if self.printROCScore:
            roc = roc_auc_score(self.data.get_test_labels(), self.test_predictions)
            print("ROC: {}".format(roc))

        sig_values = [ self.test_predictions[k] for k in range(len(self.test_predictions)) \
            if self.data.get_test_labels()[k] == 1 ]
        sig_weights =[ self.data.get_lumi_weights()[k] for k in range(len(self.test_predictions)) \
            if self.data.get_test_labels()[k] == 1]
        sig_hist = setup.setupHistogram(
            values      = sig_values,
            weights     = sig_weights,
            nbins       = self.nbins,
            bin_range   = self.bin_range,
            color       = ROOT.kCyan,
            xtitle      = "signal",
            ytitle      = setup.GetyTitle(self.privateWork),
            filled      = False)
        sig_hist.SetLineWidth(3)

        bkg_values = [ self.test_predictions[k] for k in range(len(self.test_predictions)) \
            if not self.data.get_test_labels()[k] == 1 ]
        bkg_weights =[ self.data.get_lumi_weights()[k] for k in range(len(self.test_predictions)) \
            if not self.data.get_test_labels()[k] == 1]
        bkg_hist = setup.setupHistogram(
            values      = bkg_values,
            weights     = bkg_weights,
            nbins       = self.nbins,
            bin_range   = self.bin_range,
            color       = ROOT.kOrange,
            xtitle      = "background",
            ytitle      = setup.GetyTitle(self.privateWork),
            filled      = True)

        if self.sigScale == -1:
            scaleFactor = sum(bkg_weights)/(sum(sig_weights)+1e-9)
        else:
            scaleFactor = float(self.sigScale)
        sig_hist.Scale(scaleFactor)

        # rescale histograms if privateWork enabled
        if privateWork:
            sig_hist.Scale(1./sig_hist.Integral())
            bkg_hist.Scale(1./bkg_hist.Integral())

        plotOptions = {
            "ratio":      ratio,
            "ratioTitle": "#frac{scaled Signal}{Background}",
            "logscale":   self.logscale}

        # initialize canvas
        canvas = setup.drawHistsOnCanvas(
            sig_hist, bkg_hist, plotOptions,
            canvasName = name)

        # setup legend
        legend = setup.getLegend()

        # add signal entry
        legend.AddEntry(sig_hist, "signal x {:4.0f}".format(scaleFactor), "L")

        # add background entries
        legend.AddEntry(bkg_hist, "background", "F")

        # draw legend
        legend.Draw("same")

        # add ROC score if activated
        if self.printROCScore:
            setup.printROCScore(canvas, roc, plotOptions["ratio"])

        # add lumi or private work label to plot
        if self.privateWork:
            setup.printPrivateWork(canvas, plotOptions["ratio"], nodePlot = True)
        else:
            setup.printLumi(canvas, ratio = plotOptions["ratio"])

        # add category label
        setup.printCategoryLabel(canvas, self.event_category, ratio = plotOptions["ratio"])

        out_path = self.plotdir + "/binaryDiscriminator.pdf"
        setup.saveCanvas(canvas, out_path)



class plotEventYields:
    def __init__(self, data, prediction_vector, event_classes, event_category, signal_class, plotdir, logscale):
        self.data               = data
        self.prediction_vector  = prediction_vector
        self.predicted_classes  = np.argmax(self.prediction_vector, axis = 1)

        self.event_classes      = event_classes
        self.n_classes          = len(self.event_classes)
        self.signal_class       = signal_class
        self.signalIndex       = []

        if self.signal_class:
            for signal in signal_class:
                self.signalIndex.append(self.data.class_translation[signal])
        else:
            self.signalIndex = [self.data.class_translation["ttH"]]

        self.event_category     = event_category
        self.plotdir            = plotdir

        self.logscale           = logscale

        self.privateWork = False

    def plot(self, privateWork = False, ratio = False):
        self.privateWork = privateWork

        # loop over processes
        sigHists = []
        sigLabels = []
        bkgHists = []
        bkgLabels = []

        plotOptions = {
            "ratio":      ratio,
            "ratioTitle": "#frac{scaled Signal}{Background}",
            "logscale":   self.logscale}
        yTitle = "event Yield"
        if privateWork:
            yTitle = setup.GetyTitle(privateWork)

        totalBkgYield = 0

        # generate one plot per output node
        for i, truth_cls in enumerate(self.event_classes):
            classIndex = self.data.class_translation[truth_cls]

            class_yields = []

            # loop over output nodes
            for j, node_cls in enumerate(self.event_classes):

                # get output values of this node
                out_values = self.prediction_vector[:,i]

                nodeIndex = self.data.class_translation[node_cls]

                # get yields
                class_yield = sum([ self.data.get_lumi_weights()[k] for k in range(len(out_values)) \
                    if self.data.get_test_labels(as_categorical = False)[k] == classIndex \
                    and self.predicted_classes[k] == nodeIndex])
                class_yields.append(class_yield)



            if i in self.signalIndex:
                histogram = setup.setupYieldHistogram(
                    yields  = class_yields,
                    classes = self.event_classes,
                    xtitle  = str(truth_cls)+" event yield",
                    ytitle  = yTitle,
                    color   = setup.GetPlotColor(truth_cls),
                    filled  = False)

                # set signal histogram linewidth
                histogram.SetLineWidth(3)
                sigHists.append(histogram)
                sigLabels.append(truth_cls)


            else:
                histogram = setup.setupYieldHistogram(
                    yields  = class_yields,
                    classes = self.event_classes,
                    xtitle  = str(truth_cls)+" event yield",
                    ytitle  = yTitle,
                    color   = setup.GetPlotColor(truth_cls),
                    filled  = True)
                bkgHists.append(histogram)
                bkgLabels.append(truth_cls)

                totalBkgYield += sum(class_yields)



        # scale histograms according to options
        scaleFactors=[]
        for sig in sigHists:
            scaleFactors.append(totalBkgYield/sig.Integral())
        if privateWork:
            for sig in sigHists:
                sig.Scale(1./sig.Integral())
            for h in bkgHists:
                h.Scale(1./totalBkgYield)
        else:
            for i,sig in enumerate(sigHists):
                sig.Scale(scaleFactors[i])

        # initialize canvas
        canvas = setup.drawHistsOnCanvas(
            sigHists, bkgHists, plotOptions,
            canvasName = "event yields per node")

        # setup legend
        legend = setup.getLegend()

        # add signal entry
        for i,sig in enumerate(sigHists):
            legend.AddEntry(sig, sigLabels[i]+" x {:4.0f}".format(scaleFactors[i]), "L")

        # add background entries
        for i, h in enumerate(bkgHists):
            legend.AddEntry(h, bkgLabels[i], "F")

        # draw legend
        legend.Draw("same")

        # add lumi
        setup.printLumi(canvas, ratio = plotOptions["ratio"])

        # add category label
        setup.printCategoryLabel(canvas, self.event_category, ratio = plotOptions["ratio"])

        out_path = self.plotdir + "/event_yields.pdf"
        setup.saveCanvas(canvas, out_path)
