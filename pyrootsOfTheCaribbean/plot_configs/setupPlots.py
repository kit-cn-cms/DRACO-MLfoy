import ROOT
ROOT.gROOT.SetBatch(True)
import re
import numpy as np

# dictionary for colors
def GetPlotColor( cls ):
    color_dict = {
        "ttZ":   ROOT.kBlue+4,
        "ttH":   ROOT.kBlue+1,
        "ttlf":  ROOT.kRed-7,
        "ttcc":  ROOT.kRed+1,
        "ttbb":  ROOT.kRed+3,
        "tt2b":  ROOT.kRed+2,
        "ttb":   ROOT.kRed-2,
        "ttbar": ROOT.kOrange,
        }

    if "ttZ" in cls: cls = "ttZ"
    if "ttH" in cls: cls = "ttH"
    return color_dict[cls]

def GetyTitle(privateWork = False):
    # if privateWork flag is enabled, normalize plots to unit area
    if privateWork:
        return "normalized to unit area"
    return "Events expected"


# ===============================================
# SETUP OF HISTOGRAMS 
# ===============================================
def setupHistogram(
        values, weights, 
        nbins, bin_range,
        xtitle, ytitle, 
        color = ROOT.kBlack, filled = True):
    # define histogram
    histogram = ROOT.TH1D(xtitle.replace(" ","_"), "", nbins, *bin_range)
    histogram.Sumw2(True)    

    for v, w in zip(values, weights):
        histogram.Fill(v, w)

    histogram.SetStats(False)
    histogram.GetXaxis().SetTitle(xtitle)
    histogram.GetYaxis().SetTitle(ytitle)

    histogram.GetYaxis().SetTitleOffset(1.4)
    histogram.GetXaxis().SetTitleOffset(1.2)
    histogram.GetYaxis().SetTitleSize(0.055)
    histogram.GetXaxis().SetTitleSize(0.055)
    histogram.GetYaxis().SetLabelSize(0.055)
    histogram.GetXaxis().SetLabelSize(0.055)

    histogram.SetMarkerColor(color)

    if filled:
        histogram.SetLineColor( ROOT.kBlack )
        histogram.SetFillColor( color )
        histogram.SetLineWidth(1)
    else:
        histogram.SetLineColor( color )
        histogram.SetFillColor(0)
        histogram.SetLineWidth(2)

    return histogram



def setupConfusionMatrix(matrix, ncls, xtitle, ytitle, binlabel, errors = None):
    # check if errors for matrix are given
    has_errors = isinstance(errors, np.ndarray)
    #print(has_errors)
    
    # init histogram
    cm = ROOT.TH2D("confusionMatrix", "", ncls, 0, ncls, ncls, 0, ncls)
    cm.SetStats(False)
    ROOT.gStyle.SetPaintTextFormat(".3f")
        

    for xit in range(cm.GetNbinsX()):
        for yit in range(cm.GetNbinsY()):
            cm.SetBinContent(xit+1,yit+1, matrix[xit, yit])
            if has_errors:
                cm.SetBinError(xit+1,yit+1, errors[xit, yit])

    cm.GetXaxis().SetTitle(xtitle)
    cm.GetYaxis().SetTitle(ytitle)

    cm.SetMarkerColor(ROOT.kWhite)

    minimum = np.min(matrix)
    maximum = np.max(matrix)

    cm.GetZaxis().SetRangeUser(minimum, maximum)

    for xit in range(ncls):
        cm.GetXaxis().SetBinLabel(xit+1, binlabel[xit])
    for yit in range(ncls):
        cm.GetYaxis().SetBinLabel(yit+1, binlabel[yit])

    cm.GetXaxis().SetLabelSize(0.05)
    cm.GetYaxis().SetLabelSize(0.05)
    cm.SetMarkerSize(2.)

    return cm



# ===============================================
# DRAW HISTOGRAMS ON CANVAS
# ===============================================
def drawConfusionMatrixOnCanvas(matrix, canvasName, catLabel, ROC = None, ROCerr = None, privateWork = False):
    # init canvas
    canvas = ROOT.TCanvas(canvasName, canvasName, 1024, 1024)
    canvas.SetTopMargin(0.15)
    canvas.SetBottomMargin(0.15)
    canvas.SetRightMargin(0.15)
    canvas.SetLeftMargin(0.15)
    canvas.SetTicks(1,1)

    # draw histogram
    #ROOT.gStyle.SetPalette(69)
    draw_option = "colz text1"
    if ROCerr: draw_option += "e"
    matrix.DrawCopy(draw_option)

    # setup TLatex
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextColor(ROOT.kBlack)
    latex.SetTextSize(0.03)

    l = canvas.GetLeftMargin()
    t = canvas.GetTopMargin()

    # add category label
    latex.DrawLatex(l,1.-t+0.01, catLabel)

    if privateWork:
        latex.DrawLatex(l, 1.-t+0.04, "CMS private work")

    # add ROC score if activated
    if ROC:
        text = "ROC-AUC = {:.3f}".format(ROC)
        if ROCerr:
            text += "#pm {:.3f}".format(ROCerr)
            latex.DrawLatex(l+0.4,1.-t+0.01, text)
        else:
            latex.DrawLatex(l+0.47,1.-t+0.01, text)

    
    return canvas


def drawClosureTestOnCanvas(sig_train, bkg_train, sig_test, bkg_test, plotOptions, canvasName):
    canvas = getCanvas(canvasName)

    # move over/underflow bins into plotrange
    moveOverUnderFlow(sig_train)
    moveOverUnderFlow(bkg_train)
    moveOverUnderFlow(sig_test)
    moveOverUnderFlow(bkg_test)

    # figure out plotrange
    canvas.cd(1)
    yMax = 1e-9
    yMinMax = 1000.
    for h in [sig_train, bkg_train, sig_test, bkg_test]:
        yMax = max(h.GetBinContent(h.GetMaximumBin()), yMax)
        if h.GetBinContent(h.GetMaximumBin()) > 0:
            yMinMax = min(h.GetBinContent(h.GetMaximumBin()), yMinMax)

    # draw first hist
    if plotOptions["logscale"]:
        bkg_train.GetYaxis().SetRangeUser(yMinMax/10000, yMax*10)
        canvas-SetLogy()
    else:
        bkg_train.GetYaxis().SetRangeUser(0, yMax*1.5)
    bkg_train.GetXaxis().SetTitle(generateLatexLabel(canvasName))

    option = "histo"
    bkg_train.DrawCopy(option+"E0")

    # draw the other histograms
    sig_train.DrawCopy(option+"E0 same")
    bkg_test.DrawCopy("E0 same")
    sig_test.DrawCopy("E0 same")

    # redraw axis
    canvas.cd(1)
    bkg_train.DrawCopy("axissame")

    return canvas

def drawHistsOnCanvas(sigHists, bkgHists, plotOptions, canvasName):
    if not isinstance(sigHists, list):
        sigHists = [sigHists]
    if not isinstance(bkgHists, list):
        bkgHists = [bkgHists]
    
    canvas = getCanvas(canvasName, plotOptions["ratio"])

    # move over/underflow bins into plotrange
    for h in bkgHists:
        moveOverUnderFlow(h)
    for h in sigHists:
        moveOverUnderFlow(h)
    
    # stack Histograms
    bkgHists = [bkgHists[len(bkgHists)-1-i] for i in range(len(bkgHists))]
    for i in range(len(bkgHists)-1, 0, -1):
        bkgHists[i-1].Add(bkgHists[i])

    # figure out plotrange
    canvas.cd(1)
    yMax = 1e-9
    yMinMax = 1000.
    for h in bkgHists:
        yMax = max(h.GetBinContent(h.GetMaximumBin()), yMax)
        if h.GetBinContent(h.GetMaximumBin()) > 0:
            yMinMax = min(h.GetBinContent(h.GetMaximumBin()), yMinMax)
    
    # draw the first histogram
    if len(bkgHists) == 0:
        firstHist = sigHists[0]
    else:
        firstHist = bkgHists[0]
    if plotOptions["logscale"]:
        firstHist.GetYaxis().SetRangeUser(yMinMax/10000, yMax*10)
        canvas.SetLogy()
    else:
        firstHist.GetYaxis().SetRangeUser(0, yMax*1.5)
    firstHist.GetXaxis().SetTitle(generateLatexLabel(canvasName))

    option = "histo"
    firstHist.DrawCopy(option+"E0")

    # draw the other histograms
    for h in bkgHists[1:]:
        h.DrawCopy(option+"same")

    canvas.cd(1)
    # redraw axis
    firstHist.DrawCopy("axissame")

    
    # draw signal histograms
    for sH in sigHists:
        # draw signal histogram
        sH.DrawCopy(option+" E0 same")
    

    if plotOptions["ratio"]:
        canvas.cd(2)
        line = sigHists[0].Clone()
        line.Divide(sigHists[0])
        line.GetYaxis().SetRangeUser(0.5,1.5)
        line.GetYaxis().SetTitle(plotOptions["ratioTitle"])

        line.GetXaxis().SetLabelSize(line.GetXaxis().GetLabelSize()*2.4)
        line.GetYaxis().SetLabelSize(line.GetYaxis().GetLabelSize()*2.2)
        line.GetXaxis().SetTitle(generateLatexLabel(canvasName))

        line.GetXaxis().SetTitleSize(line.GetXaxis().GetTitleSize()*3)
        line.GetYaxis().SetTitleSize(line.GetYaxis().GetTitleSize()*2.5)

        line.GetYaxis().SetTitleOffset(0.5)
        line.GetYaxis().SetNdivisions(505)
        for i in range(line.GetNbinsX()+1):
            line.SetBinContent(i, 1)
            line.SetBinError(i, 1)
        line.SetLineWidth(1)
        line.SetLineColor(ROOT.kBlack)
        line.DrawCopy("histo")
        # ratio plots
        for sigHist in sigHists:
            ratioPlot = sigHist.Clone()
            ratioPlot.Divide(bkgHists[0])
            ratioPlot.SetTitle(generateLatexLabel(canvasName))
            ratioPlot.SetLineColor(sigHist.GetLineColor())
            ratioPlot.SetLineWidth(1)
            ratioPlot.SetMarkerStyle(20)
            ratioPlot.SetMarkerColor(sigHist.GetMarkerColor())
            ROOT.gStyle.SetErrorX(0)
            ratioPlot.DrawCopy("sameP")
        canvas.cd(1)
    return canvas
    


# ===============================================
# GENERATE CANVAS AND LEGENDS
# ===============================================
def getCanvas(name, ratiopad = False):
    if ratiopad:
        canvas = ROOT.TCanvas(name, name, 1024, 1024)
        canvas.Divide(1,2)
        canvas.cd(1).SetPad(0.,0.3,1.0,1.0)
        canvas.cd(1).SetTopMargin(0.07)
        canvas.cd(1).SetBottomMargin(0.0)

        canvas.cd(2).SetPad(0.,0.0,1.0,0.3)
        canvas.cd(2).SetTopMargin(0.0)
        canvas.cd(2).SetBottomMargin(0.4)

        canvas.cd(1).SetRightMargin(0.05)
        canvas.cd(1).SetLeftMargin(0.15)
        canvas.cd(1).SetTicks(1,1)

        canvas.cd(2).SetRightMargin(0.05)
        canvas.cd(2).SetLeftMargin(0.15)
        canvas.cd(2).SetTicks(1,1)
    else:
        canvas = ROOT.TCanvas(name, name, 1024, 768)
        canvas.SetTopMargin(0.07)
        canvas.SetBottomMargin(0.15)
        canvas.SetRightMargin(0.05)
        canvas.SetLeftMargin(0.15)
        canvas.SetTicks(1,1)

    return canvas

def getLegend():
    legend=ROOT.TLegend(0.70,0.6,0.95,0.9)
    legend.SetBorderSize(0);
    legend.SetLineStyle(0);
    legend.SetTextFont(42);
    legend.SetTextSize(0.05);
    legend.SetFillStyle(0);
    return legend

def saveCanvas(canvas, path):
    canvas.SaveAs(path)
    canvas.SaveAs(path.replace(".pdf",".png"))
    canvas.Clear()


# ===============================================
# PRINT STUFF ON CANVAS
# ===============================================
def printLumi(pad, lumi = 41.5, ratio = False, twoDim = False):
    if lumi == 0.: return

    lumi_text = str(lumi)+" fb^{-1} (13 TeV)"

    pad.cd(1)
    l = pad.GetLeftMargin()
    t = pad.GetTopMargin()
    r = pad.GetRightMargin()
    b = pad.GetBottomMargin()

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextColor(ROOT.kBlack)
    
    if twoDim:  latex.DrawLatex(l+0.40,1.-t+0.01,lumi_text)
    elif ratio: latex.DrawLatex(l+0.60,1.-t+0.04,lumi_text)
    else:       latex.DrawLatex(l+0.53,1.-t+0.02,lumi_text)

def printCategoryLabel(pad, catLabel, ratio = False):
    pad.cd(1)
    l = pad.GetLeftMargin()
    t = pad.GetTopMargin()
    r = pad.GetRightMargin()
    b = pad.GetBottomMargin()

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextColor(ROOT.kBlack)

    if ratio:   latex.DrawLatex(l+0.07,1.-t-0.04, catLabel)
    else:       latex.DrawLatex(l+0.02,1.-t-0.06, catLabel)

def printROCScore(pad, ROC, ratio = False):
    pad.cd(1)
    l = pad.GetLeftMargin()
    t = pad.GetTopMargin()
    r = pad.GetRightMargin()
    b = pad.GetBottomMargin()
    
    text = "ROC-AUC = {:.3f}".format(ROC)

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextColor(ROOT.kBlack)

    if ratio:   latex.DrawLatex(l+0.05,1.-t+0.04, text)
    else:       latex.DrawLatex(l,1.-t+0.02, text)

def printPrivateWork(pad, ratio = False, twoDim = False, nodePlot = False):
    pad.cd(1) 
    l = pad.GetLeftMargin() 
    t = pad.GetTopMargin() 
    r = pad.GetRightMargin() 
    b = pad.GetBottomMargin() 
 
    latex = ROOT.TLatex() 
    latex.SetNDC() 
    latex.SetTextColor(ROOT.kBlack) 
    latex.SetTextSize(0.04)

    text = "CMS private work" 

    if nodePlot:    latex.DrawLatex(l+0.57,1.-t+0.01, text)
    elif twoDim:    latex.DrawLatex(l+0.39,1.-t+0.01, text)
    elif ratio:     latex.DrawLatex(l+0.05,1.-t+0.04, text) 
    else:           latex.DrawLatex(l,1.-t+0.01, text)

def printTitle(pad, title):
    pad.cd(1)
    l = pad.GetLeftMargin()
    t = pad.GetTopMargin()
    r = pad.GetRightMargin()
    b = pad.GetBottomMargin()

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextColor(ROOT.kBlack)
    latex.SetTextSize(0.03)
    latex.DrawLatex(l, 1.-t+0.06, title)

def generateLatexLabel(name):
    ''' try to make plot label nicer '''
    # remove starters
    starts = ["Evt_", "BDT_common5_input_"]
    for s in starts:
        if name.startswith(s):
            name = name[len(s):]
    
    if name.startswith("N_"):
        return "N("+name[2:]+")"

    # replace stuff
    name = name.replace("DeltaR","#DeltaR")
    name = name.replace("Dr","#DeltaR")
    name = name.replace("dR","#DeltaR")
    name = name.replace("deltaR","#DeltaR")

    name = name.replace("dY","#Deltay")

    name = name.replace("eta", "#eta")
    name = name.replace("Eta", "#eta")
    name = name.replace("Delta#eta","#Delta#eta")
    name = name.replace("d#eta","#Delta#eta")
    name = name.replace("D#eta","#Delta#eta")

    name = name.replace("phi", "#phi")
    name = name.replace("Phi", "#phi")
    name = name.replace("d#phi","#Delta#phi")
    name = name.replace("Delta#phi","#Delta#phi")

    name = name.replace("mass","M")
    name = name.replace("pT", "p_{T}")
    name = name.replace("pt", "p_{T}")

    # some specials
    name = name.replace("lep)Jet","(lep,Jet)")
    name = name.replace("lep)TaggedJet","(lep,taggedJet)")
    name = name.replace("Looselep)","LooseLepton")
    name = name.replace("primarylep)","primary lepton")
    name = name.replace("MHT", "missing H_{T}")
    name = name.replace("HT","H_{T}")
    name = name.replace("Jetp_{T}OverJetE","average p_{T}^{jet}/E^{jet}")
    name = name.replace("blr_ETH", "b-tag likelihood ratio")
    return name


def moveOverUnderFlow(h):
    # move underflow
    h.SetBinContent(1, h.GetBinContent(0)+h.GetBinContent(1))
    # move overflow
    h.SetBinContent(h.GetNbinsX(), h.GetBinContent(h.GetNbinsX()+1)+h.GetBinContent(h.GetNbinsX()))

    # set underflow error
    h.SetBinError(1, ROOT.TMath.Sqrt(
        ROOT.TMath.Power(h.GetBinError(0),2) + ROOT.TMath.Power(h.GetBinError(1),2) ))
    # set overflow error
    h.SetBinError(h.GetNbinsX(), ROOT.TMath.Sqrt(
        ROOT.TMath.Power(h.GetBinError(h.GetNbinsX()),2) + ROOT.TMath.Power(h.GetBinError(h.GetNbinsX()+1),2) ))


def calculateKSscore(stack, sig):
    return stack.KolmogorovTest(sig)











