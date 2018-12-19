import ROOT
ROOT.gROOT.SetBatch(True)
import re
import numpy as np

# dictionary for colors
def GetPlotColor( cls ):
    color_dict = {
        "ttH":   ROOT.kBlue+1,
        "ttlf":  ROOT.kRed-7,
        "ttcc":  ROOT.kRed+1,
        "ttbb":  ROOT.kRed+3,
        "tt2b":  ROOT.kRed+2,
        "ttb":   ROOT.kRed-2,
        "False": "orangered",
        "True":  "teal"
        }

    if "ttH" in cls: cls = "ttH"
    return color_dict[cls]

def GetyTitle():
    return "Events expected for 41.5 fb^{-1} @ 13 TeV"


def GetCategoryLabel(cat):
    cat_dict = {
        "ge6j_ge3t":    "1 lepton, \geq 6 jets, \geq 3 b-tags",
        "5j_ge3t":      "1 lepton, 5 jets, \geq 3 b-tags",
        "4j_ge3t":      "1 lepton, 4 jets, \geq 3 b-tags",
        "(N_Jets >= 6 and N_BTagsM >= 3)":
                        "1 lepton, \geq 6 jets, \geq 3 b-tags",
        "(N_Jets == 5 and N_BTagsM >= 3)":
                        "1 lepton, 5 jets, \geq 3 b-tags",
        "(N_Jets == 4 and N_BTagsM >= 3)":
                        "1 lepton, 4 jets, \geq 3 b-tags",
        "(N_Jets == 4 and N_BTagsM == 4)":
                        "1 lepton, 4 jets, 4 b-tags",
        }
    return cat_dict[cat]

def setupHistogram(
        values, weights, 
        nbins, bin_range, color,
        xtitle, ytitle, filled = True):
    # define histogram
    histogram = ROOT.TH1D(xtitle.replace(" ","_"), "", nbins, *bin_range)
    
    for v, w in zip(values, weights):
        histogram.Fill(v, w)

    histogram.SetStats(False)
    histogram.GetXaxis().SetTitle(xtitle)
    histogram.GetYaxis().SetTitle(GetyTitle())

    histogram.GetYaxis().SetTitleOffset(1.4)
    histogram.GetXaxis().SetTitleOffset(1.2)
    histogram.GetYaxis().SetTitleSize(0.05)
    histogram.GetXaxis().SetTitleSize(0.05)
    histogram.GetYaxis().SetLabelSize(0.05)
    histogram.GetXaxis().SetLabelSize(0.05)

    histogram.SetMarkerColor(color)

    if filled:
        histogram.SetLineColor( ROOT.kBlack )
        histogram.SetFillColor( color )
        histogram.SetLineWidth(2)
    else:
        histogram.SetLineColor( color )
        histogram.SetFillColor(0)
        histogram.SetLineWidth(2)

    return histogram

def setup2DHistogram(matrix, ncls, xtitle, ytitle, binlabel):
    cm = ROOT.TH2D("2Dhistogram", "", ncls, 0, ncls, ncls, 0, ncls)
    cm.SetStats(False)
    ROOT.gStyle.SetPaintTextFormat(".3f")

    for xit in range(cm.GetNbinsX()):
        for yit in range(cm.GetNbinsY()):
            cm.SetBinContent(xit+1,yit+1, matrix[xit, yit])

    cm.GetXaxis().SetTitle(xtitle)
    cm.GetYaxis().SetTitle(ytitle)

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

def draw2DHistOnCanvas(hist, canvasName, cat, ROC = None):
    # init canvas
    canvas = ROOT.TCanvas(canvasName, canvasName, 1024, 1024)
    canvas.SetTopMargin(0.15)
    canvas.SetBottomMargin(0.15)
    canvas.SetRightMargin(0.15)
    canvas.SetLeftMargin(0.15)
    canvas.SetTicks(1,1)

    # draw histogram
    #ROOT.gStyle.SetPalette(69)
    hist.DrawCopy("colz text1")

    # setup TLatex
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextColor(ROOT.kBlack)
    latex.SetTextSize(0.03)

    l = canvas.GetLeftMargin()
    t = canvas.GetTopMargin()

    # add category label
    latex.DrawLatex(l,1.-t+0.01, GetCategoryLabel(cat))

    # add ROC score if activated
    if ROC:
        text = "ROC-AUC = {:.3f}".format(ROC)
        latex.DrawLatex(l+0.47,1.-t+0.01, text)
    
    return canvas

    

def drawHistsOnCanvas(sigHist, bkgHists, plotOptions, canvasName):
    canvas = getCanvas(canvasName, plotOptions["ratio"])

    # move over/underflow bins into plotrange
    for h in bkgHists:
        moveOverUnderFlow(h)
    moveOverUnderFlow(sigHist)
    
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
    firstHist = bkgHists[0]
    if plotOptions["logscale"]:
        firstHist.GetYaxis().SetRangeUser(yMinMax/10000, yMax*10)
        canvas.SetLogy()
    else:
        firstHist.GetYaxis().SetRangeUser(0, yMax*1.5)
    firstHist.GetXaxis().SetTitle(canvasName)
    option = "histo"
    firstHist.DrawCopy(option)

    # draw the other histograms
    for h in bkgHists[1:]:
        h.DrawCopy(option+"same")
    
    # draw signal histogram
    sigHist.DrawCopy(option+"same")
    
    if plotOptions["ratio"]:
        canvas.cd(2)
        line = sigHist.Clone()
        line.Divide(sigHist)
        line.GetYaxis().SetRangeUser(0.5,1.5)
        line.GetYaxis().SetTitle(plotOptions["ratioTitle"])

        line.GetXaxis().SetLabelSize(line.GetXaxis().GetLabelSize()*2.4)
        line.GetYaxis().SetLabelSize(line.GetYaxis().GetLabelSize()*2.4)

        line.GetXaxis().SetTitleSize(line.GetXaxis().GetTitleSize()*3)
        line.GetYaxis().SetTitleSize(line.GetYaxis().GetTitleSize()*1.5)

        line.GetYaxis().SetTitleOffset(0.9)
        line.GetYaxis().SetNdivisions(505)
        for i in range(line.GetNbinsX()+1):
            line.SetBinContent(i, 1)
            line.SetBinError(i, 1)
        line.SetLineWidth(1)
        line.SetLineColor(ROOT.kBlack)
        line.DrawCopy("histo")
        ratioPlot = sigHist.Clone()
        ratioPlot.Divide(bkgHists[0])
        ratioPlot.SetTitle(canvasName)
        ratioPlot.SetLineColor(ROOT.kBlack)
        ratioPlot.SetLineWidth(ROOT.kBlack)
        ratioPlot.DrawCopy("sameP")
        canvas.cd(1)
    return canvas
    

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





def getLegend():
    legend=ROOT.TLegend(0.75,0.6,0.95,0.9)
    #legend.SetX1NDC(0.76)
    #legend.SetX2NDC(0.93)
    #legend.SetY1NDC(0.9)
    #legend.SetY2NDC(0.91)
    legend.SetBorderSize(0);
    legend.SetLineStyle(0);
    legend.SetTextFont(42);
    legend.SetTextSize(0.03);
    legend.SetFillStyle(0);
    return legend



def printLumi(pad, ratio = False):
    lumi_text = "41.5 fb^{-1} (13 TeV)"

    pad.cd(1)
    l = pad.GetLeftMargin()
    t = pad.GetTopMargin()
    r = pad.GetRightMargin()
    b = pad.GetBottomMargin()

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextColor(ROOT.kBlack)
    
    if ratio:   latex.DrawLatex(l+0.60,1.-t+0.04,lumi_text)
    else:       latex.DrawLatex(l+0.53,1.-t+0.02,lumi_text)

def printCategoryLabel(pad, cat, ratio = False):
    pad.cd(1)
    l = pad.GetLeftMargin()
    t = pad.GetTopMargin()
    r = pad.GetRightMargin()
    b = pad.GetBottomMargin()

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextColor(ROOT.kBlack)

    if ratio:   latex.DrawLatex(l+0.07,1.-t-0.04, GetCategoryLabel(cat))
    else:       latex.DrawLatex(l+0.02,1.-t-0.06, GetCategoryLabel(cat))

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

def saveCanvas(canvas, path):
    canvas.SaveAs(path)
    canvas.SaveAs(path.replace(".pdf",".png"))
    canvas.Clear()
