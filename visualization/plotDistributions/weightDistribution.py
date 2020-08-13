# -*- coding: utf-8 -*-
'''
draw output-weights-histogram
'''
# imports
import numpy as np
import pandas as pd
import ROOT
from ROOT import gStyle

ROOT.gROOT.SetBatch(True)

def setupHistogram(
        values,
        xtitle, ytitle,
        color = ROOT.kBlack, filled = True):
    # define histogram
    histogram = ROOT.TH1D(xtitle, "", 80, -.5, .5)
    #histogram.Sumw2(True)

    for value in values:
        histogram.Fill(value)

    #histogram.SetStats(False)
    histogram.GetXaxis().SetTitle(xtitle)
    histogram.GetYaxis().SetTitle(ytitle)

    histogram.GetYaxis().SetTitleOffset(1.2)
    histogram.GetXaxis().SetTitleOffset(1.2)
    histogram.GetYaxis().SetTitleSize(0.055)
    histogram.GetXaxis().SetTitleSize(0.055)
    histogram.GetYaxis().SetLabelSize(0.045)
    histogram.GetXaxis().SetLabelSize(0.045)

    #histogram.SetMarkerColor(color)

    if filled:
        histogram.SetLineColor( ROOT.kBlack )
        histogram.SetFillColor( color )
        histogram.SetLineWidth(1)
    else:
        histogram.SetLineColor( color )
        histogram.SetFillColor(0)
        histogram.SetLineWidth(3)

    return histogram



# path
wb = eval(open('weights_before.txt', 'r').read())
wa = eval(open('weights_after.txt', 'r').read())



hist_b = setupHistogram(
    values      = wb,
    color       = ROOT.kBlue+1,
    xtitle      = "Verteilung der Gewichte",
    ytitle      = "Anzahl an Gewichten",
    filled      = False)

hist_a = setupHistogram(
    values      = wa,
    color       = ROOT.kRed+1,
    xtitle      = "Verteilung der Gewichte",
    ytitle      = "Anzahl an Gewichten",
    filled      = False)

gStyle.SetOptStat(0);

# initialize canvas
canvas = ROOT.TCanvas("c","c",800,600)
#canvas.SetLogx()
canvas.SetBottomMargin(0.15)
canvas.SetLeftMargin(0.15)

legend=ROOT.TLegend(0.59,0.7,0.77,0.9)
legend.SetBorderSize(0);
legend.SetLineStyle(0);
legend.SetTextFont(42);
legend.SetTextSize(0.047);
legend.SetFillStyle(0);

# add signal entry
legend.AddEntry(hist_b,  'vor dem Training', "L")

# add background entries
legend.AddEntry(hist_a, "nach dem Training", "L")

hist_b.Draw("hist")
hist_a.Draw("histsame")

#draw legend
legend.Draw("same")


canvas.SaveAs("weights_distr.pdf")
