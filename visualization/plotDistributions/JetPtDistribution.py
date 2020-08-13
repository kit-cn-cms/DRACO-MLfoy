# -*- coding: utf-8 -*-
'''
draw Jet-Pt-histogram
'''
# imports
import numpy as np
import pandas as pd
import re
import base64
import ROOT
from ROOT import gStyle

ROOT.gROOT.SetBatch(True)

def decode_Samples(path):
    # reads h5 files and decodes channels, saves entrys for histogram

    # read input data out of h5 file
    with pd.HDFStore(path, mode = "r" ) as store:
        df = store.select("data", stop = 100000) #stop is arbitrary
        mi = store.select("meta_info")
        shape=list(mi["input_shape"])

    # set channels to decode
    columns_to_decode=[]
    for col in df.columns:
        m=re.match("(.*_Hist)", col)
        if m!=None:
            columns_to_decode.append(m.group(1))


    # decoding
    
    column_name = 'Jet_Pt[0-16]_Hist'
    empty_imgs_evtids=[]
    # collect JetPt entries
    JetPt_entries = []
    for index, row in df.iterrows():
        r=base64.b64decode(row[column_name])
        u=np.frombuffer(r,dtype=np.float64)
        u=np.reshape(u,shape)
            
        for line in u:
            for element in line:
                if element > 0.:
                    JetPt_entries.append(element)


    return JetPt_entries


def setupHistogram(
        values,
        xtitle, ytitle,
        color = ROOT.kBlack, filled = True):
    # define histogram
    histogram = ROOT.TH1D(xtitle, "", 100, 0.0, 2000.0)
    #histogram.Sumw2(True)

    for value in values:
        histogram.Fill(value)

    #histogram.SetStats(False)
    histogram.GetXaxis().SetTitle(xtitle)
    histogram.GetYaxis().SetTitle(ytitle)

    histogram.GetYaxis().SetTitleOffset(.9)
    histogram.GetXaxis().SetTitleOffset(.9)
    histogram.GetYaxis().SetTitleSize(0.05)
    histogram.GetXaxis().SetTitleSize(0.05)
    histogram.GetYaxis().SetLabelSize(0.04)
    histogram.GetXaxis().SetLabelSize(0.04)

    #histogram.SetMarkerColor(color)

    if filled:
        histogram.SetLineColor( ROOT.kBlack )
        histogram.SetFillColor( color )
        histogram.SetLineWidth(1)
    else:
        histogram.SetLineColor( color )
        histogram.SetFillColor(0)
        histogram.SetLineWidth(2)

    return histogram



# path
path1 = '/ceph/jvautz/NN/CNNInputs/testCNN/CSV_channel/all_rotations/ttH_CSV_rot_MaxJetPt.h5'
JetPt_entries = decode_Samples(path1)

#print JetPt_entries

# set quantile

quantile =np.quantile(JetPt_entries, 0.95)

entries= []
for entry in JetPt_entries:
    if entry < quantile:
        entries.append(entry)

hist = setupHistogram(
    values      = JetPt_entries,
    color       = ROOT.kBlue-8,
    xtitle      = "JetPt Verteilung [GeV]",
    ytitle      = "Anzahl an Jets",
    filled      = True)

hist_qu = setupHistogram(
    values      = entries,
    color       = ROOT.kBlue-2,
    xtitle      = "JetPt Verteilung [GeV]",
    ytitle      = "Anzahl an Jets",
    filled      = True)


gStyle.SetOptStat(0);

# initialize canvas
canvas = ROOT.TCanvas("c","c",800,600)
canvas.SetLogy()

# normalize to 1 integral 
if hist.Integral()>0:
    hist.Scale(1./float(hist.Integral()))
    hist_qu.Scale(0.95/float(hist_qu.Integral()))

legend=ROOT.TLegend(0.35,0.7,0.55,0.9)
legend.SetBorderSize(0);
legend.SetLineStyle(0);
legend.SetTextFont(42);
legend.SetTextSize(0.05);
legend.SetFillStyle(0);

# add signal entry
legend.AddEntry(hist_qu,  'Jets innerhalb des 95%-Quantils', "F")

# add background entries
legend.AddEntry(hist, "weitere Jets", "F")

hist.Draw("hist")
hist_qu.Draw("histsame")

#draw legend
legend.Draw("same")
hist.Draw("axis same")

'''
line = ROOT.TLine(quantile,0,quantile,0.001);
line.SetLineColor(ROOT.kPink+7);
line.SetLineWidth(2)
line.Draw();
'''
canvas.SaveAs("JetPt_distr.pdf")
