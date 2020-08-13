import ROOT
import json 

ROOT.gROOT.SetBatch(True)
# WORKS ON NAF
# samples is used. Older MC sampels fitting to 2017 data.  Change for your samples. Should also work. But you need the root files not the hdf5 files. 
# format: [ name, wildcard expression to root files, Color for sample]
# in principle it should work for many different samples (not only 2). For exampel if you have also ttbb events.
# BUT: the separation only works if the samples have different root files. If you want to define different samples using event variables, for example tt+lf vs tt+cc we could hack that too by modifying the event selection further down.
samples=[
    ["ttHbb", "/nfs/dust/cms/user/kelmorab/ttH_2018/ntuples_v5/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*root",ROOT.kBlue],
    ["ttbarSL", "/nfs/dust/cms/user/kelmorab/ttH_2018/ntuples_v5/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*root",ROOT.kRed],
    ]

# no need to wait for ages for all events
maxEvents=50000

# lists to hold the histos before plotting
histos=[]
histos_phi=[]
histos_eta=[]
histos_2D_eta_phi=[]
histos_minDr=[]

# loop over the samples 
for isample, sample in enumerate(samples):
    name=sample[0]
    path=sample[1]
    color=sample[2]
    
    # set up histograms
    # first histogram has not a good variable name because i hacked it in first. Sorry.
    # you could also change the number of bins for the histograms if you like

    hist=ROOT.TH1D("histo_"+name,"histo_"+name,100,0,5)
    hist.SetLineColor(color)
    hist.GetXaxis().SetTitle("dR(Jet,Jet)")
    
    hist_minDR=ROOT.TH1D("histo_minDR_"+name,"histo_minDR_"+name,100,0,5)
    hist_minDR.SetLineColor(color)
    hist_minDR.GetXaxis().SetTitle("min dR(Jet,Jet)")
    
    hist_phi=ROOT.TH1D("histo_phi_"+name,"histo_phi_"+name,100,0,5)
    hist_phi.SetLineColor(color)
    hist_phi.GetXaxis().SetTitle("dPhi(Jet,Jet)")
    
    hist_eta=ROOT.TH1D("histo_eta_"+name,"histo_eta_"+name,100,0,5)
    hist_eta.SetLineColor(color)
    hist_eta.GetXaxis().SetTitle("dEta(Jet,Jet)")
    
    hist_2D_eta_phi=ROOT.TH2D("histo_2D_eta_phi_"+name,"histo_2D_eta_phi_"+name,100,0,5,100,0,5)
    hist_2D_eta_phi.SetLineColor(color)
    hist_2D_eta_phi.GetXaxis().SetTitle("dEta(Jet,Jet)")
    hist_2D_eta_phi.GetYaxis().SetTitle("dPhi(Jet,Jet)")
    
    
    # create TChain object (can be treated like a TTree and add all root files for the sample to the chain
    chain=ROOT.TChain("MVATree")
    chain.Add(path)
    nEntries=chain.GetEntries()
    
    # loop over events in sample
    for ievt in range(min(maxEvents, nEntries)):
        if ievt%1000==0:
            print("at event "+str(ievt)+" of "+str(nEntries))
        chain.GetEntry(ievt)
        
        # simple event selection
        if not (chain.N_Jets >=4 and chain.N_BTagsM >=3):
            continue
        
        jets=[]
        dRs=[]
        dPhis=[]
        dEtas=[]
        # loop over jets in event and create Lorentz 4-vectors to later calculate dR values
        # the 4-vectors are then stored in the jets list
        for ijet in range(chain.N_Jets):
            lv=ROOT.TLorentzVector()
            lv.SetPtEtaPhiE(chain.Jet_Pt[ijet],chain.Jet_Eta[ijet],chain.Jet_Phi[ijet],chain.Jet_E[ijet])
            jets.append(lv)

        # calculate pairwise angles between jet 4-vectors and store in the lists defined above
        for ijet in range(len(jets)):
            for jjet in range(ijet+1,len(jets)):
                if ijet>=jjet:
                    continue
                dR=jets[ijet].DeltaR(jets[jjet])
                dRs.append(dR)
                dPhi=abs(jets[ijet].DeltaPhi(jets[jjet]))
                dPhis.append(dPhi)
                dEta=abs(jets[ijet].Eta()-jets[jjet].Eta())
                dEtas.append(dEta)
        
        # debug printout        
        #print("N_Jets "+str(chain.N_Jets)+" -> combs "+str(len(dRs))+" "+str(dRs))
        # fill histograms with calculated angles
        for dR in dRs:
            hist.Fill(dR)
        for dPhi in dPhis:
            hist_phi.Fill(dPhi)
        for dEta in dEtas:
            hist_eta.Fill(dEta)
        for dPhi, dEta in zip(dPhis,dEtas):
            hist_2D_eta_phi.Fill(dEta,dPhi)
        hist_minDR.Fill(min(dRs))    
            
    # store histograms for later plotting        
    histos.append(hist)
    histos_phi.append(hist_phi)
    histos_eta.append(hist_eta)
    histos_2D_eta_phi.append(hist_2D_eta_phi)
    histos_minDr.append(hist_minDR)

#now plot everything and store the things
# here you can make the plots nicer
# if you want to draw green lines for pixel size similar to the green cirlce in the 2D plots you could use a TLine object. I have not done this here but it should be simple
#canvas.cd()
canvas=ROOT.TCanvas("c","c",800,600)

for ihist, hist in enumerate(histos):
    # normalize to 1 integral 
    if hist.Integral()>0:
      hist.Scale(1./float(hist.Integral()))
    # first histogram sets up axis and later histograms are drawn in same axis 
    if ihist==0:
        hist.Draw("hist")
    else:
        hist.Draw("histsame")

# atumatically creates legend
canvas.BuildLegend()
canvas.SaveAs("dRs.png")

canvas=ROOT.TCanvas("c","c",800,600)
#canvas.cd()
for ihist, hist in enumerate(histos_phi):
    if hist.Integral()>0:
      hist.Scale(1./float(hist.Integral()))
    if ihist==0:
        hist.Draw("hist")
    else:
        hist.Draw("histsame")

canvas.BuildLegend()
canvas.SaveAs("dPhis.png")
                    
canvas=ROOT.TCanvas("c","c",800,600)
for ihist, hist in enumerate(histos_eta):
    if hist.Integral()>0:
      hist.Scale(1./float(hist.Integral()))
    if ihist==0:
        hist.Draw("hist")
    else:
        hist.Draw("histsame")

canvas.BuildLegend()
canvas.SaveAs("dEtas.png")

canvas=ROOT.TCanvas("c","c",800,600)
for ihist, hist in enumerate(histos_minDr):
    if hist.Integral()>0:
      hist.Scale(1./float(hist.Integral()))
    if ihist==0:
        hist.Draw("hist")
    else:
        hist.Draw("histsame")

canvas.BuildLegend()
canvas.SaveAs("minDrs.png")

# now do the 2D histograms                               
pixelsize=0.4                              
for hist in histos_2D_eta_phi:
    canvas=ROOT.TCanvas("c","c",800,600)
    hist.Draw("colz")
    # draw green circle indicating pixel size
    circ=ROOT.TEllipse(0,0,4*pixelsize, 4*pixelsize)
    circ.SetLineColor(ROOT.kSpring)
    circ.SetLineWidth(5)
    circ.SetFillStyle(0)
    circ.Draw()
    
    canvas.SaveAs(hist.GetName()+".png")
    
