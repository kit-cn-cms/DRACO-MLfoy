import ROOT
import os
from math import sin, cos
import numpy as np


pathName="/nfs/dust/cms/user/mwassmer/ttH_2019/ntuples_2018/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8"
pathName2 = "/nfs/dust/cms/user/vdlinden/legacyTTH/ntuples/legacy_2018_ttZ_v2/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8"

chain = ROOT.TChain("MVATree")
print(pathName2)
toadd = os.path.join(pathName2, "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_*00_nominal_Tree.root")
#toadd0 = os.path.join(pathName, "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_*0_nominal_Tree.root")
#toadd1 = os.path.join(pathName, "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_*1_nominal_Tree.root")


print(toadd)
print("\n Adding from " + toadd)
#for no in range(486):
chain.Add(toadd)
#print("\n Adding from " + toadd1)
#chain.Add(toadd1)


#loop over all events
nchain=chain.GetEntries()
print(nchain)
i=0

jet_pt= ROOT.TH1F("jet_pt","transverse momentum", 100,0,1000)
top_mass = ROOT.TH1F("top_mass","mass of top quark", 35, 0 ,700)
delta_rlep = ROOT.TH1F("delta_rlep", "#Delta r lep;#Delta r of leptonic decaying tops; number of events", 80,0,8)
delta_rhad = ROOT.TH1F("delta_rhad", "#Delta r had;#Delta r of hadronic decaying tops; number of events", 80,0,8)

delta_rlep_split = ROOT.TH2F("delta_rlep_split", ";#Delta #Phi of leptonic decaying tops; #Delta #eta", 100,-5,5,100,-10,10)
delta_rhad_split = ROOT.TH2F("delta_rhad_split", ";#Delta #Phi of hadronic decaying tops; #Delta #eta", 100,-5,5,100,-10,10)


delta_r_comb = ROOT.TH1F("delta_r_comb", "#Delta r combined; #Delta r of tops; number of events", 100,0,10)
delta_r_comb_split = ROOT.TH2F("delta_r_comb_split", "#Delta #Phi of tops; #Delta #eta", 100,-5,5,100,-10,10)

delta_phi_lep_stack = ROOT.TH1F("delta_phi_lep_stack", "#Delta #Phi lep; #Delta #Phi of leptonic decaying tops; number of events", 100,0,10)
delta_phi_had_stack = ROOT.TH1F("delta_phi_had_stack", "#Delta #Phi had; #Delta #Phi of hadronic decaying tops; number of events", 100,0,10)

delta_eta_lep_stack = ROOT.TH1F("delta_eta_lep_stack", "#Delta #eta lep; #Delta #eta of leptonic decaying tops; number of events", 100,0,10)
delta_eta_had_stack = ROOT.TH1F("delta_eta_had_stack", "#Delta #eta had; #Delta #eta of hadronic decaying tops; number of events", 100,0,10)

delta_rlep_stack = ROOT.THStack("delta_rlep_stack", "#Delta r lep; #Delta r of leptonic decaying tops; number of events")
delta_rhad_stack = ROOT.THStack("delta_rhad_stack", "#Delta r had; #Delta r of hadronic decaying tops; number of events")


#myreader = TTreeReader(chain)

#mypt=TTreeReader<Float_t>(myreader, "Jet_Pt")
#while(myReader.Next()):
    #myHist.Fill(mypt)
    
#myHist.Draw()
mW=80.4
mWhad=80.0
sigmaW=10.
#mWhad= 80.379
#sigmaW=0.012
mThad = 165.
mTlep = 168.

sigma_thad = 17.
sigma_tlep = 26.

sigmaD=43.
mD=3.
counter=0
bestcombi = np.zeros((3,4))

eventcounter = 0

                

for event in chain:
    njets = event.N_Jets
    if(njets>10):
        eventcounter+=1
        print "gefunden, counter: ", eventcounter, "/n"
    #print(njets)
    i+=1
    jet_pts  = event.Jet_Pt
    jet_etas = event.Jet_Eta
    jet_phis = event.Jet_Phi
    jet_ms   = event.Jet_M
    jet_csv  = event.Jet_CSV
    
    #print(event.Evt_Phi_MET)
    
    
    if(event.N_TightMuons==1):
        lepton4=ROOT.TLorentzVector()
        lepton4.SetPtEtaPhiM(event.Muon_Pt[0], event.Muon_Eta[0], event.Muon_Phi[0], event.Muon_M[0])
    if(event.N_TightElectrons==1):
        lepton4=ROOT.TLorentzVector()
        lepton4.SetPtEtaPhiM(event.Electron_Pt[0], event.Electron_Eta[0], event.Electron_Phi[0], event.Electron_M[0])
    if(i%2000==0):
        print i, '/', nchain, '=', 100.*i/nchain,'%'
        
    minchi=[999999.,999999.,999999.]    
    thadbest0, thadbest1, thadbest2 = ROOT.TLorentzVector(), ROOT.TLorentzVector(), ROOT.TLorentzVector()
    tlepbest0, tlepbest1, tlepbest2 = ROOT.TLorentzVector(), ROOT.TLorentzVector(), ROOT.TLorentzVector()
    #Loop over all combinations of 4 jets
    for j in range(njets):
        #print(jet_pts[i])
        #jet_pt.Fill(jet_pts[j])
        for k in range(njets):
            for l in range(njets):
                for m in range(njets):
                    if(j==k or j==l or j==m or k==l or k==m or l==m):
                        continue
                    if(jet_csv[j]<0.227 or jet_csv[k]<0.227):
                        continue
                    
                    Pt_MET     = event.Evt_MET_Pt
                    Phi_MET    = event.Evt_MET_Phi
                    
                    bhad = ROOT.TLorentzVector()
                    bhad.SetPtEtaPhiM(jet_pts[j],jet_etas[j],jet_phis[j], jet_ms[j])
                    blep = ROOT.TLorentzVector()
                    blep.SetPtEtaPhiM(jet_pts[k],jet_etas[k],jet_phis[k], jet_ms[k])
                    q1 = ROOT.TLorentzVector()
                    q1.SetPtEtaPhiM(jet_pts[l],jet_etas[l],jet_phis[l], jet_ms[l])
                    q2 = ROOT.TLorentzVector()
                    q2.SetPtEtaPhiM(jet_pts[m],jet_etas[m],jet_phis[m], jet_ms[m])
                    
                    #Neutrino-Berechnung
                    neutrino = ROOT.TLorentzVector(Pt_MET*cos(Phi_MET),Pt_MET*sin(Phi_MET),0.,Pt_MET)
                    mu = ((mW*mW)/2) + lepton4.Px()*neutrino.Px()+ lepton4.Py()*neutrino.Py()
                    a = (mu*lepton4.Pz())/(lepton4.Pt()**2)
                    a2 = a**2
                    b = (lepton4.E()**2*neutrino.Pt()**2 - mu**2)/(lepton4.Pt()**2)
                    if(a2<b):
                        neutrino.SetPz(a)
                    else:
                        pz1=a+(a2-b)**0.5
                        pz2=a-(a2-b)**0.5
	
                        if(abs(pz1) <= abs(pz2)):
                            neutrino.SetPz(pz1)
                        else:
                            neutrino.SetPz(pz2)
                            
                            
                            
                    #Rekonstruktionen
                    whad = q1+q2
                    wlep = neutrino + lepton4
                    thad = whad + bhad
                    tlep = wlep + blep
                    
                    #chi2 = (whad.M() - mWhad)**2/sigmaW**2 + (thad.M()-tlep.M()-mD)**2/sigmaD**2
                    chi2 = (whad.M() - mWhad)**2/sigmaW**2 + (thad.M()-mThad)**2/sigma_thad**2 + (tlep.M()-mTlep)/sigma_tlep**2
                    
                    if(chi2<minchi[0]):
                        tlepbest2  = tlepbest1
                        tlepbest1  = tlepbest0
                        tlepbest0  = tlep
                        thadbest2  = thadbest1
                        thadbest1  = thadbest0
                        thadbest0  = thad
                        minchi[2]  = minchi[1]
                        minchi[1]  = minchi[0]
                        minchi[0]  = chi2
                        bestcombi[2][:] = bestcombi[1][:]
                        bestcombi[1][:] = bestcombi[0][:]
                        bestcombi[0][:] = [j,k,l,m]
                            
                    if(chi2>minchi[0] and chi2 < minchi[1]):
                        tlepbest2  = tlepbest1
                        tlepbest1  = tlep
                        thadbest2  = thadbest1
                        thadbest1  = thad
                        minchi[2]  = minchi[1]
                        minchi[1]  = chi2
                        bestcombi[2][:] = bestcombi[1][:]
                        bestcombi[1][:] = [j,k,l,m]
                        
                    if(chi2>minchi[1] and chi2<minchi[2]):
                        tlepbest2  = tlep
                        thadbest2  = thad
                        minchi[2]  = chi2
                        bestcombi[2][:] = [j,k,l,m]
                        
    if(i%1000 == 0):
        print bestcombi, "/n"
    
    mindeltar = np.zeros(3)
    for perm in [2,1,0]:
        tlepbest = globals()["tlepbest"+ str(perm)]
        thadbest = globals()["thadbest"+ str(perm)]
        
        
    #maverage = (tlepbest.M() + thadbest.M())/2.
    #top_mass.Fill(maverage)
    
        delta_phi_lep = (tlepbest.Phi()-event.GenTopLep_Phi[0])
        delta_phi_had = (thadbest.Phi()-event.GenTopHad_Phi[0])
        
        delta_eta_lep = tlepbest.Eta() - event.GenTopLep_Eta[0]
        delta_eta_had = thadbest.Eta() - event.GenTopHad_Eta[0]
        
        #todo: korrigieren der phi-werte, nicht mit modulo sondern Fallunterscheidung nach min(abs(delta_phi),abs(abs(delta_phi) - 2pi))
        if(delta_phi_lep  <=  -np.pi):
            delta_phi_lep += 2*np.pi
        if(delta_phi_lep  >    np.pi):
            delta_phi_lep -= 2*np.pi
        if(delta_phi_had  <=  -np.pi):
            delta_phi_had += 2*np.pi
        if(delta_phi_had  >    np.pi):
            delta_phi_had -= 2*np.pi
            
        deltaRl = ((delta_phi_lep)**2. + (delta_eta_lep)**2.)**0.5
        deltaRh = ((delta_phi_had)**2. + (delta_eta_had)**2.)**0.5
        
        mindeltar[perm] = (deltaRl < 0.4 and deltaRl<0.4)
    
    if(np.sum(mindeltar)>0):
        counter+=1
        
    
    delta_rlep.Fill(deltaRl)
    delta_rhad.Fill(deltaRh)
                    
    
    delta_rlep_split.Fill((delta_phi_lep), (delta_eta_lep))
    delta_rhad_split.Fill((delta_phi_had), (delta_eta_had))
    
    delta_r_comb.Fill(deltaRl)
    delta_r_comb.Fill(deltaRh)
    
    delta_r_comb_split.Fill(delta_phi_lep, (delta_eta_lep))
    delta_r_comb_split.Fill(delta_phi_had, (delta_eta_had))
    
    delta_phi_lep_stack.Fill(abs(delta_phi_lep))
    delta_phi_lep_stack.SetFillColor(ROOT.kBlue)
    delta_phi_had_stack.Fill(abs(delta_phi_had))
    delta_phi_had_stack.SetFillColor(ROOT.kBlue)
    
    delta_eta_lep_stack.Fill(abs(delta_eta_lep))
    delta_eta_lep_stack.SetFillColor(ROOT.kRed)
    delta_eta_had_stack.Fill(abs(delta_eta_had))    
    delta_eta_had_stack.SetFillColor(ROOT.kRed)

    
delta_rlep_stack.Add(delta_phi_lep_stack)
delta_rlep_stack.Add(delta_eta_lep_stack)
delta_rhad_stack.Add(delta_phi_had_stack)
delta_rhad_stack.Add(delta_eta_had_stack)
    
x,y,r = np.zeros(100),np.zeros(100),np.zeros(100)

for i in range(100):
    for j in range(i,100):
        x[j] += delta_rlep.GetBinContent(i)/delta_rlep.Integral()*delta_rlep.GetBinContent(delta_rlep.GetMaximumBin())
        y[j] += delta_rhad.GetBinContent(i)/delta_rhad.Integral()*delta_rhad.GetBinContent(delta_rhad.GetMaximumBin())
##r+=0.1/2
    r[i]=(i+0.5)/10. 
    
#delta_rlep.Scale(1./delta_rlep.GetBinContent(delta_rlep.GetMaximumBin()))
#delta_rhad.Scale(1./delta_rhad.GetBinContent(delta_rhad.GetMaximumBin()))
#print delta_rlep.GetBinContent(delta_rlep.GetMaximumBin())/delta_rlep.Integral()
#x*=delta_rlep.GetBinContent(delta_rlep.GetMaximumBin())/delta_rlep.Integral()
#y*=delta_rhad.GetBinContent(delta_rhad.GetMaximumBin())/delta_rhad.Integral()
##r+=0.1/2

efficiency_lep = ROOT.TGraph(100, r, x)
efficiency_had = ROOT.TGraph(100, r, y)

#delta_rray = np.zeros((100,3))

#for i in range(100):
    #delta_rray[i][0]=i/10.
    #delta_rray[i][1]=delta_rlep.GetBinContent(i)
    #delta_rray[i][2]=delta_rhad.GetBinContent(i)
#np.savetxt('w80_rwerte.txt',delta_rray,delimiter=',')
    
c1=ROOT.TCanvas("c1","delta r and efficiency",1200,500)
c1.Divide(2,1)

#c1.cd(1)
#jet_pt.Draw()

#c1.cd(2)
#top_mass.Draw()
c1.cd(1)

delta_rlep.SetFillColor(ROOT.kCyan-9)
delta_rlep.SetStats(0)
delta_rlep.Draw("HIST")

efficiency_lep.SetLineColor(ROOT.kRed)
efficiency_lep.SetLineWidth(2)
efficiency_lep.Draw("SAME")

eff_lep = str(int(10000*x[3]/delta_rlep.GetBinContent(delta_rlep.GetMaximumBin())))
eff_had = str(int(10000*y[3]/delta_rhad.GetBinContent(delta_rhad.GetMaximumBin())))
eff_comb= str(int(10000.*counter/nchain))

line_lep = ROOT.TLine(0.4,0,0.4,delta_rlep.GetMaximum())
line_lep.SetLineColor(ROOT.kBlack)
line_lep.SetLineWidth(2)
line_lep.Draw()

axis_lep = ROOT.TGaxis(8,0,8,delta_rlep.GetBinContent(delta_rlep.GetMaximumBin()),0,1)
axis_lep.SetTitle("efficiency")
axis_lep.Draw()


#buffer1 = str
buffer1 = "efficiency at #Delta r < 0.4: " + eff_lep[:2] + "," + eff_lep[2:] + "%"
text1 = ROOT.TLatex()
text1.DrawLatex(2,0.4*delta_rlep.GetMaximum(),buffer1)


leg_lep = ROOT.TLegend(0.5,0.5,0.9,0.65)
leg_lep.AddEntry(delta_rlep, "histrogram filled with #Delta r of tleps","f")
leg_lep.AddEntry(efficiency_lep, "efficiency","l")
leg_lep.AddEntry(line_lep, "#Delta r = 0.4", "l")
leg_lep.Draw()


c1.cd(2)
#delta_rhad.Draw("HIST")
#efficiency_had.Draw("SAME")
#efficiency_had.Draw("*","SAME")
#c1.cd(3)
#delta_rlep_split.Draw()
#c1.cd(4)
#delta_rhad_split.Draw()
#c2.SetGrid(2,1)
delta_rhad.SetFillColor(ROOT.kCyan-9)
delta_rhad.SetStats(0)
delta_rhad.Draw("HIST")

efficiency_had.SetLineColor(ROOT.kRed)
efficiency_had.SetLineWidth(2)
efficiency_had.Draw("SAME")

line_had = ROOT.TLine(0.4,0,0.4,delta_rhad.GetMaximum())
line_had.SetLineColor(ROOT.kBlack)
line_had.SetLineWidth(2)
line_had.Draw()

axis_had = ROOT.TGaxis(8,0,8,delta_rhad.GetBinContent(delta_rhad.GetMaximumBin()),0,1)
axis_had.SetTitle("efficiency")
axis_had.Draw("SAME")

buffer2 = "efficiency at #Delta r < 0.4: " + eff_had[:2]+","+eff_had[2:] + "%"
text2 = ROOT.TLatex()
text2.DrawLatex(2,0.4*delta_rhad.GetMaximum(),buffer2)
text3 = ROOT.TLatex()
text3.DrawLatex(2,0.3*delta_rhad.GetMaximum(), "combined efficiency:  " + eff_comb[:2] + "," + eff_comb[2:] + "%")

leg_had = ROOT.TLegend(0.5,0.5,0.9,0.65)
leg_had.AddEntry(delta_rhad, "histrogram filled with #Delta r of thads","f")
leg_had.AddEntry(efficiency_had, "efficiency","l")
leg_had.AddEntry(line_had, "#Delta r = 0.4", "l")
leg_had.Draw()

#print "Efficiency for hadronic tops at #Delta r = 0.4: ", y[4]/delta_rhad.GetBinContent(delta_rhad.GetMaximumBin())
#print "Combined Efficiency at #Delta r < 0.4: ", counter*1./nchain


c1.SaveAs("/nfs/dust/cms/user/jdriesch/results/testtesttest_1pc_efficiency.png")
#Reconstruction of Muon    
#ROOT.TLorentzVector::SetPtEtaPhiM(pt,eta, phi, m)
#c2 = ROOT.TCanvas("c2","tlep and thad combined", 1000,500)
#c2.Divide(2,1)
#c2.cd(1)
#delta_r_comb.Draw()
#c2.cd(2)
#delta_r_comb_split.Draw()
#c2.SaveAs("results/deltar_w80_10pc_btag_nocut_comb_phi1.png")

#stacked plots
#c3 = ROOT.TCanvas("c3", "stacked plot of #Delta #Phi and #Delta #eta", 1200,600)
#c3.Divide(2,1)
#c3.cd(1)
#delta_rlep_stack.Draw()
#c3.cd(2)
#delta_rhad_stack.Draw()
#c3.SaveAs("results/deltar_w80_10pc_btag_nocut_stacked.png")