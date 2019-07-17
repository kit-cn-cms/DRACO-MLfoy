import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
from math import sin, cos, log

import os
import sys
import numpy as np
import pandas as pd
import json
import keras
# import class for DNN training
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)
import DRACO_Frameworks.DNN.DNN as DNN
import DRACO_Frameworks.DNN.data_frame as data_frame

#################################################################################################################################

def loadDNN(inputDirectory, outputDirectory):

    # get net config json
    configFile = inputDirectory+"/checkpoints/net_config.json"
    if not os.path.exists(configFile):
        sys.exit("config needed to load trained DNN not found\n{}".format(configFile))

    with open(configFile) as f:
        config = f.read()
    config = json.loads(config)

    # load samples
    input_samples = data_frame.InputSamples(config["inputData"])

    for sample in config["eventClasses"]:
        input_samples.addSample(sample["samplePath"], sample["sampleLabel"], normalization_weight = sample["sampleWeight"])

    print("shuffle seed: {}".format(config["shuffleSeed"]))
    # init DNN class
    dnn = DNN.DNN(
        save_path       = outputDirectory,
        input_samples   = input_samples,
        event_category  = config["JetTagCategory"],
        train_variables = config["trainVariables"],
        shuffle_seed    = config["shuffleSeed"]
        )

    # load the trained model
    # dnn.load_trained_model(inputDirectory)

    checkpoint_path = inputDirectory+"/checkpoints/trained_model.h5py"

    # get the model
    dnn.model = keras.models.load_model(checkpoint_path)
    dnn.model.summary()

    return dnn


def findttbar(dataframe, inputDirectory):
    # evaluate test dataset
    # model_eval = dnn.model.evaluate(dataframe.values, dnn.data.get_test_labels())

    # save predicitons
    ones_counter=0
    model_predict = dnn.model.predict(dataframe.values)
    max1 = -10
    best_index=0
    for ind in range(len(model_predict)):
        if model_predict[ind]==1.0:
            ones_counter +=1
        if model_predict[ind] > max1:
            max1 = model_predict[ind]
            best_index = ind

    if(max1<-1): print "error!!11!!1!1!"
    # print best_index, (max1), model_predict, len(model_predict)
    # np.set_printoptions(suppress=True)
    # np.set_printoptions(precision=75)
    # print best_index, np.round(max1,20), len(model_predict)
    # print np.array(model_predict)
    # if ones_counter>1:
    #     print "warning: more than one index maximal"
    #     best_index = -1
    return best_index


def normalize(df,inputdir):
    unnormed_df = df

    df_norms = pd.read_csv(inputdir+"/checkpoints/variable_norm.csv", index_col=0).transpose()
    for ind in df.columns:
        df[ind] = (unnormed_df[ind] - df_norms[ind][0])/df_norms[ind][1]
    return df


def correct_phi(phi):
    if(phi  <=  -np.pi):
        phi += 2*np.pi
    if(phi  >    np.pi):
        phi -= 2*np.pi
    return phi
###################################################################################################################################


# pathName = "/nfs/dust/cms/user/vdlinden/legacyTTH/ntuples/legacy_2018_ttZ_v2/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8"
# ntuplefile = os.path.join(pathName, "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_101_nominal_Tree.root")
# #print ntuplefile
# # open roofile
# f = ROOT.TFile(ntuplefile)
# # get tree
# tree = f.Get("MVATree")
# # loop over events in tree
# var1 = np.array([])
# entries=0
# for i in xrange(tree.GetEntries()):
#     entries += tree.GetEntry(i)
#     # read out random variables
#     var1 = np.append(var1,tree.GetLeaf("Jet_CSV").GetValue(0))
# print entries

pepec = ["Pt", "Eta", "Phi", "E", "CSV"]
jets  = ["TopHad_B", "TopLep_B", "TopHad_Q1", "TopHad_Q2"]

print "\n  done part 1  \n"




def getTopMass(Pt, Eta, Phi, E, lepton_4vec, Pt_MET, Phi_MET,Ht, j,k,l,m):
    # Rekonstruktion der top quarks:
    mW = 80.4

    #Neutrino-Berechnung
    neutrino_4vec = ROOT.TLorentzVector(Pt_MET*cos(Phi_MET),Pt_MET*sin(Phi_MET),0.,Pt_MET)
    mu = ((mW*mW)/2) + lepton_4vec.Px()*neutrino_4vec.Px()+ lepton_4vec.Py()*neutrino_4vec.Py()
    a = (mu*lepton_4vec.Pz())/(lepton_4vec.Pt()**2)
    a2 = a**2
    b = (lepton_4vec.E()**2*neutrino_4vec.Pt()**2 - mu**2)/(lepton_4vec.Pt()**2)
    if(a2<b):
        neutrino_4vec.SetPz(a)
    else:
        pz1=a+(a2-b)**0.5
        pz2=a-(a2-b)**0.5

        if(abs(pz1) <= abs(pz2)):
            neutrino_4vec.SetPz(pz1)
        else:
            neutrino_4vec.SetPz(pz2)

    neutrino_4vec.SetE(neutrino_4vec.P())

    combi = [j,k,l,m]
    ind = 0
    for index in ["TopHad_B", "TopLep_B", "TopHad_Q1", "TopHad_Q2"]:
        globals()[index + "_4vec"] = ROOT.TLorentzVector()
        globals()[index + "_4vec"].SetPtEtaPhiE(Pt[combi[ind]],Eta[(combi[ind])],Phi[(combi[ind])],E[(combi[ind])])
        ind+=1

    #reconstructions
    whad_4vec = TopHad_Q1_4vec + TopHad_Q2_4vec
    wlep_4vec = neutrino_4vec + lepton_4vec
    thad_4vec = whad_4vec + TopHad_B_4vec
    tlep_4vec = wlep_4vec + TopLep_B_4vec

    return thad_4vec, tlep_4vec, whad_4vec, wlep_4vec, lepton_4vec
################################################################################################################################
pathName2 = "/nfs/dust/cms/user/vdlinden/legacyTTH/ntuples/legacy_2018_ttZ_v2/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8"

chain = ROOT.TChain("MVATree")
print(pathName2)
toadd = os.path.join(pathName2, "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_*_nominal_Tree.root")
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
mW = 80.4
counter = 0
lcounter = 0
rcounter = 0
eventcounter= 0

inputdir = "/nfs/dust/cms/user/jdriesch/draco/DRACO-MLfoy/workdir/allvar_all_v1_ge4j_ge3t"

dnn = loadDNN(inputdir, "output")

top_mass = ROOT.TH1F("top_mass","mass of top quark", 35, 0 ,700)
delta_rlep = ROOT.TH1F("delta_rlep", "#Delta r lep;#Delta r of leptonic decaying tops; number of events", 100,0,8)
delta_rhad = ROOT.TH1F("delta_rhad", "#Delta r had;#Delta r of hadronic decaying tops; number of events", 100,0,8)
delta_r_bhad_hist = ROOT.TH1F("delta_r_bhad_hist", "#Delta r b had;#Delta r of hadronic b; number of events", 100,0,6)
delta_r_blep_hist = ROOT.TH1F("delta_r_blep_hist", "#Delta r b lep;#Delta r of leptonic b; number of events", 100,0,6)
delta_r_q1_hist = ROOT.TH1F("delta_r_q1_hist", "#Delta r q1;#Delta r of q1; number of events", 100,0,6)
delta_r_q2_hist = ROOT.TH1F("delta_r_q2_hist", "#Delta r q2;#Delta r of q2; number of events", 100,0,6)

delta_phi = ROOT.TH1F("delta_phi", "#Delta #Phi b had;#Delta #Phi of hadronic b; number of events", 100,0,6)
delta_eta = ROOT.TH1F("delta_eta", "#Delta #eta b had;#Delta #eta of hadronic b; number of events", 100,0,6)



for event in chain:
    njets = event.N_Jets
    # print njets
    # if(njets>12 or event.N_BTagsM<3 or event.Evt_MET_Pt < 20. or event.Weight_GEN_nom < 0. or (event.N_TightMuons==1 and event.Muon_Pt < 29.) or event.Evt_Odd ==1):
    if(njets>12 or event.N_BTagsM<3 or event.Evt_Odd ==1):

        # print "ok cool next one"
        continue

    for index in jets:
        for index2 in pepec:
            globals()[index + "_" + index2] = np.array([])

    #print(njets)
    i+=1
    Pt  = event.Jet_Pt
    Eta = event.Jet_Eta
    Phi = event.Jet_Phi
    E   = event.Jet_E
    CSV  = event.Jet_CSV
    Pt_MET = event.Evt_MET_Pt
    Phi_MET= event.Evt_MET_Phi
    Ht     = event.Evt_HT

    if event.N_TightMuons:
        Muon_Pt = event.Muon_Pt[0]
        Muon_Eta= event.Muon_Eta[0]
        Muon_Phi= event.Muon_Phi[0]
        Muon_E  = event.Muon_E[0]
        Electron_Pt = 0
        Electron_Eta= 0
        Electron_Phi= 0
        Electron_E  = 0
        lepton4     = ROOT.TLorentzVector()
        lepton4.SetPtEtaPhiE(Muon_Pt, Muon_Eta, Muon_Phi, Muon_E)

    if event.N_TightElectrons:
        Muon_Pt = 0
        Muon_Eta= 0
        Muon_Phi= 0
        Muon_E  = 0
        Electron_Pt = event.Electron_Pt[0]
        Electron_Eta= event.Electron_Eta[0]
        Electron_Phi= event.Electron_Phi[0]
        Electron_E  = event.Electron_E[0]
        lepton4     = ROOT.TLorentzVector()
        lepton4.SetPtEtaPhiE(Electron_Pt, Electron_Eta, Electron_Phi, Electron_E)



    #create df for event
    df = pd.DataFrame()
    #print(event.Evt_Phi_MET)

    # if i==5: break
    #reco_TopHad_M = np.array([])
    #reco_TopLep_M = np.array([])
    #reco_WHad_M   = np.array([])
    for index in ["TopHad","TopLep", "WHad", "WLep"]:
        for index2 in ["Pt", "Eta", "Phi", "M", "logM"]:
	    globals()["reco_" + index + "_" + index2] = np.array([])
    Pt_div_Ht     = np.array([])


    for j in range(njets):
        #print(jet_pts[i])
        #jet_pt.Fill(jet_pts[j])
        for k in range(njets):
            for l in range(njets):
                for m in range(njets):
                    if(j==k or j==l or j==m or k==l or k==m or l==m):
                        continue
                    if(CSV[j]<0.227 or CSV[k]<0.227):
                        continue

                    #filling arrays with different combinations
                    for index2 in pepec:
                        globals()["TopHad_B" + "_" + index2] = np.append(globals()["TopHad_B" + "_" + index2], globals()[index2][j])
                        globals()["TopLep_B" + "_" + index2] = np.append(globals()["TopLep_B" + "_" + index2], globals()[index2][k])
                        globals()["TopHad_Q1" + "_" + index2] = np.append(globals()["TopHad_Q1" + "_" + index2], globals()[index2][l])
                        globals()["TopHad_Q2" + "_" + index2] = np.append(globals()["TopHad_Q2" + "_" + index2], globals()[index2][m])


                    reco_TopHad_4vec, reco_TopLep_4vec, reco_WHad_4vec, reco_WLep_4vec, lepton_4vec  = getTopMass(Pt, Eta, Phi, E, lepton4, Pt_MET, Phi_MET, Ht, j,k,l,m)
                    #reco_TopHad_M = np.append(reco_TopHad_M, reco_tophad_4vec.M())
                    #reco_TopLep_M = np.append(reco_TopLep_M, reco_toplep_4vec.M())
                    #reco_WHad_M   = np.append(reco_WHad_M,   reco_whad_4vec.M())

		    for index in ["TopHad","TopLep", "WHad","WLep"]:
  	                globals()["reco_" + index +"_Pt"] = np.append(globals()["reco_" + index +"_Pt"], locals()["reco_" + index + "_4vec"].Pt())
       		        globals()["reco_" + index + "_Eta"] = np.append(globals()["reco_" + index +"_Eta"], locals()["reco_" + index + "_4vec"].Eta())
      		        globals()["reco_" + index + "_Phi"] = np.append(globals()["reco_" + index +"_Phi"], locals()["reco_" + index + "_4vec"].Phi())
       		        globals()["reco_" + index + "_M"] = np.append(globals()["reco_" + index +"_M"], locals()["reco_" + index + "_4vec"].M())
     		        globals()["reco_" + index + "_logM"] = np.append(globals()["reco_" + index +"_logM"],log( locals()["reco_" + index + "_4vec"].M()))


                    Pt_div_Ht     = np.append(Pt_div_Ht,     (reco_TopHad_4vec.Pt() + reco_TopLep_4vec.Pt())/(Ht + Pt_MET + lepton_4vec.Pt()))

    eventcounter +=1
    n=0
    for index in jets:
        for index2 in pepec:
            n+=1
            df[index +"_"+ index2] = globals()[index + "_" + index2]

    combis = len(TopHad_B_Phi)
    df["Muon_Pt[0]"]   = np.zeros(combis) + Muon_Pt
    df["Muon_Eta[0]"]  = np.zeros(combis) + Muon_Eta
    df["Muon_Phi[0]"]  = np.zeros(combis) + Muon_Phi
    df["Muon_E[0]"]    = np.zeros(combis) + Muon_E
    df["Electron_Pt[0]"] = np.zeros(combis) + Electron_Pt
    df["Electron_Eta[0]"] = np.zeros(combis) + Electron_Eta
    df["Electron_Phi[0]"] = np.zeros(combis) + Electron_Phi
    df["Electron_E[0]"] = np.zeros(combis) + Electron_E
    df["Evt_MET_Pt"]    = np.zeros(combis) + event.Evt_MET_Pt
    df["Evt_MET_Phi"]   = np.zeros(combis) + event.Evt_MET_Phi


    #df["reco_TopHad_M"] = reco_TopHad_M
    #df["reco_TopLep_M"] = reco_TopLep_M
    #df["reco_WHad_M"]   = reco_WHad_M


    vars_toadd = ["TopHad","TopLep", "WHad"]

    for index in vars_toadd:
        for index2 in ["Pt","Eta","Phi","M","logM"]:
	    df["reco_" + index + "_" + index2] = globals()["reco_" + index + "_" + index2]

    df["ttbar_pt_div_ht_p_met"]     = Pt_div_Ht


    df_normed = normalize(df,inputdir)

    # print df, df_normed
    best_index = findttbar(df_normed,dnn)

    if best_index < 0:
        continue




    for index in jets:
        globals()[index] = ROOT.TLorentzVector()
        globals()[index].SetPtEtaPhiE(globals()[index + "_Pt"][best_index],globals()[index + "_Eta"][best_index],globals()[index + "_Phi"][best_index],globals()[index + "_E"][best_index])


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

    neutrino.SetE(neutrino.P())

    #Rekonstruktionen
    whad = TopHad_Q1 + TopHad_Q2
    wlep = neutrino + lepton4
    thad = whad + TopHad_B
    tlep = wlep + TopLep_B

    maverage = (tlep.M() + thad.M())/2.
    # print maverage
    top_mass.Fill(maverage)

    delta_phi_lep = (tlep.Phi()-event.GenTopLep_Phi[0])
    delta_phi_had = (thad.Phi()-event.GenTopHad_Phi[0])

    delta_eta_lep = tlep.Eta() - event.GenTopLep_Eta[0]
    delta_eta_had = thad.Eta() - event.GenTopHad_Eta[0]

    if(delta_phi_lep  <=  -np.pi):
        delta_phi_lep += 2*np.pi
    if(delta_phi_lep  >    np.pi):
        delta_phi_lep -= 2*np.pi
    if(delta_phi_had  <=  -np.pi):
        delta_phi_had += 2*np.pi
    if(delta_phi_had  >    np.pi):
        delta_phi_had -= 2*np.pi

    deltaRl = ((correct_phi(delta_phi_lep))**2. + (delta_eta_lep)**2.)**0.5
    deltaRh = ((correct_phi(delta_phi_had))**2. + (delta_eta_had)**2.)**0.5

    if(deltaRl<0.4 and deltaRh<0.4):
        counter+=1

    if(deltaRl<0.4):
        lcounter +=1

    if(deltaRh<0.4):
        rcounter +=1

    if(i%1000==0):
        print i, '/', nchain/2., '=', 200.*i/nchain,'%', "      , eff: ", 1.*counter/eventcounter

    delta_rlep.Fill(deltaRl)
    delta_rhad.Fill(deltaRh)


    delta_phi_bhad = TopHad_B_Phi[best_index] - event.GenTopHad_B_Phi[0]
    delta_phi_blep = TopLep_B_Phi[best_index] - event.GenTopLep_B_Phi[0]
    delta_phi_q1 = TopHad_Q1_Phi[best_index] - event.GenTopHad_Q1_Phi[0]
    delta_phi_q2 = TopHad_Q2_Phi[best_index] - event.GenTopHad_Q2_Phi[0]

    delta_eta_bhad= TopHad_B_Eta[best_index] - event.GenTopHad_B_Eta[0]
    delta_eta_blep= TopLep_B_Eta[best_index] - event.GenTopLep_B_Eta[0]
    delta_eta_q1= TopHad_Q1_Eta[best_index] - event.GenTopHad_Q1_Eta[0]
    delta_eta_q2= TopHad_Q2_Eta[best_index] - event.GenTopHad_Q2_Eta[0]

    delta_r_bhad = ((correct_phi(delta_phi_bhad))**2 + (delta_eta_bhad)**2)**0.5
    delta_r_blep = ((correct_phi(delta_phi_blep))**2 + (delta_eta_blep)**2)**0.5
    delta_r_q1 = ((correct_phi(delta_phi_q1))**2 + (delta_eta_q1)**2)**0.5
    delta_r_q2 = ((correct_phi(delta_phi_q2))**2 + (delta_eta_q2)**2)**0.5


    #falls q1 und q2 falsch herum zugeordnet sind:
    delta_r_q12 = ((correct_phi(TopHad_Q1_Phi[best_index] - event.GenTopHad_Q2_Phi))**2 + (TopHad_Q1_Eta[best_index] - event.GenTopHad_Q2_Eta)**2)**0.5
    delta_r_q21 = ((correct_phi(TopHad_Q2_Phi[best_index] - event.GenTopHad_Q1_Phi))**2 + (TopHad_Q2_Eta[best_index] - event.GenTopHad_Q1_Eta)**2)**0.5

    if(delta_r_q1+delta_r_q2 > delta_r_q12 + delta_r_q21):
        delta_r_q1 = delta_r_q12
        delta_r_q2 = delta_r_q21

    delta_r_bhad_hist.Fill(delta_r_bhad)
    delta_r_blep_hist.Fill(delta_r_blep)
    delta_r_q1_hist.Fill(delta_r_q1)
    delta_r_q2_hist.Fill(delta_r_q2)


    delta_phi.Fill(correct_phi(delta_phi_bhad))
    delta_eta.Fill(abs(delta_eta_bhad))



print "eff: ", 1.*counter/eventcounter, "  ", 1.*lcounter/eventcounter, "  ", 1.*rcounter/eventcounter

x,y,r = np.zeros(100),np.zeros(100),np.zeros(100)

for i in range(100):
    for j in range(i,100):
        x[j] += delta_rlep.GetBinContent(i)/delta_rlep.Integral()*delta_rlep.GetBinContent(delta_rlep.GetMaximumBin())
        y[j] += delta_rhad.GetBinContent(i)/delta_rhad.Integral()*delta_rhad.GetBinContent(delta_rhad.GetMaximumBin())
##r+=0.1/2
    r[i]=(i-0.5)/10.

efficiency_lep = ROOT.TGraph(100, r, x)
efficiency_had = ROOT.TGraph(100, r, y)


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

eff_lep = str(int(10000*x[4]/delta_rlep.GetBinContent(delta_rlep.GetMaximumBin())))
eff_had = str(int(10000*y[4]/delta_rhad.GetBinContent(delta_rhad.GetMaximumBin())))
eff_comb= str(int(10000.*counter/eventcounter))

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
# delta_rhad.Draw("HIST")
# efficiency_had.Draw("SAME")
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
c1.SaveAs("/nfs/dust/cms/user/jdriesch/results/results/allvar_all_v1_10_100p.pdf")

#######################################################################################################

c2 = ROOT.TCanvas("c1", "delta r", 1200,1200)
c2.Divide(2,2)
c2.cd(1)
delta_r_bhad_hist.SetFillColor(ROOT.kCyan-9)
delta_r_bhad_hist.Draw("HIST")

c2.cd(2)
delta_r_blep_hist.SetFillColor(ROOT.kCyan-9)
delta_r_blep_hist.Draw("HIST")

c2.cd(3)
delta_r_q1_hist.SetFillColor(ROOT.kCyan-9)
delta_r_q1_hist.Draw("HIST")

c2.cd(4)
delta_r_q2_hist.SetFillColor(ROOT.kCyan-9)
delta_r_q2_hist.Draw("HIST")

# c1.cd(1)
# delta_phi.Draw("HIST")
#
# c1.cd(2)
# delta_eta.Draw("HIST")

c2.SaveAs("/nfs/dust/cms/user/jdriesch/results/results/allvar_all_v1_11_100p.pdf")
