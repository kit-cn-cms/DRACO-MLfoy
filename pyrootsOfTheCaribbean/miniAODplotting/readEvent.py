import sys
import ROOT
import numpy as np
# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so");
ROOT.gSystem.Load("libDataFormatsFWLite.so");
ROOT.FWLiteEnabler.enable()

# load FWlite python libraries
from DataFormats.FWLite import Handle, Events

muons,          muonLabel           = Handle("std::vector<pat::Muon>"),         "slimmedMuons"
electrons,      electronLabel       = Handle("std::vector<pat::Electron>"),     "slimmedElectrons"
photons,        photonLabel         = Handle("std::vector<pat::Photon>"),       "slimmedPhotons"
taus,           tauLabel            = Handle("std::vector<pat::Tau>"),          "slimmedTaus"
jets,           jetLabel            = Handle("std::vector<pat::Jet>"),          "slimmedJets"
fatjets,        fatjetLabel         = Handle("std::vector<pat::Jet>"),          "slimmedJetsAK8"
mets,           metLabel            = Handle("std::vector<pat::MET>"),          "slimmedMETs"
vertices,       vertexLabel         = Handle("std::vector<reco::Vertex>"),      "offlineSlimmedPrimaryVertices"
verticesScore                       = Handle("edm::ValueMap<float>")
genJets,        genJetLabel         = Handle("std::vector<reco::GenJet>"),      "slimmedGenJets"
genParticles,   genParticleLabel    = Handle("std::vector<reco::GenParticle>"), "prunedGenParticles"
genInfo,        genInfoLabel        = Handle("GenEventInfoProduct"),            "generator"


class Event:
    def __init__(self, event, XSWeight = 1., event_type = "ttH"):
        # initialize variables
        self.variables = {}

        # save indices for event
        self.variables["Evt_Run"]    = event.eventAuxiliary().run()
        self.variables["Evt_Lumi"]   = event.eventAuxiliary().luminosityBlock()
        self.variables["Evt_ID"]     = event.eventAuxiliary().event()
        
        # save weights
        self.variables["Weight_XS"]  = XSWeight
        event.getByLabel(genInfoLabel, genInfo)
        self.variables["Weight_GEN_nom"] = genInfo.product().weight()

        # save event type
        self.event_type = event_type

        # initialize objects
        self.objects = {}
        self.ERROR = False    

    def getJetTagInfo(self, event):
        nJets = 0
        nTags = 0

        event.getByLabel(jetLabel, jets)
        for i,j in enumerate(jets.product()):
            if j.pt() < 20. or j.eta() > 2.4: continue
            nJets += 1
            if j.bDiscriminator("pfDeepCSVJetTags:probb")+j.bDiscriminator("pfDeepCSVJetTags:probbb") > 0.48:
                nTags += 1
        
        self.variables["N_Jets"] = nJets
        self.variables["N_BTagsM"] = nTags

    def getAdditionalJets(self, event):
        # TODO
        nJets = 0
        event.getByLabel(genJetLabel, genJets)
        for i,j in enumerate(genJets.product()):
            if j.pt() < 20. or j.eta() > 2.4: continue
            mothers = [j.mother(i2) for i2 in xrange(j.numberOfMothers())]
            if len(mothers) != 1: print("not ==1 mother particle")
            isAddJet = True
            for m in mothers:
                if m.pdgId() in [-6,6,25,24,-24,23]:
                    isAddJet = False
            if isAddJet: nJets += 1
        self.variables["N_AddGenJets"] = nJets

    def getSLTTSystem(self, event):
        event.getByLabel(genParticleLabel, genParticles)

        # search for top and anti top quark
        top     = find_particle(genParticles, id = 6)
        topbar  = find_particle(genParticles, id = -6)

        # get their daughters
        top_decays      = get_daughters(top)
        topbar_decays   = get_daughters(topbar)

        # get b quarks from top decays
        try:
            top_b       = find_particle(top_decays, id = 5)
            topbar_bbar = find_particle(topbar_decays, id = -5)
        except:
            self.ERROR = True
            return

        # get w bosons from top decays
        try:
            top_W       = find_particle(top_decays, id = 24)
            topbar_W    = find_particle(topbar_decays, id = -24)
        except:
            self.ERROR = True
            return

        # match leptonic and hadronic decay
        if is_leptonic_W(top_W):
            self.objects["Lepton"] = get_lepton(top_W)
            self.objects["hadTop"] = topbar 
            self.objects["lepTop"] = top
            self.objects["hadB"]   = topbar_bbar
            self.objects["lepB"]   = top_b
        elif is_leptonic_W(topbar_W):
            self.objects["Lepton"] = get_lepton(top_W)
            self.objects["hadTop"] = top
            self.objects["lepTop"] = topbar
            self.objects["hadB"]   = top_b
            self.objects["lepB"]   = topbar_bbar
        else:
            self.ERROR = True
            return


    def getHiggsBBSystem(self, event):
        event.getByLabel(genParticleLabel, genParticles)
        
        higgs = find_particle(genParticles, id = 25)
        
        self.objects["Higgs"] = higgs
        self.objects["Boson"] = higgs

        if is_bb_decay(higgs):
            higgs_decays = get_daughters(higgs)
            higgs_b1 = find_particle(higgs_decays, id = 5)
            higgs_b2 = find_particle(higgs_decays, id = -5)
        
            self.objects["HiggsB1"] = higgs_b1
            self.objects["BosonB1"] = higgs_b1
            self.objects["HiggsB2"] = higgs_b2
            self.objects["BosonB2"] = higgs_b2
        else:
            self.error = True
            return

        
    def getBoson(self, event, bosonID):
        event.getByLabel(genParticleLabel, genParticles)

        try:
            boson = find_particle(genParticles, id = bosonID)
            self.objects["Boson"] = boson
        except:
            self.ERROR = True
            return

    def getZBBSystem(self, event):
        event.getByLabel(genParticleLabel, genParticles)
        
        Z = find_particle(genParticles, id = 23)
        
        self.objects["ZBoson"] = Z
        self.objects["Boson"] = Z

        if is_bb_decay(Z):
            Z_decays = get_daughters(Z)
            Z_b1 = find_particle(Z_decays, id = 5)
            Z_b2 = find_particle(Z_decays, id = -5)
        
            self.objects["ZBosonB1"] = Z_b1
            self.objects["BosonB1"] = Z_b1
            self.objects["ZBosonB2"] = Z_b2
            self.objects["BosonB2"] = Z_b2
        else:
            self.ERROR = True
            return


    def passesCuts(self):
        if self.ERROR: return False
        for key in self.objects:
            obj = self.objects[key]
            if obj == None: return False
            if np.abs(obj.eta()) > 10: return False
            if np.abs(obj.phi()) > np.pi: return False

        return True
        




    ## variable Calulations
    def get_dEta(self, obj1, obj2):
        return np.abs(self.objects[obj1].eta() - self.objects[obj2].eta())

    def get_dPhi(self, obj1, obj2):
        dphi = np.abs(self.objects[obj1].phi() - self.objects[obj2].phi())
        if dphi > np.pi: dphi = 2.*np.pi - dphi
        if dphi > np.pi: print("but why")
        return dphi
    
    def get_dY(self, obj1, obj2):
        return np.abs(self.objects[obj1].y() - self.objects[obj2].y())

    def get_dTheta(self, obj1, obj2):
        return np.abs(self.objects[obj1].theta() - self.objects[obj2].theta())

    def get_dcosTheta(self, obj1, obj2):
        return np.abs(np.cos(self.objects[obj1].theta()) - np.cos(self.objects[obj2].theta()))

    def get_dR(self, obj1, obj2):
        return np.sqrt( self.get_dEta(obj1, obj2)**2 + self.get_dPhi(obj1, obj2)**2 )

    def get_pT(self, obj):
        return self.objects[obj].pt()

    def get_eta(self, obj):
        return self.objects[obj].eta()

    def get_phi(self, obj):
        return self.objects[obj].phi()

    def get_mass(self,obj):
        return self.objects[obj].mass()

    def get_y(self,obj):
        return self.objects[obj].y()
        
    def get_theta(self,obj):
        return self.objects[obj].theta()

    def get_costheta(self,obj):
        return np.cos(self.objects[obj].theta())







def readEvent(iev, event, XSWeight = 1., event_type = "ttH"):
    ''' read information of a single event '''
    if not event_type in ["ttH", "ttHbb", "ttZ", "ttZll", "ttZqq"]:
        print("special cases for event_type {} have not yet been implemented".format(event_type))

    # intialize event class
    evt = Event(event, XSWeight, event_type)

    # read jet tag
    evt.getJetTagInfo(event) 
    if evt.ERROR: return evt

    #evt.getAdditionalJets(event)
    #if evt.ERROR: return evt

    # read semi leptonic ttbar system
    evt.getSLTTSystem(event)
    if evt.ERROR: return evt

    ## read higgs system
    if event_type in ["ttH","ttHbb"]:
        evt.getBoson(event, bosonID = 25)
        #evt.getHiggs(event)
        #evt.getHiggsBBSystem(event)
    # read Z system
    if event_type in ["ttZ", "ttZll", "ttZqq"]:
        evt.getBoson(event, bosonID = 23)
        #evt.getZ(event)

    #if event_type in ["ttZqq","ttZ"]:
    #    evt.getZBBSystem(event)
    #if evt.ERROR: return evt

    

    return evt

    





# particle searching functions
def find_particle(particles, id):
    ''' find particle candidate via id
        returns found candidate '''
    try:    enum = enumerate(particles.product())
    except: enum = enumerate(particles)
    for i, p in enum:
        if p.pdgId() == id:
            candidate = p
            break

    while candidate.numberOfDaughters() == 1 and candidate.daughter(0).pdgId() == id:
        candidate = candidate.daughter(0)
    return candidate

def get_daughters(obj):
    ''' get list of daughter particles of object
        only return particles that are not intermediate particles '''
    list_of_daughters = [obj.daughter(di) for di in xrange(obj.numberOfDaughters())]
    return list_of_daughters

def is_leptonic_W(w):
    w_daughters = get_daughters(w)
    for d in w_daughters:
        if d.pdgId() in [11,-11,13,-13]: return True
    return False

def get_lepton(w):
    w_daughters = get_daughters(w)
    for d in w_daughters:
        if d.pdgId() in [11,-11,13,-13]: return d

def is_bb_decay(boson):
    boson_daughters = get_daughters(boson)
    n_found = 0
    for d in boson_daughters:
        if d.pdgId() in [-5,5]:
            n_found += 1
        if n_found >= 2: return True
    return False

