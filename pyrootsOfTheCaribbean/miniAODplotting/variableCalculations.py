# import ROOT in batch mode
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
gen_jets,       gen_jetLabel        = Handle("std::vector<reco::GenJet>"),      "ak4GenJets"#"slimmedGenJets"
genParticles,   genParticle_Label   = Handle("std::vector<reco::GenParticle>"), "prunedGenParticles"


class Candidate:
    ''' container for some particle candidate information '''
    def __init__(self, obj, type = "unknown"):
        self.type   = type
        self.obj    = obj
        #self.ID     = obj.pdgId()

def readEvent(iev, event, verbosity = 0):
    ''' read particle-flow level event data
        extract all particle candidates '''

    # list of objects
    candidates = []

    # get muons, electrons, photons and taus from event
    event.getByLabel(muonLabel, muons) 
    event.getByLabel(electronLabel, electrons)
    event.getByLabel(photonLabel, photons)
    event.getByLabel(tauLabel, taus)

    # read Muons
    for i,mu in enumerate(muons.product()):
        # cuts
        if mu.pt() < 25. or mu.eta() > 2.4: continue
        if verbosity > 1: print_info(mu)
        candidates.append( Candidate(mu, "muon") )

    # read Electrons
    for i,el in enumerate(electrons.product()):
        if el.pt() < 25. or el.eta() > 2.4: continue
        if verbosity > 1: print_info(el)
        candidates.append( Candidate(el, "electron") )

    # Photon
    for i,pho in enumerate(photons.product()):
        if pho.pt() < 20 or pho.chargedHadronIso()/pho.pt() > 0.3: continue
        if verbosity > 1: print_info(pho)
        candidates.append( Candidate(pho, "photon") )
       
    # Tau
    event.getByLabel(tauLabel, taus)
    for i,tau in enumerate(taus.product()):
        if tau.pt() < 25.: continue
        if verbosity > 1: print_info(tau)
        candidates.append( Candidate(tau, "tau") )
    
    algo = "CHS"                # 
    event.getByLabel(jetLabel, jets)
    for i,j in enumerate(jets.product()):
        if j.pt() < 20. or j.eta() > 2.4: continue
        if verbosity > 1: print_info(j)
        
        candidates.append( Candidate(j, "jet") )
    return candidates
    




def print_info(obj ):
    ''' print particle info '''
    print "particle (PID=%+6d): pt %+5.2f | eta %+2.2f | phi %+4.2f | nDaughters %3d | nMothers %3d"%(
        obj.pdgId(), obj.pt(), obj.eta(), obj.phi(), obj.numberOfDaughters(), obj.numberOfMothers())

class Event:
    def __init__(self, event_type = "ttH"):
        self.event_type = event_type

    def addHadTop(self, obj):
        self.hadronic_top = obj

    def addHadB(self, obj):
        self.hadronic_bquark = obj
    
    def addLepTop(self, obj):
        self.leptonic_top = obj

    def addLepB(self, obj):
        self.leptonic_bquark = obj

    def addLepton(self, obj):
        self.lepton = obj
        
    def addBoson(self, obj):
        self.boson = obj
    
    def addBosonB1(self, obj):
        self.boson_bquark1 = obj

    def addBosonB2(self, obj):
        self.boson_bquark2 = obj

    def genObjDictionary(self):
        objects = {}
        objects["hadTop"] = [self.hadronic_top]
        objects["lepTop"] = [self.leptonic_top]
        
        objects["hadB"] = [self.hadronic_bquark]
        objects["lepB"] = [self.leptonic_bquark]

        objects["Lepton"] = [self.lepton]

        objects["Boson"] = [self.boson]
        objects["BosonB"] = [self.boson_bquark1, self.boson_bquark2]

        objects["B"] = [self.boson_bquark1, self.boson_bquark2, self.hadronic_bquark, self.leptonic_bquark]
        self.objects = objects

    def passesCuts(self):
        for key in self.objects:
            for obj in self.objects[key]:
                if obj.eta() == -22760.31640625:
                    print("what the actual fuck")
                if not type(obj) == ROOT.reco.GenParticle:
                    print(type(obj))
                    return False
                if obj.eta() > 5. or obj.eta() < -5.: return False
                if obj.phi() > 5. or obj.phi() < -5.: return False
        return True

    def printEventInfo(self):
        print("EVENT {}".format(self.event_type))
        print("\thadronic top: "+self.printObj(self.hadronic_top))
        print("\thadronic b:   "+self.printObj(self.hadronic_bquark))
        print("\tleptonic top: "+self.printObj(self.leptonic_top))
        print("\tleptonic b:   "+self.printObj(self.leptonic_bquark))
        print("\tlepton:       "+self.printObj(self.lepton))
        print("\tH/Z boson:    "+self.printObj(self.boson))
        print("\tboson b1:     "+self.printObj(self.boson_bquark1))
        print("\tboson b2:     "+self.printObj(self.boson_bquark2))

    def printObj(self, obj):
        return " | ID: {id:+5d} | eta: {eta:+1.2f} | phi: {phi:+3.2f} | pt: {pt:3d} | mass: {mass:3d}".format(
            id=int(obj.pdgId()), eta=obj.eta(), phi=obj.phi(), pt=int(obj.pt()), mass=int(obj.mass()))

    def get_dEta(self, obj1, obj2):
        detas = []
        for o1 in self.objects[obj1]:
            for o2 in self.objects[obj2]:
                detas.append( np.abs(o1.eta() - o2.eta()) )
        return detas

    def get_dPhi(self, obj1, obj2):
        dphis = []
        for o1 in self.objects[obj1]:
            for o2 in self.objects[obj2]:
                dphis.append( np.abs(o1.phi() - o2.phi()) )
        return dphis

    def get_dR(self, obj1, obj2):
        drs = []
        for o1 in self.objects[obj1]:
            for o2 in self.objects[obj2]:
                drs.append( np.sqrt( (o1.phi()-o2.phi())**2 + (o1.eta()-o2.eta())**2 ) )
        return drs

def readGenEvent(iev, event, event_type = "ttH"):
    event.getByLabel(jetLabel, jets)
    n_jets = 0
    n_tags = 0
    for i,j in enumerate(jets.product()):
        if j.pt() < 20. or j.eta() > 2.4: continue
        n_jets += 1
        if j.bDiscriminator("pfDeepCSVJetTags:probb")+j.bDiscriminator("pfDeepCSVJetTags:probbb") > 0.45:
            n_tags += 1
    if n_jets < 4: return None
    if n_tags < 3: return None

    event.getByLabel(genParticle_Label, genParticles)
    
    # common particles
    # search for top and anti top quark
    top = find_particle(genParticles, id = 6)
    topbar = find_particle(genParticles, id = -6)

    # get top and anti top daughters
    top_daughters = get_daughters(top)
    topbar_daughters = get_daughters(topbar)
    
    # get b quarks from top decays
    try:
        top_b = find_particle(top_daughters, id = 5)
    except:
        return None
    try: 
        topbar_bbar = find_particle(topbar_daughters, id = -5)
    except:
        return None

    # get w bosons from top decays
    try:
        top_w = find_particle(top_daughters, id = 24)
        topbar_w = find_particle(topbar_daughters, id = -24)
    except:
        return None

    # find out which top is leptonic
    # add particles accordingly
    if is_leptonic_W(top_w):
        lepton  = get_lepton(top_w)
        had_top = topbar
        lep_top = top
        had_b   = topbar_bbar
        lep_b   = top_b
    elif is_leptonic_W(topbar_w):
        lepton  = get_lepton(topbar_w)
        had_top = top
        lep_top = topbar
        had_b   = top_b
        lep_b   = topbar_bbar
    else:
        return None

    # boson
    if "ttH" in event_type:
        boson = find_particle(genParticles, id = 25)
    elif "ttZ" in event_type:
        boson = find_particle(genParticles, id = 23)
    else:
        return None


    # check if boson decays to bs
    if is_bb_decay(boson):
        boson_daughters = get_daughters(boson)
        boson_b1 = find_particle(boson_daughters, id = 5)
        boson_b2 = find_particle(boson_daughters, id = -5)
    else:
        return None

    evt = Event(event_type)

    # hadronic top
    evt.addHadTop(had_top)
    evt.addHadB(had_b)

    # leptonic top
    evt.addLepTop(lep_top)
    evt.addLepB(lep_b)
    evt.addLepton(lepton)

    # boson
    evt.addBoson(boson)
    evt.addBosonB1(boson_b1)
    evt.addBosonB2(boson_b2)

    evt.genObjDictionary()
    if not evt.passesCuts():
        del evt
        return None

    return evt
# particle searching functions
def find_particle(particles, id):
    ''' find particle candidate via id
        returns found candidate '''
    try: enum = enumerate(particles.product())
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
