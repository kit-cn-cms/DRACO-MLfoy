# import ROOT in batch mode
import sys
oldargv = sys.argv[:]
sys.argv = [ '-b-' ]
import ROOT
ROOT.gROOT.SetBatch(True)
sys.argv = oldargv

# import dictionary for MC numbering scheme
from mc_numbers_dict import pdg_ids

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
import numpy as np

# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so");
ROOT.gSystem.Load("libDataFormatsFWLite.so");
ROOT.FWLiteEnabler.enable()

# load FWlite python libraries
from DataFormats.FWLite import Handle, Events

muons, muonLabel = Handle("std::vector<pat::Muon>"), "slimmedMuons"
electrons, electronLabel = Handle("std::vector<pat::Electron>"), "slimmedElectrons"
photons, photonLabel = Handle("std::vector<pat::Photon>"), "slimmedPhotons"
taus, tauLabel = Handle("std::vector<pat::Tau>"), "slimmedTaus"
tauLabelb = "slimmedTausBoosted"
jets = Handle("std::vector<pat::Jet>")
fatjets, fatjetLabel = Handle("std::vector<pat::Jet>"), "slimmedJetsAK8"
mets, metLabel = Handle("std::vector<pat::MET>"), "slimmedMETs"
vertices, vertexLabel = Handle("std::vector<reco::Vertex>"), "offlineSlimmedPrimaryVertices"
verticesScore = Handle("edm::ValueMap<float>")
gen_jets, gen_jetLabel = Handle("std::vector<reco::GenJet>"), "ak4GenJets"
gen_jetLabel = "slimmedGenJets"
genParticles, genParticle_Label = Handle("std::vector<reco::GenParticle>"), "prunedGenParticles"


# 
binning = [50, 62]
color_codes = {}

# example input files
ttZqq = "/pnfs/desy.de/cms/tier2/store/user/pkeicher/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/KIT_tthbb_sl_skims_MC_94X/180618_082211/0000/Skim_1.root"
ttHbb = "/pnfs/desy.de/cms/tier2/store/user/mwassmer/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/KIT_tthbb_sl_skims_MC_94X/180617_093100/0000/Skim_1.root"


def Candidates( objects ):
    return [ Candidate(obj) for obj in objects ]

class Candidate:
    ''' container for some particle candidate information '''
    def __init__(self, obj, type = "unknown"):
        self.type = type
        self.eta = obj.eta()
        self.phi = obj.phi()/(np.pi)*180.
        self.pt = obj.pt()
        self.e = obj.energy()
        self.obj = obj
        self.ID = obj.pdgId()
    def get_entry(self):
        return (self.eta, self.phi, self.pt)

def calc_dEta(obj1, obj2):
    d_eta = np.abs( obj1.eta() - obj2.eta() )
    return d_eta

def calc_dPhi(obj1, obj2):
    d_phi = np.abs( obj1.phi() - obj2.phi() )
    if d_phi > np.pi:
        d_phi -= 2.*np.pi
    d_phi = np.abs(d_phi)
    return d_phi

def calc_dR(d_phi, d_eta):
    d_r2 = d_phi**2 + d_eta**2
    return np.sqrt( d_r2 )



def get_members(obj):
    ''' show member variables of object '''
    variables = dir(obj)
    for v in variables:
        print v

def show_mothers(obj):
    ''' print info of all mother particles of object '''
    print("\tmother candidates:")
    mothers = [obj.mother(i2) for i2 in xrange(obj.numberOfMothers())]
    for im, mother in enumerate(mothers):
        print_info(mother)

def show_daughters(obj):
    ''' print info of all daughter particles of object '''
    print("\tdaughter candidates:")
    daughters = [obj.daughter(i2) for i2 in xrange(obj.numberOfDaughters())]
    for im, daughter in enumerate(daughters):
        print_info(daughter)

def get_daughters(obj):
    ''' get list of daughter particles of object
        return all particles also intermediate particles '''
    list_of_daughters = [obj]
    if obj.numberOfDaughters() > 0:
        for di in xrange(obj.numberOfDaughters()):
            list_of_daughters += get_daughters( obj.daughter(di) )
        return list_of_daughters
    else:
        return []

def get_daughters_only_finalstate(obj):
    ''' get list of daughter particles of object
        only return particles that are not intermediate particles '''
    if obj.numberOfDaughters() == 0:
        return [obj]
    else:
        list_of_daughters = []
        for di in xrange(obj.numberOfDaughters()):
            list_of_daughters += get_daughters_only_finalstate( obj.daughter(di) )
        return list_of_daughters

def print_info(obj ):
    ''' print particle info '''
    print "particle (PID=%+6d): pt %+5.2f | eta %+2.2f | phi %+4.2f | nDaughters %3d | nMothers %3d"%(
        obj.pdgId(), obj.pt(), obj.eta(), obj.phi(), obj.numberOfDaughters(), obj.numberOfMothers())

def find_particle(particles, id, name):
    ''' find particle candidate via id
        returns found candidate '''

    for i, p in enumerate(particles.product()):
        if p.pdgId() == id:
            candidate = p
            break

    while candidate.numberOfDaughters() == 1 and candidate.daughter(0).pdgId() == id:
        candidate = candidate.daughter(0)
    return candidate
    
def find_decay(particle, ids = [], mother_ids = []):
    ''' fild decay of particle with mother_ids to particles with ids
        return the found candidates '''

    daughters = get_daughters(particle)

    candidates = []
    for d in daughters:
        mother = d.mother(0)
        if mother.pdgId() in mother_ids and d.pdgId() in ids:
            candidates.append( d )
    return candidates



def find_boson_candidate(iev, event, event_type = "ttH"):
    ''' extract boson and its decay from ttH/ttZ events '''
    if event_type == "ttH":
        pdgid = 25
        decay_ids = [5,-5]
    elif event_type == "ttZ":
        pdgid = 23
        decay_ids = [1,2,3,4,5,-1,-2,-3,-4,-5]
    else:
        print("wrong choice of event_type")
        exit()

    event.getByLabel(genParticle_Label, genParticles)

    boson_candidate = find_particle( genParticles, id = pdgid, name = event_type )
    decay_products = find_decay( boson_candidate, ids = decay_ids, mother_ids = [pdgid] )
    return boson_candidate, decay_products


def read_gen_jets(iev, event, event_type = "ttH"):
    ''' extract ttH/ttZ system from genlevel information '''

    event.getByLabel(genParticle_Label, genParticles)
    gen_info = {}

    # search for top quark
    top = find_particle(genParticles, id = 6, name = "top")
    top_daughters = get_daughters_only_finalstate(top)
    # search for b from top decay
    top_bs = find_decay(top, ids = [-5,5], mother_ids = [6])
    
    # add top to gen_info
    gen_info["top"] = {"mother": Candidate(top, "top"), "daughters": Candidates(top_daughters), "quarks": Candidates(top_bs)}
    color_codes["top"] = "blue"


    # search for anti top quark
    topb = find_particle(genParticles, id = -6, name = "anti top")
    topb_daughters = get_daughters_only_finalstate(topb)
    # search for b from top decay
    topb_bs = find_decay(topb, ids = [-5,5], mother_ids = [-6])

    # add antitop to gen_info
    gen_info["topbar"] = {"mother": Candidate(topb, "top"), "daughters": Candidates(topb_daughters), "quarks": Candidates(topb_bs)}
    color_codes["topbar"] = "green"


    if event_type == "ttH":
        # search for higgs 
        higgs = find_particle(genParticles, id = 25, name = "higgs")
        higgs_daughters = get_daughters_only_finalstate(higgs)
        # search for bs from higgs decay
        higgs_bs = find_decay(higgs, ids = [-5,5], mother_ids = [25])

        # add higgs to gen_info
        gen_info["higgs"] = {"mother": Candidate(higgs, "higgs"), "daughters": Candidates(higgs_daughters), "quarks": Candidates(higgs_bs)}
        color_codes["higgs"] = "black"

    if event_type == "ttZ":
        # search for Z boson
        Zboson = find_particle(genParticles, id = 23, name = "Z")
        Zboson_daughters = get_daughters_only_finalstate(Zboson)
        # search for quarks from higgs decay
        Zboson_qs = find_decay(Zboson, ids = [-1,1,-2,2,-3,3,-4,4,-5,5], mother_ids = [23])

        # add Zboson to gen_info
        gen_info["Zboson"] = {"mother": Candidate(Zboson, "Z"), "daughters": Candidates(Zboson_daughters), "quarks": Candidates(Zboson_qs)}
        color_codes["Zboson"] = "red"

    return gen_info


def read_event(iev, event, verbosity = 1):
    ''' read particle-flow level event data
        extract all particle candidates '''

    candidates = []
    event.getByLabel(muonLabel, muons) 
    event.getByLabel(electronLabel, electrons)
    event.getByLabel(photonLabel, photons)
    event.getByLabel(tauLabel, taus)

    # read Muons
    for i,mu in enumerate(muons.product()):
        # cuts
        if mu.pt() < 5 or not mu.isLooseMuon(): continue
        if verbosity > 1:
            print_info(mu)
        candidates.append( Candidate(mu, "muon") )

    # read Electrons
    for i,el in enumerate(electrons.product()):
        if el.pt() < 5: continue
        if verbosity > 1:
            print_info(el)
        candidates.append( Candidate(el, "electron") )

    # Photon
    for i,pho in enumerate(photons.product()):
        if pho.pt() < 20 or pho.chargedHadronIso()/pho.pt() > 0.3: continue
        if verbosity > 1:
            print_info(pho)
        candidates.append( Candidate(pho, "photon") )
       
    # Tau
    event.getByLabel(tauLabel, taus)
    for i,tau in enumerate(taus.product()):
        if tau.pt() < 20: continue
        if verbosity > 1:
            print_info(tau)
        candidates.append( Candidate(tau, "tau") )
    
    jetLabel = "slimmedJets"    # 
    algo = "CHS"                # 
    event.getByLabel(jetLabel, jets)
    for i,j in enumerate(jets.product()):
        if j.pt() < 20: continue
        if verbosity > 1:
            print_info(j)
        
        # loop over jet constituents
        constituents = [ j.daughter(i2) for i2 in xrange(j.numberOfDaughters()) ]
        constituents.sort(key = lambda c:c.pt(), reverse=True)
        for i2, cand in enumerate(constituents):
            if verbosity > 1:
                print_info(cand)
            candidates.append( Candidate(cand, "jet"+str(i)+"_constituent") )

    return candidates

def list_pdg_values( cands ):
    ''' list all particles from pdg ids '''
    parts = {}
    for c in cands:
        try:
            parts[pdg_ids[c.ID]] += 1
        except:
            parts[pdg_ids[c.ID]] = 1
    for key in parts:
        print("%10s %4d"%(key,parts[key]))

def plot_2dhist( candidates , variable = "pt"):
    ''' plot 2d map of event in eta and phi
        bin entry corresponds to given variable (TODO) '''

    print("creating "+str(variable)+" map")
    eta = [ c.eta for c in candidates ]
    phi = [ c.phi for c in candidates ]
    
    # determine bin entries
    if variable == "pt":
        z = [ c.pt for c in candidates ]
    elif variable == "energy":
        z = [ c.obj.energy() for c in candidates ]

    plt.hist2d( eta, phi, bins = binning, range = [[-2.5, 2.5],[-180, 180]], weights = z, cmin = 1e-3, cmap = "Wistia", norm = LogNorm())
    cbar = plt.colorbar()
    cbar.set_label(str(variable)+" sum in bin")


def plot_single_event(gen_particles):
    ''' plot gen level ttH/ttZ system on eta/phi plane '''

    print("adding gen level info")
    for key in gen_particles:
        c = color_codes[key]

        # plot mother particle
        part = gen_particles[key]["mother"]
        eta = part.eta
        phi = part.phi
        plt.plot(eta, phi, "o", markersize = 20, label = key, color = c)

        # plot decay quarks
        quarks = gen_particles[key]["quarks"]
        eta = []
        phi = []
        for q in quarks:
            eta.append( q.eta )
            phi.append( q.phi )
        plt.plot(eta, phi, "p", markersize = 15, label = key+"_dec_qs", color = c)

        # plot daughter particles
        daughters = gen_particles[key]["daughters"]
        eta_l, eta_p, eta = [], [], []
        phi_l, phi_p, phi = [], [], []
        for d in daughters:
            # do not plot neutrinos
            if d.ID in (12,14,16,-12,-14,-16): continue
            # add leptons
            if d.ID in (11,-11,13,-13,15,-15):
                eta_l.append( d.eta)
                phi_l.append( d.phi)
            # add photons
            elif d.ID in (22,):
                eta_p.append( d.eta )
                phi_p.append( d.phi )
            # add hadrons
            else:
                eta.append( d.eta )
                phi.append( d.phi )
        # plot them all
        plt.plot(eta, phi, "o", markersize = 5, label = key+"_dtrs_had", color = c)
        plt.plot(eta_l, phi_l, "^", markersize = 5, label = key+"_dtrs_lep", color = c)
        plt.plot(eta_p, phi_p, "*", markersize = 5, label = key+"_dtrs_pho", color = c)

        #print("pdg values of daughters of "+str(key))
        #list_pdg_values( daughters )

def plot_variable(data_sets, labels, var_name):
    ''' generate histogram for datasets '''
    glob_max = 0
    glob_min = 0
    plt.figure(figsize = [10,10])
    for data in data_sets:
        mx = np.max(data)
        mn = np.min(data)
        if mx > glob_max:
            glob_max = mx
        if mn < glob_min:
            glob_min = mn

    for i, data in enumerate(data_sets):
        plt.hist(data, bins = 30, range = [glob_min, glob_max], histtype = "step", label = labels[i] )
    plt.grid()
    plt.legend()
    plt.xlabel(var_name)


########
# MAIN #
########
n_events = 15000

# plot angular difference of boson decay products
ttZ_dr = []
ttZ_deta = []
ttZ_dphi = []

events = Events(ttZqq)
print("extracting ttZ events...")
for iev, event in enumerate(events):
    if iev > n_events: break
    if iev%1000 == 0:
        print("#"+str(iev))
    boson_candidate, decay_products = find_boson_candidate(iev, event, event_type = "ttZ")
    if len(decay_products) != 2: continue
    
    dphi = calc_dPhi( *decay_products )
    deta = calc_dEta( *decay_products )
    dr   = calc_dR(dphi, deta)

    ttZ_dr += [dr]
    ttZ_deta += [deta]
    ttZ_dphi += [dphi]

ttH_dr = []
ttH_deta = []
ttH_dphi = []

events = Events(ttHbb)
print("extracting ttH events...")
for iev, event in enumerate(events):
    if iev > n_events: break
    if iev%1000 == 0:
        print("#"+str(iev))
    boson_candidate, decay_products = find_boson_candidate(iev, event, event_type = "ttH")
    if len(decay_products) != 2: continue
    
    dphi = calc_dPhi( *decay_products )
    deta = calc_dEta( *decay_products )
    dr   = calc_dR(dphi, deta)

    ttH_dr += [dr]
    ttH_deta += [deta]
    ttH_dphi += [dphi]
    
plot_variable( data_sets = [ttZ_dphi, ttH_dphi], labels = ["ttZ", "ttH"], var_name = "delta_Phi")
plot_variable( data_sets = [ttZ_deta, ttH_deta], labels = ["ttZ", "ttH"], var_name = "delta_Eta")
plot_variable( data_sets = [ttZ_dr,   ttH_dr  ], labels = ["ttZ", "ttH"], var_name = "delta_R")
plt.show()
