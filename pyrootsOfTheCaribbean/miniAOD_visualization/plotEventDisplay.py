# import ROOT in batch mode
import sys
import ROOT
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so");
ROOT.gSystem.Load("libDataFormatsFWLite.so");
ROOT.FWLiteEnabler.enable()

# load FWlite python libraries
from DataFormats.FWLite import Handle, Events

genParticles,   genParticleLabel    = Handle("std::vector<reco::GenParticle>"), "prunedGenParticles"

def read_sample(sample):

    events = Events(sample)
    n_events = events.size()
    print("total number of events in file: {}".format(n_events))

    # start event loop
    for iev, event in enumerate(events):
        if iev%1000==0: print("#{}".format(iev))
        
        event.getByLabel(genParticleLabel, genParticles)

        already_seen = []
        ax = plt.subplot(111, projection = "3d")
        for i,p in enumerate(genParticles.product()):
            if not p.isHardProcess(): continue
            if not p.mother().pdgId() == 2212: continue
            #print(p.pdgId())
            #if p.numberOfMothers() >= 1: continue
            already_seen += get_daughter_chain(p, already_seen = already_seen)

        plt.grid()
        plt.tight_layout()
        #plt.show()
        plt.savefig("events/event_no_{}.png".format(iev))
        if iev==100: exit()
        plt.clf()

def get_daughter_chain(p, n = 0, x = 0, y = 0, z = 0, already_seen = []):
    if seen_that(p, already_seen): return []

    #print("   ."*n+str(p.pdgId()))
    x1 = x
    y1 = y
    z1 = z
    if n>=1:
        xdir = np.sin(p.theta())*np.cos(p.phi())
        ydir = np.sin(p.theta())*np.sin(p.phi())
        zdir = np.cos(p.theta())
        x1 = x + xdir/np.sqrt(xdir**2 + ydir**2 + zdir**2)
        y1 = y + ydir/np.sqrt(xdir**2 + ydir**2 + zdir**2)
        z1 = z + zdir/np.sqrt(xdir**2 + ydir**2 + zdir**2)

    plt.plot([x,x1],[y,y1],[z,z1],"-", color = get_color(p.pdgId()), lw = 2)

    already_seen += [p]
    for iD in xrange(p.numberOfDaughters()):
        already_seen += get_daughter_chain(p.daughter(iD),n+1,x1,y1,z1,already_seen)
    return [p]

def seen_that(p, already_seen):
    for elem in already_seen:
        if not elem.pdgId() == p.pdgId(): continue
        if not elem.phi() == p.phi(): continue
        if not elem.eta() == p.eta(): continue
        return True
    return False
        
def get_set(particles):
    new_set = []
    for p in particles:
        seen = False
        for unique in new_set:
            if not unique["p"].pdgId() == p["p"].pdgId(): continue
            if not unique["p"].phi() == p["p"].phi() and unique["p"].eta() == unique["p"].eta(): continue
            #if not unique["n"] == p["n"]: continue
            seen = True
        if not seen:
            new_set.append(p)

    return new_set


def plot_3d_event(particles):
    ax = plt.subplot(111, projection = "3d")
    for entry in particles:
        n = entry["n"]
        if n == 0: continue

        p = entry["p"]
        m = p.mother()
        
        x = xcoord(p,m,n)
        y = ycoord(p,m,n)
        z = zcoord(p,m,n)

        ax.plot(x,y,z,"-")

    plt.show()

def xcoord(p,m,n):
    return [
        1.*(n-1)*np.sin(m.theta())*np.cos(m.phi()),
        1.*(n  )*np.sin(p.theta())*np.cos(p.phi())
        ]

def ycoord(p,m,n):
    return [
        1.*(n-1)*np.sin(m.theta())*np.sin(m.phi()),
        1.*(n  )*np.sin(p.theta())*np.sin(p.phi())
        ]

def zcoord(p,m,n):
    return [
        1.*(n-1)*np.cos(m.theta()),
        1.*(n  )*np.cos(p.theta())
        ]

def plot_phi_projection(particles):
    ax = plt.subplot(111, projection = "polar")
    for entry in particles:
        n = entry["n"]
        if n == 0: continue
        ax.plot([m.phi(),p.phi()],[n-1,n],"-")

    plt.show()

        


def get_color(id):
    if abs(id) == 6:            return "red"
    if abs(id) == 5:            return "green"
    if abs(id) in [23,25]:      return "black"
    if abs(id) == 24:           return "yellow"
    if abs(id) in [4,3,2,1]:    return "orange"
    if abs(id) in [11,13,15]:   return "brown"
    if abs(id) in [13,14,16]:   return "pink"
    if abs(id) == 21:           return "blue"
    else:                       return "grey"



sample = "root://xrootd-cms.infn.it///store/mc/RunIIFall17MiniAODv2/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/270000/F225C5E3-FECD-E811-9141-6CC2173DAD00.root"
read_sample(sample)       
