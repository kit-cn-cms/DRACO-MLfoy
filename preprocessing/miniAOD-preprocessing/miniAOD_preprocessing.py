import sys
import numpy as np
import pandas as pd

# import ROOT in batch mode
oldargv = sys.argv[:]
sys.argv = [ '-b-' ]
import ROOT
ROOT.gROOT.SetBatch(True)
sys.argv = oldargv

# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so");
ROOT.gSystem.Load("libDataFormatsFWLite.so");
ROOT.FWLiteEnabler.enable()

# load FWlite python libraries
from DataFormats.FWLite import Handle, Events

# initialization of particle classes
muons,          muonLabel       = Handle("std::vector<pat::Muon>"),     "slimmedMuons"
electrons,      electronLabel   = Handle("std::vector<pat::Electron>"), "slimmedElectrons"
photons,        photonLabel     = Handle("std::vector<pat::Photon>"),   "slimmedPhotons"
taus,           tauLabel        = Handle("std::vector<pat::Tau>"),      "slimmedTaus"
jets,           jetLabel        = Handle("std::vector<pat::Jet>"),      "slimmedJets"

class HDFConfig():
    ''' configuration of data written to h5 file '''
    def __init__(self, imageSize, etaRange, phiRange, logNorm):
        # image size with [x,y,z] size
        self.imageSize = imageSize[:2]
        self.nImages = imageSize[2]
        self.etaRange = etaRange
        self.phiRange = phiRange
        self.lognorm = logNorm

class Candidate():
    ''' save neccesary info about a particle candidate instead of copying the whole object '''
    def __init__(self, obj, name):
        self.name = name
        self.eta = obj.eta()
        self.phi = obj.phi()*180./(np.pi)
        #define what to use as histogram weight
        self.weight = obj.pt()
        del obj

        
def read_event(iev, event, applyCuts = True):
    ''' read info about a single event 
        returns list of Candidate-type classes 
            and
        dictionary of event data, which can be edited in the function
            the dictionary is used to write additional information to the dataframes
            like number of leptons or jets '''
    # initialize list for candidates
    candidates = []
    event_data = {}

    event.getByLabel(muonLabel, muons)
    event.getByLabel(electronLabel, electrons)
    event.getByLabel(photonLabel, photons)
    event.getByLabel(tauLabel, taus)
    event.getByLabel(jetLabel, jets)

    # read Muons
    for i,mu in enumerate(muons.product()):
        # cuts
        if applyCuts and (mu.pt() < 5 or not mu.isLooseMuon()): continue
        candidates.append( Candidate(mu, "muon") )

    # read Electrons
    for i,el in enumerate(electrons.product()):
        if applyCuts and (el.pt() < 5): continue
        candidates.append( Candidate(el, "electron") )
        
    event_data["nLeptons"] = len(candidates)

    # Photon
    for i,pho in enumerate(photons.product()):
        if applyCuts and (pho.pt() < 20 or pho.chargedHadronIso()/pho.pt() > 0.3): continue
        candidates.append( Candidate(pho, "photon") )
       
    # Tau
    for i,tau in enumerate(taus.product()):
        if applyCuts and (tau.pt() < 20): continue
        candidates.append( Candidate(tau, "tau") )
    
    # Jets
    nJets = 0
    for i,j in enumerate(jets.product()):
        if applyCuts and (j.pt() < 20): continue
        # candidates.append( Candidate(j, "jet") )
        nJets += 1
        # loop over jet constituents and only add them to the histogram
        constituents = [ j.daughter(i2) for i2 in xrange(j.numberOfDaughters()) ]
        for i2, cand in enumerate(constituents):
            candidates.append( Candidate(cand, "jet_"+str(i)+"_candidate") )

    event_data["nJets"] = nJets

    return candidates, event_data


def get_2dhist( candidates, hdfConfig):
    ''' create 2d histogram of event 
        returns flattened matrix ready to be saved in dataframe '''
    eta = [ c.eta for c in candidates ]
    phi = [ c.phi for c in candidates ]
    weights = [ c.weight for c in candidates ]
    H, _, _ = np.histogram2d(
        x =             eta, 
        y =             phi, 
        bins =          hdfConfig.imageSize, 
        range =         [hdfConfig.etaRange, hdfConfig.phiRange], 
        weights =       weights )
    # transpose histogram (makes reshaping easier) and flatten into 1d string
    flattened = H.T.flatten()

    if hdfConfig.lognorm:
        flattened = np.array([ np.log(f) if f > 1. else 0. for f in flattened])

    # norm entries between 0 and 1
    maximum = np.max(flattened)
    flattened = flattened/maximum
    return flattened

def load_data( inFile, outFile, hdfConfig):
    ''' loading data from a single .root file
        saving it a s dataframe in given outFile '''
    # initializing events
    events = Events(inFile)

    # init empty data
    evt_data = []
    data = []
    for iev, event in enumerate(events):
        if iev%1000 == 0:
            print("at event #"+str(iev))
        #read particle candidates of event
        candidates, event_data = read_event(iev, event)
        # generate 2dhistogram 
        hist_flat = get_2dhist( candidates, hdfConfig )
        # append flattened hist to data list
        data.append( hist_flat )
        # append additional event data to list
        evt_data.append( event_data )

    df = pd.DataFrame.from_records(data)
    for key in evt_data[0]:
        values = [evt_data[i][key] for i in range(len(evt_data))]
        df[key] = pd.Series( values, index = df.index )

    print("writing data to dataframe")
    df.to_hdf( outFile, key = "data", mode = "w" )
    
    #meta info
    print("saving meta info")
    evts_in_file = len(list(events))
    input_shape = (hdfConfig.imageSize[0], hdfConfig.imageSize[1], hdfConfig.nImages)

    meta_info_dict = {"input_shape": input_shape, "n_events": evts_in_file}

    meta_info_df = pd.DataFrame.from_dict( meta_info_dict )
    df.to_hdf( outFile, key = "meta_info", mode = "a")


def split_dataframe(df, train_pc, val_pc):
    ''' split dataframe into train/validation/test set '''
    if train_pc + val_pc >= 1.:
        print("train and validation percentages too big")
        exit()
    
    entries = df.shape[0]
    train_size = int(entries*train_pc)
    val_size = int(entries*val_pc)
    train_df = df.head(train_size)
    leftover = df.tail(entries - train_size)
    
    val_df = leftover.head(val_size)
    test_df = leftover.tail(entries-train_size-val_size)

    return train_df, val_df, test_df
    

def merge_training_sample(sample_dict, outFile, train_percentage = 0.5, val_percentage = 0.3):
    ''' merge dataframes from single processes into training dataframe
        output dataframe is shuffled and split into train/val/test sets '''

    from sklearn.utils import shuffle

    # looping over all classes
    data = None
    for n, key in enumerate(sample_dict):
        inFile = sample_dict[key]
        print("loading "+str(inFile))
        
        # getting keys
        with pd.HDFStore(inFile,"r") as s:
            df_keys = s.keys()

        # loading all subsamples in file
        for ik, df_key in enumerate(df_keys):
            print("... at dataframe "+str(df_key))
            df = pd.read_hdf(inFile, key = df_key)
            if not isinstance(data, pd.DataFrame):
                data = df
            else:
                data = data.append(df, ignore_index = True)

        print("... adding labels")
        data["textLabel"] = pd.Series( [key]*data.shape[0], index = data.index )
        data["numLabel"] = pd.Series( [n]*data.shape[0], index = data.index)

    print("done loading all data")
    print("shuffling data")
    data = shuffle(data).reset_index(drop=True)
    train_df, val_df, test_df = split_dataframe(data, 
        train_pc = train_percentage, val_pc = val_percentage)

    print("writing train sample with size "+str(train_df.shape))
    train_df.to_hdf(outFile, key = "df_train", mode = "a")
    print("writing val sample with size "+str(val_df.shape))
    val_df.to_hdf(outFile, key = "df_val", mode = "a")
    print("writing test sample with size "+str(test_df.shape))
    test_df.to_hdf(outFile, key = "df_test", mode = "a")

    print("done.")


        







