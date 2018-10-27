import sys
import numpy as np
import pandas as pd
import glob

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

class Candidate():
    ''' save neccesary info about a particle candidate instead of copying the whole object '''
    def __init__(self, obj, name):
        self.name = name
        self.obj = obj
        self.pt = obj.pt()
        self.phi = obj.phi()
        del obj

    def return_vector(self, first_jet):
        pt = self.obj.pt()
        pz = self.obj.pz()
        energy = self.obj.energy()
        phi = self.obj.phi() - first_jet.obj.phi()
        eta = self.obj.eta()
        mass2 = self.obj.mass()**2        
        
        features = [energy, pt, pz, mass2, phi, eta]
        names = ["E", "pT", "pz", "M2", "phi", "eta"]
        return features, names
        

        
def read_event(iev, event, applyCuts = True):
    ''' read info about a single event 
        returns list of Candidate-type classes 
            and
        dictionary of event data, which can be edited in the function
            the dictionary is used to write additional information to the dataframes
            like number of leptons or jets '''
    # initialize list for candidates
    candidates = {}
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
        candidates[Candidate(mu, "muon")] = mu.pt()

    # read Electrons
    for i,el in enumerate(electrons.product()):
        if applyCuts and (el.pt() < 5): continue
        candidates[Candidate(el, "electron")] = el.pt()
        
    event_data["nLeptons"] = len(candidates)

    # Photon
    for i,pho in enumerate(photons.product()):
        if applyCuts and (pho.pt() < 20 or pho.chargedHadronIso()/pho.pt() > 0.3): continue
        candidates[Candidate(pho, "photon")] = pho.pt()
       
    # Tau
    for i,tau in enumerate(taus.product()):
        if applyCuts and (tau.pt() < 20): continue
        candidates[Candidate(tau, "tau")] = tau.pt()
    
    # Jets
    nJets = 0
    jet_candidates = {}
    for i,j in enumerate(jets.product()):
        if applyCuts and (j.pt() < 20): continue
        # only read out jets with eta < 2.5
        if np.abs(j.eta()) > 2.5: continue
        jet_candidates[Candidate(j, "jet")] = j.pt()
        # candidates.append( Candidate(j, "jet") )
        nJets += 1
        # loop over jet constituents and only add them to the histogram
        constituents = [ j.daughter(i2) for i2 in xrange(j.numberOfDaughters()) ]
        for i2, cand in enumerate(constituents):
            candidates[Candidate(cand, "jet_"+str(i)+"_candidate")] = cand.pt()

    event_data["nJets"] = nJets
    pt_sorted_candidates = pt_sort(candidates)
    first_jet = pt_sort(jet_candidates)[0]
        

    return pt_sorted_candidates, event_data, first_jet

def pt_sort(candidates):
    sorted_cands = np.array(sorted(candidates.iteritems(), key = lambda (k,v): (v,k), reverse = True))
    return sorted_cands[:,0]

def load_data( inFile, outFile):
    ''' loading data from a single .root file
        saving it a s dataframe in given outFile '''
    # initializing events
    events = Events(inFile)
    n_cands = 75
    # init empty data
    evt_data = []
    data = []
    for iev, event in enumerate(events):
        if iev == 15000: break
        if iev%1000 == 0:
            print("at event #"+str(iev))
        #read particle candidates of event
        pt_sorted_candidates, event_data, first_jet = read_event(iev, event)
        pt_sorted_candidates = pt_candidates[:n_cands]    
        features = []
        labels = []
        for i, cand in enumerate(pt_sorted_candidates):
            f, l = cand.return_vector(first_jet)
            n_features = len(f)
            l = [label+"_"+str(i) for label in l]
            features += f
            labels += l
        data.append( features )
        evt_data.append(event_data)
        

    df = pd.DataFrame.from_records(data, columns = labels)
    for key in evt_data[0]:
        values = [evt_data[i][key] for i in range(len(evt_data))]
        df[key] = pd.Series( values, index = df.index )

    print("writing data to dataframe")
    df.to_hdf( outFile, key = "data", mode = "w" )

    #meta info
    print("saving meta info")
    input_shape = (1,n_cands,n_features)

    meta_info_dict = {"input_shape": input_shape}

    meta_info_df = pd.DataFrame.from_dict( meta_info_dict )
    meta_info_df.to_hdf( outFile, key = "meta_info", mode = "a")




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

def check_image_size(old_size, new_size):
    if not old_size.equals( new_size ):
        print("sizes are not compatible")
        print("old image size: "+str(old_size))
        print("new image size: "+str(new_size))
        exit()

def prepare_training_sample(sample_dict, outFile, train_pc = 0.5, val_pc = 0.3):
    ''' prepare dataframes from single processes into training dataframe
        output dataframe is shuffled and split into train/val/test sets '''
    from sklearn.utils import shuffle

    df = None
    key_dict = {}
    image_size = None

    # looping over all classes
    for n, key in enumerate(sample_dict):
        class_label = len(key_dict)
        print(str(key)+" got class label "+str(class_label))
        key_dict[key] = class_label


        # grep all files with filename_*.h5 loop over them
	file_path = sample_dict[key]
        files = [file_path]
	if "*" in file_path:
            print("globbed files from "+str(file_path))
            files = glob.glob( file_path )
            print(files)

        # add them all to one data frame
        for f in files:
            print("loading "+str(f))
            new_df = pd.read_hdf(f, key = "data")
            meta = pd.read_hdf(f, key = "meta_info")
            if not isinstance(image_size, pd.Series):
                image_size = meta["input_shape"]
            else:
                # check if image sizes are different (that would not be compatible for training)
                check_image_size( image_size, meta["input_shape"] )       

            # add labels depending on classes
            new_df["class_label"] = pd.Series( [class_label]*new_df.shape[0], index = new_df.index )

            # concatenating dataframes
            if not isinstance(df, pd.DataFrame):
                print("creating new instance of dataframe")
                df = new_df
            else:
                df = df.append(new_df, ignore_index = True)
            del new_df
    
        
    print("done loading all data")
    print("creating meta info dataframes")
    label_df = pd.DataFrame(key_dict, index = [0])
    image_size = image_size.values
    shape = {"x": image_size[0], "y": image_size[1], "z": image_size[2]}
    print("output images will have shape "+str(shape))
    shape_df = pd.DataFrame( shape, index = [0] )

    print("shuffling data")
    df = shuffle(df).reset_index(drop=True)
    
    # get splitting ratios for train/val/test set  
    entries = df.shape[0]
    train_size = int(entries*train_pc)
    val_size = int(entries*val_pc)
    test_size = entries - train_size - val_size

    # split into three dataframe according to split percentage
    
    # handle training data
    train_file = outFile+"_train.h5"
    print("writing train data to "+str(train_file)+" with "+str(train_size)+" entries")
    train_df = df.head(train_size)
    train_df.to_hdf( train_file, key = "data", mode = "w")
    del train_df
    # add meta info
    label_df.to_hdf( train_file, key = "label_dict", mode = "a" )
    shape_df.to_hdf( train_file, key = "image_size", mode = "a" )


    # handle validation data
    val_file = outFile+"_val.h5"
    print("writing val data to "+str(val_file)+" with "+str(val_size)+" entries")
    leftover = df.tail(entries - train_size)
    val_df = leftover.head(val_size)
    val_df.to_hdf( val_file, key = "data", mode = "w")
    del df
    del val_df
    # add meta info
    label_df.to_hdf( val_file, key = "label_dict", mode = "a" )
    shape_df.to_hdf( val_file, key = "image_size", mode = "a" )


    # handle test data
    test_file = outFile+"_test.h5"
    print("writing test data to "+str(test_file)+" with "+str(test_size)+" entries")
    test_df = leftover.tail(test_size)
    test_df.to_hdf( test_file, key = "data", mode = "w")
    del leftover
    del test_df
    # add meta info
    label_df.to_hdf( test_file, key = "label_dict", mode = "a" )
    shape_df.to_hdf( test_file, key = "image_size", mode = "a" )

    print("done.")


        







