import sys
import os
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

    event_data["Evt_Run"] = event.eventAuxiliary().run()
    event_data["Evt_Lumi"] = event.eventAuxiliary().luminosityBlock()
    event_data["Evt_ID"] = event.eventAuxiliary().event()

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
        
    event_data["nLeptons_mAOD"] = len(candidates)

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
        # only read out jets with eta < 2.5
        if np.abs(j.eta()) > 2.5: continue
        
        # candidates.append( Candidate(j, "jet") )
        nJets += 1
        # loop over jet constituents and only add them to the histogram
        constituents = [ j.daughter(i2) for i2 in xrange(j.numberOfDaughters()) ]
        for i2, cand in enumerate(constituents):
            candidates.append( Candidate(cand, "jet_"+str(i)+"_candidate") )

    event_data["nJets_mAOD"] = nJets

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
    flattened = H.flatten()

    if hdfConfig.lognorm:
        flattened = np.array([ np.log(f) if f > 1. else 0. for f in flattened])

    # norm entries between 0 and 1
    maximum = np.max(flattened)
    flattened = [np.uint8(f/maximum*255) for f in flattened]
    return flattened

def passes_cuts(event):
    ''' determine whether event fulfills the cut criteria '''
    
    event.getByLabel(jetLabel, jets)
    nJets = 0
    nTags = 0
    for i,j in enumerate(jets.product()):
        if abs(j.eta()) < 2.5 and j.pt() > 20:
            nJets += 1
            if j.bDiscriminator("pfDeepCSVJetTags:probb")+j.bDiscriminator("pfDeepCSVJetTags:probbb") > 0.45:
                nTags += 1

    if nJets >= 4 and nTags >=3: return True
    else: return False

def load_data( inFile, outFile, hdfConfig):
    ''' loading data from a single .root file
        saving it a s dataframe in given outFile '''
    # initializing events
    events = Events(inFile)

    # determine list of indices from hdfConfig
    etabins, phibins = hdfConfig.imageSize
    indices = ["eta{}phi{}".format(eta,phi) for eta in range(etabins) for phi in range(phibins)]

    # init empty data
    evt_data = []
    data = []
    n_evts_in_file = 0
    n_evts_in_acc = 0
    for iev, event in enumerate(events):
        if iev%1000 == 0: print("at event #"+str(iev))
        n_evts_in_file+=1
        #if iev > 10: break
        if not passes_cuts(event): continue
        n_evts_in_acc += 1

        #read particle candidates of event
        candidates, event_data = read_event(iev, event)
        # generate 2dhistogram 
        hist_flat = get_2dhist( candidates, hdfConfig )
        # append flattened hist to data list
        data.append( hist_flat )
        # append additional event data to list
        evt_data.append( event_data )

    df = pd.DataFrame.from_records(data, columns = indices)
    # cast pixels as unsigned integer in [0,255]
    for col in df.columns:
        df[col] = df[col].astype(np.uint8)

    for key in evt_data[0]:
        values = [evt_data[i][key] for i in range(len(evt_data))]
        df[key] = pd.Series( values, index = df.index )

    print("reindexing")
    df.set_index(["Evt_Run", "Evt_Lumi", "Evt_ID"], inplace = True, drop = True)
    print("writing data to dataframe")
    df.to_hdf( outFile, key = "data", mode = "w" )
    
    #meta info
    print("saving meta info")
    print("{}/{} events in acceptance".format(n_evts_in_acc, n_evts_in_file))
    input_shape = (hdfConfig.imageSize[0], hdfConfig.imageSize[1], hdfConfig.nImages)

    meta_info_dict = {"input_shape": input_shape, "n_events": n_evts_in_acc}

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


 
def add_output_files(sample_dict, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # loop over samples
    for sample in sample_dict:
        print("\n\nadding together {} files ...".format(sample))
        print("="*50)
    
        # set path to output file
        out_file = out_path + "/"+ sample + ".h5"
        if os.path.exists(out_file):
            print("removing old h5file ...")
            os.remove(out_file)

        # count number of events
        n_total = 0

        # glob input files
        in_files = glob.glob( sample_dict[sample] )
        n_files = len(in_files)
        print("number of input files: {}".format(n_files))

        # initializing loop over files
        n_entries = 0
        max_entries = 50000
        concat_df = pd.DataFrame()

        # loop over input files
        for i,f in enumerate(in_files):
            print("({}/{}) loading file {}".format(i+1,n_files,f))
        
            # open file
            df = pd.read_hdf(f, key = "data")
            n_entries += df.shape[0]
            n_total += df.shape[0]

            # concatenate dataframes
            if concat_df.empty: concat_df = df
            else:               concat_df = concat_df.append(df)

            # if number of entries exceeds max threshold, save to output file
            if (n_entries > max_entries or f == in_files[-1]):
                print("*"*50)
                print("max_entries reached ...")
                with pd.HDFStore(out_file, "a") as store:
                    store.append("data", concat_df, index = False)
                print("{} events added to {}".format(n_entries, out_file))
                n_entries = 0
                concat_df = pd.DataFrame()
                print("*"*50)
    
        # end of sample printout
        print("="*50)
        print("done with {}".format(sample))
        print("total events: {}".format(n_total))
    
        # print adding meta data to file
        last_file = in_files[-1]
        meta_df = pd.read_hdf(last_file, "meta_info")
        meta_df["n_events"] = n_total
        with pd.HDFStore(out_file, "a") as store:   
            store.append("meta_info", meta_df, index = False)
        print("added meta information:")
        print(meta_df)

    print("done.")
            








