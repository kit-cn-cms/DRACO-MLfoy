import ROOT
import pandas as pd
import os
import sys
import stat
import copy
# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so");
ROOT.gSystem.Load("libDataFormatsFWLite.so");
ROOT.FWLiteEnabler.enable()

# load FWlite python libraries
from DataFormats.FWLite import Handle, Events

jets,           jetLabel            = Handle("std::vector<pat::Jet>"),          "slimmedJets"
genParticles,   genParticleLabel    = Handle("std::vector<reco::GenParticle>"), "prunedGenParticles"




def generate_submit_scripts(samples, output_dir, file_dir):
    ''' loop over all the samples and create a shellscript for submission to NAF '''
    python_file = file_dir+"/analyzeSingleSample.py"

    shellscripts = []
    output_files = []
    # get all specified samples
    for i, s in enumerate(samples):
        shell_path, output_file = write_single_script(i, s, output_dir, python_file)
        shellscripts.append(shell_path)
        output_files.append(output_file)

    return shellscripts, output_files

    
def write_single_script(i, s, output_dir, python_file):
    shell_dir = output_dir + "/shell_scripts/"
    if not os.path.exists(shell_dir):
        os.makedirs(shell_dir)
    out_dir = output_dir + "/output_parts/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    shell_path = shell_dir+"/analyzeSample_part_{}.sh".format(i)
    out_file = out_dir+"/sample_part_{}.h5".format(i)

    script = """
#!/bin/bash
export X509_USER_PROXY=/nfs/dust/cms/user/vdlinden/VOMSPROXY/vomsproxy
export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch
source $VO_CMS_SW_DIR/cmsset_default.sh
export SCRAM_ARCH={arch}
cd /nfs/dust/cms/user/vdlinden/CMSSW/CMSSW_9_2_4/src
eval `scram runtime -sh`
cd - 
rm {out_file}
python {python_file} {s} {out_file}
    """.format(
        arch        = os.environ['SCRAM_ARCH'],
        python_file = python_file,
        out_file    = out_file,
        s           = s)

    # save shell file
    with open(shell_path, "w") as shf:
        shf.write(script)
    st = os.stat(shell_path)
    os.chmod(shell_path, st.st_mode | stat.S_IEXEC)

    print("wrote shell script "+str(shell_path))
    return shell_path, out_file

def concat_samples(output_parts, outfile):
    ''' concatenate all the single files produced by NAF jobs '''
    for iPart, p in enumerate(output_parts):
        print("({}/{}) adding file {}".format(iPart+1,len(output_parts),p))

        # load part file
        with pd.HDFStore(p, mode = "r") as store:
            df = store.select("data")
        print("\t{} events.".format(df.shape[0]))

        # save to concatenated file
        with pd.HDFStore(outfile, mode = "a") as store:
            store.append("data", df, index = False)


def analyze(sample, output_dir):
    # read events
    events = Events(sample)

    dfs = []
    # start event loop
    for iev, event in enumerate(events):
        if iev%1000==0: print("#{}".format(iev))

        event_variables = {}
        # save indices for event
        event_variables["Evt_Run"]    = [event.eventAuxiliary().run()]
        event_variables["Evt_Lumi"]   = [event.eventAuxiliary().luminosityBlock()]
        event_variables["Evt_ID"]     = [event.eventAuxiliary().event()]

        # determine number of jets and btags
        nJets = 0
        nTags = 0
        event.getByLabel(jetLabel, jets)
        for i,j in enumerate(jets.product()):
            if j.pt() < 20. or abs(j.eta()) > 2.4: continue
            nJets += 1
            if j.bDiscriminator("pfDeepCSVJetTags:probb")+j.bDiscriminator("pfDeepCSVJetTags:probbb") > 0.48:
                nTags += 1
    
        event_variables["N_Jets"] = [nJets]
        event_variables["N_BTagsM"] = [nTags]


        # determine ttbar final state
        event.getByLabel(genParticleLabel, genParticles)

        # search for top and anti top quark
        top     = find_particle(genParticles, id = 6)
        topbar  = find_particle(genParticles, id = -6)

        # get their daughters
        top_decays      = get_daughters(top)
        topbar_decays   = get_daughters(topbar)

        # get w bosons from top decays
        try:
            top_W       = find_particle(top_decays, id = 24)
            topbar_W    = find_particle(topbar_decays, id = -24)
        except:
            continue

        nLeptonicWs = 0
        if is_leptonic_W(top_W): nLeptonicWs+=1
        if is_leptonic_W(topbar_W): nLeptonicWs+=1
        event_variables["leptonicTops"] = [nLeptonicWs]


        # determine Z decay channel
        try:
            Z = find_particle(genParticles, id = 23)
        except:
            continue
        
        ZToBB = 0
        ZToQQ = 0
        ZToLL = 0
        ZToNN = 0
        if isZdecay(Z, decay_products = [5,-5]):                    ZToBB=1
        if isZdecay(Z, decay_products = [1,2,3,4,-1,-2,-3,-4]):     ZToQQ=1
        if isZdecay(Z, decay_products = [11,13,15,-11,-13,-15]):    ZToLL=1
        if isZdecay(Z, decay_products = [12,14,16,-12,-14,-16]):    ZToNN=1
        
        event_variables["ZToBB"] = [ZToBB]
        event_variables["ZToQQ"] = [ZToQQ]
        event_variables["ZToLL"] = [ZToLL]
        event_variables["ZToNN"] = [ZToNN]

        df = pd.DataFrame.from_dict(copy.deepcopy(event_variables))
        dfs.append(df)

    full_df = pd.concat(dfs)
    print("number of events: {}".format(full_df.shape[0]))

    full_df.set_index(["Evt_Run", "Evt_Lumi", "Evt_ID"], inplace = True, drop = True)

    full_df.to_hdf(output_dir, key = "data", mode = "w")

    








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

def isZdecay(Z, decay_products):
    daughters = get_daughters(Z)
    found_products = 0
    for d in daughters:
        if d.pdgId() in decay_products:
            found_products+=1
    if found_products >= 2: return True
    return False

def is_leptonic_W(w):
    w_daughters = get_daughters(w)
    for d in w_daughters:
        if d.pdgId() in [11,-11,13,-13,15,-15]: return True
    return False

