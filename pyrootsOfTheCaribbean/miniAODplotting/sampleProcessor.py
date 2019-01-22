# import ROOT in batch mode
import sys
import ROOT
import pandas as pd
import numpy as np
import glob
import os
import stat
# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so");
ROOT.gSystem.Load("libDataFormatsFWLite.so");
ROOT.FWLiteEnabler.enable()

# load FWlite python libraries
from DataFormats.FWLite import Handle, Events
import variableCalculations
import copy


def processSample(sample, out_path, sample_type = "ttH", XSWeight = 1.):

    print("processing {}".format(sample))
    events = Events(sample)
    dfs = []
    for iev, event in enumerate(events):
        if iev%1000==0: print("#{}".format(iev))
        
        evt = variableCalculations.readGenEvent(iev, event, event_type = sample_type)
        if evt == None: continue
        #evt.printEventInfo()
        event_df = calculateVariables(copy.deepcopy(evt), XSWeight)
        dfs.append(event_df)
    
    df = pd.concat(dfs)
    print("number of events passing cuts n stuff: {}".format(df.shape[0]))
        
    # indices
    df.set_index(["Evt_Run", "Evt_Lumi", "Evt_ID"], inplace = True, drop = True)
    
    # save dataframe
    df.to_hdf(out_path, key = "data", mode = "w")
    
    return out_path


# define which variables to add
def calculateVariables(evt, XSWeight):
    event_data = {}

    # get indices
    event_data["Evt_Run"] =         [evt.run]
    event_data["Evt_Lumi"] =        [evt.lumi]
    event_data["Evt_ID"] =          [evt.ID]
    event_data["N_Jets"] =          [evt.njets]
    event_data["N_BTagsM"] =        [evt.ntags]
    event_data["Weight_GEN_nom"] =  [evt.genWeight]
    event_data["Weight_XS"] =       [XSWeight]

    objList = ["Lepton", "hadB", "lepB", "BosonB1", "BosonB2"]
    #multiList = ["BosonB"]
    #objlist = ["hadTop", "lepTop", "Boson"]
    #objList = ["Lepton", "hadTop", "lepTop", "hadB", "lepB", "Boson", "BosonB1", "BosonB2"]
    for i, obj1 in enumerate(objList):
        for j, obj2 in enumerate(objList):
            if obj1 == obj2: continue
            if j < i: continue
            event_data["dEta_{}_{}".format(obj1,obj2)] = [evt.get_dEta(obj1,obj2)]
            event_data["dPhi_{}_{}".format(obj1,obj2)] = [evt.get_dPhi(obj1,obj2)]
            event_data["dR_{}_{}".format(obj1,obj2)] = [evt.get_dR(obj1,obj2)]
        event_data["pT_{}".format(obj1)] = [evt.get_pT(obj1)]
        event_data["eta_{}".format(obj1)] = [evt.get_eta(obj1)]
        event_data["phi_{}".format(obj1)] = [evt.get_phi(obj1)]

    df = pd.DataFrame.from_dict(event_data)
    return df






def generate_submit_scripts(sample_dict, out_dir, filedir):
    shellscripts = []
    sample_parts = {}
    for sample in sample_dict:
        files = glob.glob(sample_dict[sample]["data"])
        XSWeight = sample_dict[sample]["XSWeight"]
        outputs = []
        for i, f in enumerate(files):
            shell_path, output_file = write_single_script(i, f, out_dir, filedir, sample, XSWeight)
            shellscripts.append(shell_path)
            outputs.append(output_file)
        sample_parts[sample] = outputs

    return shellscripts, sample_parts

def write_single_script(i, f, out_dir, file_dir, sample, XSWeight):
    shell_path = out_dir + "/shell_scripts/"
    python_file = file_dir + "/miniAODplotting/processSingleSample.py"
    
    if not os.path.exists(shell_path):
        os.makedirs(shell_path)

    shell_path += str(sample)+"_"+str(i)+".sh"
    out_file = out_dir + "/h5parts/"
    if not os.path.exists(out_file):
        os.makedirs(out_file)
    out_file += str(sample)+"_"+str(i)+".h5"

    script =  "#!/bin/bash\n"
    script += "export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch \n"
    script += "source $VO_CMS_SW_DIR/cmsset_default.sh \n"
    script += "export SCRAM_ARCH="+os.environ['SCRAM_ARCH']+"\n"
    script += "cd /nfs/dust/cms/user/vdlinden/CMSSW/CMSSW_9_2_4/src\n"
    script += "eval `scram runtime -sh`\n"
    script += "cd - \n"
    script += "python "+str(python_file)+" "+str(f)+" "+str(out_file)+" "+str(sample)+" "+str(XSWeight)+"\n"

    with open(shell_path, "w") as shf:
        shf.write(script)
    st = os.stat(shell_path)
    os.chmod(shell_path, st.st_mode | stat.S_IEXEC)
    print("wrote shell script "+str(shell_path))
    return shell_path, out_file

def concat_samples(sample_parts, path):
    for sample in sample_parts:
        print("concatenating for sample "+str(sample))
        parts = sample_parts[sample]

        out_file = path + "/" + sample + ".h5"

        concat_df = pd.DataFrame()
        for iPart, p in enumerate(parts):
            print("({}/{}) adding file {}".format(iPart+1,len(parts),p))
            with pd.HDFStore(p, mode = "r") as store:
                df = store.select("data")

            with pd.HDFStore(out_file, mode = "a") as store:
                store.append("data", df, index = False)
            







