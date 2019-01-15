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


def processSample(sample, out_path, sample_type = "ttH"):

    print("processing {}".format(sample))
    events = Events(sample)

    event_list = []
    for iev, event in enumerate(events):
        if iev%1000==0: print("#{}".format(iev))
        
        evt = variableCalculations.readGenEvent(iev, event, event_type = sample_type)
        if evt == None: continue
        #evt.printEventInfo()
        event_list.append(copy.deepcopy(evt))

    print("number of events passing cuts n stuff: {}".format(len(event_list)))
    df = calculateVariables(event_list)
    out_file = out_path + "/"+str(sample_type)+".h5"
    with pd.HDFStore(out_file, "a") as store:
        store.append("data", df, index = False)






# define which variables to add
def calculateVariables(event_list):
    df = pd.DataFrame()
    
    # add variables
    objList = ["Lepton", "hadTop", "lepTop", "hadB", "lepB", "Boson"]
    multiObjects = ["B", "BosonB"]

    for i, obj1 in enumerate(objList):
        for j, obj2 in enumerate(multiObjects+objList):
            if obj1 == obj2: continue
            if j+2 < i: continue
            df["dEta_{}_{}".format(obj1,obj2)] = pd.Series([elem for list_elem in [evt.get_dEta(obj1, obj2) for evt in event_list] for elem in list_elem])
            df["dPhi_{}_{}".format(obj1,obj2)] = pd.Series([elem for list_elem in [evt.get_dPhi(obj1, obj2) for evt in event_list] for elem in list_elem])
            df["dR_{}_{}".format(obj1,obj2)] = pd.Series([elem for list_elem in [evt.get_dR(obj1, obj2) for evt in event_list] for elem in list_elem])
    
    return df






def generate_submit_scripts(sample_dict, out_dir, filedir):
    shellscripts = []
    for sample in sample_dict:
        files = glob.glob(sample_dict[sample])
        for i, f in enumerate(files):
            shell_path = write_single_script(i, f, out_dir, filedir, sample)
            shellscripts.append(shell_path)

    return shellscripts

def write_single_script(i, f, out_dir, file_dir, sample):
    shell_path = out_dir + "/shell_scripts/"
    python_file = file_dir + "/miniAODplotting/processSingleSample.py"
    
    if not os.path.exists(shell_path):
        os.makedirs(shell_path)

    shell_path += str(sample)+"_"+str(i)+".sh"


    script =  "#!/bin/bash\n"
    script += "export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch \n"
    script += "source $VO_CMS_SW_DIR/cmsset_default.sh \n"
    script += "export SCRAM_ARCH="+os.environ['SCRAM_ARCH']+"\n"
    script += "cd /nfs/dust/cms/user/vdlinden/CMSSW/CMSSW_9_2_4/src\n"
    script += "eval `scram runtime -sh`\n"
    script += "cd - \n"
    script += "python "+str(python_file)+" "+str(f)+" "+str(out_dir)+" "+str(sample)+"\n"

    with open(shell_path, "w") as shf:
        shf.write(script)
    st = os.stat(shell_path)
    os.chmod(shell_path, st.st_mode | stat.S_IEXEC)
    print("wrote shell script "+str(shell_path))
    return shell_path


