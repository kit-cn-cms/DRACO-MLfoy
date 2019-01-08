# global imports
import numpy as np
import os
import sys
import socket

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

import DRACO_Frameworks.DNN.DNN as DNN
import variable_sets.dnnVariableSet as variable_info
import DRACO_Frameworks.DNN.data_frame as data_frame

        
categories = {
    "4j_ge3t":   "(N_Jets == 4 and N_BTagsM >= 3)",
    "5j_ge3t":   "(N_Jets == 5 and N_BTagsM >= 3)",
    "ge6j_ge3t": "(N_Jets >= 6 and N_BTagsM >= 3)",
    }

event_classes = ["ttHbb", "ttbb", "tt2b", "ttb", "ttcc", "ttlf"]

workpath = "/storage/9/jschindler/ttbb_studies/amcatnlo"

inPath = "/storage/9/jschindler/AachenDNN_files"

key = sys.argv[1]

outpath = workpath+"/DNN_"+str(key)+"/"

dnn = DNN.DNN(
    in_path             = inPath,
    save_path           = outpath,
    event_classes       = event_classes,
    event_category      = key,
    train_variables     = variable_info.variables[key],
    additional_cut      = None)

dataset_paths = ["/storage/9/jschindler/ttbb_studies/Powheg_Helac","/storage/9/jschindler/ttbb_studies/Powheg_openloops","/storage/9/jschindler/ttbb_studies/amcatnlo"]

dnn.load_trained_model( "/storage/9/jschindler/DNN_preaproval/DRACO-MLfoy/workdir"+"/DNN_"+str(key)+"/checkpoints/" )
dnn.eval_model_custom_dataset(dataset_path=dataset_paths, classes=event_classes)
dnn.plot_ttbb()