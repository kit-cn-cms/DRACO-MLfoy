# global imports
import numpy as np
import os
import sys


# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

import DRACO_Frameworks.DNN_Aachen.DNN_Aachen as DNN_Aachen
import DRACO_Frameworks.DNN_Aachen.variable_info as variable_info

category_vars = {
    "4j_ge3t": variable_info.variables_4j_3b,
    "5j_ge3t": variable_info.variables_5j_3b,
    "ge6j_ge3t": variable_info.variables_6j_3b}            
categories = {
    "4j_ge3t":   "(N_Jets == 4 and N_BTagsM >= 3)",
    "5j_ge3t":   "(N_Jets == 5 and N_BTagsM >= 3)",
    "ge6j_ge3t": "(N_Jets >= 6 and N_BTagsM >= 3)",
    }
prenet_targets = [
    #"GenAdd_BB_inacceptance",
    #"GenAdd_B_inacceptance",
    "GenHiggs_BB_inacceptance",
    "GenHiggs_B_inacceptance",
    #"GenTopHad_B_inacceptance",
    #"GenTopHad_QQ_inacceptance",
    #"GenTopHad_Q_inacceptance",
    #"GenTopLep_B_inacceptance"
    ]

event_classes = ["ttHbb", "ttbb", "tt2b", "ttb", "ttcc", "ttlf"]

inPath = "/nfs/dust/cms/user/vdlinden/DRACO-MLfoy/workdir/AachenDNN_files"

key = sys.argv[1]

outpath = "/nfs/dust/cms/user/vdlinden/DRACO-MLfoy/workdir/AachenDNN_KLD_v1_"+str(key)+"/"


dnn_aachen = DNN_Aachen.DNN(
    in_path             = inPath, 
    save_path           = outpath,		
    event_classes       = event_classes, 
    event_category      = categories[key],
    train_variables     = category_vars[key], 
    prenet_targets      = prenet_targets,
    train_epochs        = 500,
    early_stopping      = 10,
    eval_metrics        = ["acc"])#, "mean_squared_error", "mean_squared_logarithmic_error"])


dnn_aachen.build_model()
dnn_aachen.train_models()
dnn_aachen.eval_model()
dnn_aachen.plot_metrics()
dnn_aachen.plot_prenet_nodes(log = True)
dnn_aachen.plot_classification_nodes(log = True)
dnn_aachen.plot_confusion_matrix()
#dnn_aachen.plot_input_output_correlation()

