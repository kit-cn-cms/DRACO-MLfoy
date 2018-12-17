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
import DRACO_Frameworks.DNN.data_frame as data_frame
import variable_sets.top_10_variables as variable_set

# category
key = sys.argv[1]
category_vars = {
    "4j_ge3t": variable_set.variables_4j_ge3t,
    "5j_ge3t": variable_set.variables_5j_ge3t,
    "ge6j_ge3t": variable_set.variables_ge6j_ge3t}            
categories = {
    "4j_ge3t":   "(N_Jets == 4 and N_BTagsM >= 3)",
    "5j_ge3t":   "(N_Jets == 5 and N_BTagsM >= 3)",
    "ge6j_ge3t": "(N_Jets >= 6 and N_BTagsM >= 3)",
    }


event_classes = ["ttHbb", "ttbb", "tt2b", "ttb", "ttcc", "ttlf"]

if "naf" in socket.gethostname():
    workpath = "/nfs/dust/cms/user/vdlinden/DRACO-MLfoy/workdir/"
else:
    workpath = "/ceph/vanderlinden/DRACO-MLfoy/workdir/"


# path to input data files
inPath = workpath+"/AachenDNN_files"
# output path in workdir
outpath = workpath+"/top10_DNN_"+str(key)+"/"

dnn = DNN.DNN(
    in_path             = inPath,
    save_path           = outpath,
    event_classes       = event_classes,
    event_category      = categories[key],
    train_variables     = category_vars[key],
    additional_cut      = None)

dnn.load_trained_model()
dnn.predict_event_query("(Evt_ID == 7230872)")
dnn.predict_event_query("(Evt_ID == 7230984)")
dnn.predict_event_query("(Evt_ID == 7231382)")
dnn.predict_event_query("(Evt_ID == 7231690)")
#dnn.plot_class_differences()
#dnn.plot_discriminators()
#dnn.plot_classification()
#dnn.plot_confusion_matrix()
#dnn.plot_output_output_correlation(plot=True)
#dnn.plot_input_output_correlation(plot=False)

