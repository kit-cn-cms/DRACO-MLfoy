# global imports
import rootpy.plotting as rp
import numpy as np
import os
import sys
import socket

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

import DRACO_Frameworks.DNN.DNN as DNN
import variable_sets.top_10_variables as variable_set

category_vars = {
    "4j_ge3t": variable_set.variables_4j_ge3t,
    "5j_ge3t": variable_set.variables_5j_ge3t,
    "ge6j_ge3t": variable_set.variables_ge6j_ge3t
    }
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

key = sys.argv[1]

inPath   = workpath + "/AachenDNN_files"
savepath = workpath + "/top10_DNN_"+str(key)+"/"


dnn = DNN.DNN(
    in_path         = inPath,
    save_path       = savepath,
    event_classes   = event_classes,
    event_category  = categories[key],
    train_variables = category_vars[key],
    train_epochs    = 500,
    early_stopping  = 20,
    eval_metrics    = ["acc"],
    test_percentage = 0.2)

dnn.build_model()
dnn.train_model()
dnn.eval_model()
dnn.get_input_weights()
dnn.plot_metrics()
#dnn.plot_class_differences()
dnn.plot_discriminators()
dnn.plot_classification()
dnn.plot_confusion_matrix()
