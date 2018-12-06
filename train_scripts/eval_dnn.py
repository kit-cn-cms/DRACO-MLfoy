# global imports
import numpy as np
import os
import sys
import socket

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

import DRACO_Frameworks.DNN_Aachen.DNN_Aachen as DNN_Aachen
import DRACO_Frameworks.DNN_Aachen.variable_info as variable_info
import DRACO_Frameworks.DNN_Aachen.data_frame as data_frame

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
    "GenTopHad_B_inacceptance",
    "GenTopHad_QQ_inacceptance",
    "GenTopHad_Q_inacceptance",
    "GenTopLep_B_inacceptance"
    ]

event_classes = ["ttHbb", "ttbb", "tt2b", "ttb", "ttcc", "ttlf"]

if "naf" in socket.gethostname():
    workpath = "/nfs/dust/cms/user/vdlinden/DRACO-MLfoy/workdir/"
else:
    workpath = "/ceph/vanderlinden/DRACO-MLfoy/workdir/"


inPath = workpath+"/AachenDNN_files"

key = sys.argv[1]

workpath = "/ceph/vanderlinden/DRACO-MLfoy/train_scripts/Aachen_DNN_checkpoints/"
outpath = workpath+"/"+str(key)+"/"
checkpoint_path = outpath + "/checkpoints/trained_main_net.h5py"

dnn_aachen = DNN_Aachen.DNN(
    in_path             = inPath,
    save_path           = outpath,
    event_classes       = event_classes,
    event_category      = categories[key],
    train_variables     = category_vars[key],
    prenet_targets      = prenet_targets,
    train_epochs        = 500,
    early_stopping      = 20,
    eval_metrics        = ["acc"],
    additional_cut      = None)#"N_BTagsM == 3")

dnn_aachen.load_trained_model()
#dnn_aachen.predict_event_query("(Evt_ID == 16706929)")
#dnn_aachen.plot_class_differences()
dnn_aachen.plot_discriminators()
dnn_aachen.plot_classification()
#dnn_aachen.plot_confusion_matrix()
#dnn_aachen.plot_output_output_correlation(plot=True)
#dnn_aachen.plot_input_output_correlation(plot=False)

'''
# search for 4j4t events with ttlf predictions
data = dnn_aachen.data.get_full_df()
for i, row in data.iterrows():
    sample = np.array([list(row.values)])
    ev = dnn_aachen.main_net.predict( sample )
    if np.argmax(ev) == 5:
        print("IDs:" +str(i))
        print("evaulation:")
        print(ev)
        print("-------------------------------------------------------")
'''
