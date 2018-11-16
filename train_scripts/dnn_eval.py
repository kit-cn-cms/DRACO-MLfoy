import keras
from keras.models import load_model
import os
import sys
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import numpy as np
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)


import DRACO_Frameworks.DNN_Aachen.variable_info as variable_info
import DRACO_Frameworks.DNN_Aachen.data_frame as data_frame
import DRACO_Frameworks.DNN_Aachen.DNN_Aachen as DNN_Aachen
categories_dic = {
    "(N_Jets == 4 and N_BTagsM >= 3)": [variable_info.variables_4j_3b, "4j_ge3t"],
    "(N_Jets == 5 and N_BTagsM >= 3)": [variable_info.variables_5j_3b, "5j_ge3t"],
    "(N_Jets >= 6 and N_BTagsM >= 3)": [variable_info.variables_6j_3b, "ge6j_ge3t"]}


prenet_targets = [
    #"GenAdd_BB_inacceptance",
    "GenHiggs_BB_inacceptance",
    "GenHiggs_B_inacceptance",
    #"GenTopHad_B_inacceptance",
    #"GenTopHad_QQ_inacceptance",
    #"GenTopHad_Q_inacceptance",
    #"GenTopLep_B_inacceptance"
    ]


data = data_frame.DataFrame(
            path_to_input_files = "/nfs/dust/cms/user/vdlinden/DRACO-MLfoy/workdir/AachenDNN_files/",
            classes             = ["ttHbb", "ttbb", "tt2b", "ttb", "ttcc", "ttlf"],
            event_category      =  "(N_Jets >= 6 and N_BTagsM >= 3)",
            train_variables     = categories_dic["(N_Jets >= 6 and N_BTagsM >= 3)"][0],
            prenet_targets      = prenet_targets,
            test_percentage     = 0.2,
            norm_variables      = True)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

model = load_model('/nfs/dust/cms/user/vdlinden/DRACO-MLfoy/workdir/AachenDNN_KLD_v1_ge6j_ge3t/checkpoints/trained_main_net.h5py')
#print(data.get_full_df())
x = data.get_full_df()
runs = [1,1,1,1,1,1,1,1,1,1,1,1]
lumi = [3545, 3545, 3545, 3545, 3545, 5357, 5357, 5357, 5358, 3553, 3553, 3553]
ID = [2114554, 2114596, 2114608, 2114680, 2114818, 3194896, 3195426, 3195558, 3196028, 2118816, 2118824, 2119080]

fake_pred = np.array([ [0.]*x.shape[1] ])
print(fake_pred)
y = model.predict(fake_pred)
print(y)
for r,l,i in zip(runs, lumi, ID):
    event = "(Evt_Run == {} and Evt_Lumi == {} and Evt_ID == {})".format(r,l,i)
    print(event)
    cut = x.query(event)#"Evt_Lumi == 3545")

    y = model.predict(cut.as_matrix())
    #print(cut.as_matrix())
    #print("\n\n")
    print(y)
    print("-"*20)
