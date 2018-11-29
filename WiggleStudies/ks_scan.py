# global imports
import numpy as np
import rootpy.plotting as rp
import os
import sys
import socket
import matplotlib.pyplot as plt
from matplotlib import gridspec
import ROOT
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

import math
import DRACO_Frameworks.DNN_Aachen.DNN_Aachen as DNN_Aachen
import DRACO_Frameworks.DNN_Aachen.variable_info as variable_info
import DRACO_Frameworks.DNN_Aachen.data_frame as data_frame
import pyrootsOfTheCaribbean.plot_configs.plotting_styles as pltstyle

category_vars = {
    "4j_ge3t": variable_info.variables_4j_3b,
    "4j_4t": variable_info.variables_4j_3b,
    "5j_ge3t": variable_info.variables_5j_3b,
    "ge6j_ge3t": variable_info.variables_6j_3b}            
categories = {
    "4j_ge3t":   "(N_Jets == 4 and N_BTagsM >= 3)",
    "4j_4t":   "(N_Jets == 4 and N_BTagsM == 4)",
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

try:
    key = sys.argv[1]
    smearing = eval(sys.argv[2])
    if smearing: print("doing smearing (x->x+rnd.normal(0,sigma))")
    else:        print("doing scaling  (x->x*rnd.normal(1,sigma))")
except:
    print("first  argument: JT cagegory")
    print("second argument: smearing/scaling (1/0)")
    exit()

outpath = workpath+"/AachenDNN_v2_"+str(key)+"/"
checkpoint_path = outpath + "/checkpoints/trained_main_net.h5py"

result_dir = "/ceph/vanderlinden/DRACO-MLfoy/WiggleStudies/results/KSscans"
if smearing:    result_dir += "_smearing/"
else:           result_dir += "_scaling/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
result_dir += "/"+str(key)+"/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

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
    additional_cut      = None)


def gen_discrs( predictions ):
    discr_nodes = [[] for _ in event_classes]
    for evt in predictions:
        pred_class = np.argmax(evt)
        pred_value = evt[pred_class]
        discr_nodes[pred_class].append(pred_value)

    return discr_nodes



dnn_aachen.load_trained_model()
data = dnn_aachen.data.get_test_data(as_matrix=False)
prediction_before = dnn_aachen.main_net.predict(data.values)

discriminators_before = gen_discrs(prediction_before)
before_hists = []

bins = 100
bin_range = [0.,1.]
for i_node in range(len(event_classes)):
    node_values = discriminators_before[i_node]
    h = rp.Hist(bins, *bin_range, title = "before smearing")
    h.markersize = 0
    h.legendstyle = "F"
    h.fillstyle = "solid"
    h.linecolor = "black"
    h.fill_array( node_values )
    before_hists.append(h)

# generate loop over different std deviation
stddevs = np.arange(0.005,0.305,0.01)
print(stddevs)
#np.arange(0.01,0.31,0.01)

rate_of_other_argmax = []
mean_diff = []
std_diff = []

ks_per_node = [[] for _ in event_classes]
ks_std_per_node = [[] for _ in event_classes]
n_samples = 20
for sigma in stddevs:
    print("at sigma "+str(sigma))
    # apply some uncertainties to data
    def func(x):
        if smearing: return x + np.random.normal(0,sigma)
        else:        return x*np.random.normal(1,sigma)

    ks_values = [[] for _ in event_classes]
    for n_iter in range(n_samples):
        # wiggle data
        data_new = data.applymap(func)

        # generate new predictions
        prediction_after = dnn_aachen.main_net.predict(data_new.values)

        discriminators_after = gen_discrs(prediction_after)

        for i_node in range(len(event_classes)):
            node_values = discriminators_after[i_node]
            new_h = rp.Hist(bins,*bin_range, title = "after smearing (s = {:.4f})".format(sigma))
            new_h.fill_array( node_values )

            ks_prob = before_hists[i_node].KolmogorovTest(new_h)#,"N")
            ks_values[i_node].append(ks_prob)

    for i_node in range(len(event_classes)):
        ks_per_node[i_node].append(np.mean(ks_values[i_node]))
        ks_std_per_node[i_node].append(np.std(ks_values[i_node]))

for i_node in range(len(event_classes)):
    print("plotting hist for node {}".format(i_node))
    plt.clf()
    plt.figure(figsize = [7,5])
    
    plt.errorbar( stddevs, ks_per_node[i_node], yerr = ks_std_per_node[i_node], fmt = "o", color = "black")    

    plt.xlabel("smearing factor s")
    plt.ylabel("KS prob")
    plt.title(str(event_classes[i_node])+" node", loc = "left")
    plt.title(categories[key], loc = "right")
    save_dir = result_dir +"/KS_scan_{}_node.pdf".format( event_classes[i_node])
    plt.savefig(save_dir )
    print("saved plot at {}".format(save_dir))
