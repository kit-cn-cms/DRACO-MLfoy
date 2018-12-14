# global imports
import numpy as np
import os
import sys
import socket
import matplotlib.pyplot as plt
from matplotlib import gridspec
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

import math
import DRACO_Frameworks.DNN_Aachen.DNN_Aachen as DNN_Aachen
import DRACO_Frameworks.DNN_Aachen.variable_info as variable_info
import DRACO_Frameworks.DNN_Aachen.data_frame as data_frame

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

key = sys.argv[1]

outpath = workpath+"/AachenDNN_v2_"+str(key)+"/"
checkpoint_path = outpath + "/checkpoints/trained_main_net.h5py"

result_dir = "/ceph/vanderlinden/DRACO-MLfoy/WiggleStudies/results/fullWiggle/"
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

dnn_aachen.load_trained_model()
data = dnn_aachen.data.get_test_data(as_matrix=False)
prediction_before = dnn_aachen.main_net.predict(data.values)

# generate loop over different std deviation
stddevs = np.arange(0.01,0.31,0.01)

rate_of_other_argmax = []
mean_diff = []
std_diff = []

plt.figure(figsize = [10,5])
for sigma in stddevs:
    print("at sigma "+str(sigma))
    # apply some uncertainties to data
    def func(x):
        return x + np.random.normal(0,sigma)

    # wiggle data
    data_new = data.applymap(func)

    # generate new predictions
    prediction_after = dnn_aachen.main_net.predict(data_new.values)

    # lists for counting stuff
    absolute_differences_unchanged = []
    absolute_differences_changed = []
    largest_value_unchanged = []
    largest_value_changed = []
    other_pred = 0

    for i in range(len(prediction_before)):
        diff = 0
        for j in range(len(prediction_before[i])):
            diff += np.abs( prediction_before[i][j]-prediction_after[i][j] )
        
        max_val = prediction_after[i][ np.argmax(prediction_after[i]) ]

        if np.argmax( prediction_before[i] ) == np.argmax( prediction_after[i] ):
            # prediction was not changed
            absolute_differences_unchanged.append( diff )
            largest_value_unchanged.append( max_val )

        else:
            # prediction was changed
            other_pred += 1
            absolute_differences_changed.append( diff )
            largest_value_changed.append( max_val )


    mean_diff.append( np.mean( absolute_differences_changed + absolute_differences_unchanged ) )
    std_diff.append(  np.std( absolute_differences_changed + absolute_differences_unchanged ) )
    rate_of_other_argmax.append( 1.*other_pred/len(prediction_before) )

    save_dir = result_dir + "/wiggle_{:.3f}".format(sigma)
    
    # generate plots
    # DIFF PLOT
    fig = plt.figure(figsize = [10,5])
    gs = gridspec.GridSpec(2,1, height_ratios = [3,1], hspace = 0.)    

    ax0 = plt.subplot( gs[0])
    ax1 = plt.subplot( gs[1])

    # histogram 
    bin_range = [0.,0.3]
    n,bins,patches = ax0.hist([absolute_differences_unchanged, absolute_differences_changed], stacked = True, 
        label = ["abs diff (argmax unchanged)", "abs diff (argmax changed)"],
        bins = 20, range = bin_range)
    ax0.set_ylabel("Events")
    ax0.set_title("wiggle factor {:.5f}".format(sigma), loc = "left")
    ax0.set_title("{}/{} events with changed argmax".format(other_pred, len(prediction_after)), loc = "right")
    ax0.set_xlim(bin_range)
    ax0.set_xticks([], minor = False)
    ax0.legend()

    # ratio plot
    rates = n[0]/n[1]
    rates = [0 if math.isnan(r) else r for r in rates]
    b_width = bins[1]-bins[0]
    bins = bins+b_width/2.
    bins = list(bins)[:-1]
    
    ax1.plot(bins,rates, "o", color = "black")
    ax1.set_ylabel("unchanged ratio")
    ax1.set_xlabel("absolute difference of all classes per event")
    ax1.set_xlim(bin_range)
    ax1.set_ylim([0.,1.])
    plt.tight_layout()
    plt.savefig(save_dir + "_prediction_difference.pdf")
    plt.clf()
    


    # ARGMAX PLOT    
    bin_range = [0.15,1.]
    fig = plt.figure(figsize = [10,5])
    ax0 = plt.subplot( gs[0] )
    ax1 = plt.subplot( gs[1] )
    
    # histogram
    n, bins, patches = ax0.hist([largest_value_unchanged, largest_value_changed], stacked = True, 
        label = ["argmax (unchanged)", "argmax (changed)"],
        bins = 20, range = bin_range)

    ax0.set_xlim(bin_range)
    ax0.set_xticks([], minor = False)
    ax0.set_ylabel("Events")
    ax0.set_title("wiggle factor {:.5f}".format(sigma), loc = "left")
    ax0.set_title("{}/{} events with changed argmax".format(other_pred, len(prediction_after)), loc = "right")
    ax0.legend()

    # ratio plot
    rates = n[0]/n[1]
    rates = [0 if math.isnan(r) else r for r in rates]
    b_width = bins[1]-bins[0]
    bins = bins+b_width/2
    bins = list(bins)[:-1]

    ax1.plot(bins, rates, "o", color = "black")
    ax1.set_ylabel("unchanged ratio")
    ax1.set_xlabel("argmax per event")
    ax1.set_xlim(bin_range)
    ax1.set_ylim([0.,1.])
    plt.tight_layout()
    plt.savefig(save_dir + "_argmax_values.pdf")    
    plt.clf()


# DIFF PLOT
plt.errorbar( stddevs, mean_diff, yerr = std_diff, fmt = "o")
plt.xlabel("wiggle factor")
plt.ylabel("mean of absolute difference in prediction")
plt.title("category "+str(key), loc = "right")
plt.savefig( result_dir + "absoulte_differences.pdf")
plt.clf()

# ARGMAX PLOT
plt.plot( stddevs, rate_of_other_argmax, "o")
plt.xlabel("wiggle factor")
plt.ylabel("rate of changed argmax node")
plt.title("category "+str(key), loc = "right")
plt.savefig( result_dir + "changed_argmax_rate.pdf")
plt.clf()
