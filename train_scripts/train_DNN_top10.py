# global imports
import os
import sys

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

import DRACO_Frameworks.DNN.DNN as DNN
import variable_sets.top_10_variables as variable_set

key = sys.argv[1]
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


inPath   = "/ceph/vanderlinden/MLFoyTrainData/DNN/"
savepath = basedir+"/workdir/top10_DNN_"+str(key)

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

# plotting 
dnn.plot_confusionMatrix(norm_matrix = True)
dnn.plot_outputNodes()
dnn.plot_discriminators()
