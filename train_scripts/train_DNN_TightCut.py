# global imports
import os
import sys

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

import DRACO_Frameworks.DNN.DNN as DNN
import variable_sets.topVariables_T as variable_set

JTcategory      = sys.argv[1]
variables       = variable_set.variables[JTcategory]

event_classes = ["ttHbb", "ttbb", "tt2b", "ttb", "ttcc", "ttlf"]

inPath   = "/ceph/vanderlinden/MLFoyTrainData/DNN/"
savepath = basedir+"/workdir/topVariablesTight_DNN_"+str(JTcategory)

dnn = DNN.DNN(
    in_path         = inPath,
    save_path       = savepath,
    event_classes   = event_classes,
    event_category  = JTcategory,
    train_variables = variables,
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
