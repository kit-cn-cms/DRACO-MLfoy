# global imports
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)
import DRACO_Frameworks.LSTM.LSTM as LSTM


inPath = "/storage/c/vanderlinden/DRACO-MLfoy/workdir/train_samples/pfc_train_set"

lstm = LSTM.LSTM(
    in_path         = inPath,
    save_path       = basedir+"/workdir/lstm_test",
    class_label     = "class_label",
    input_variables = ["E", "pT", "phi", "eta", "M2"],
    n_particles     = 50,
    normed_inputs   = True,
    batch_size      = 256,
    train_epochs    = 20,
    early_stopping  = 5,
    optimizer       = "adam",
    loss_function   = "categorical_crossentropy",
    eval_metrics    = ["mean_absolute_error", "mean_squared_error"] )

lstm.load_datasets()
lstm.build_model()
lstm.train_model()
lstm.eval_model()

# evaluate stuff
lstm.print_classification_report()
lstm.plot_metrics()
lstm.plot_discriminators()
lstm.plot_confusion_matrix()







