# global imports
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)
import DRACO_Frameworks.AutoEncoder.AutoEncoder as AutoEncoder


inPath = "/storage/c/vanderlinden/DRACO-MLfoy/workdir/train_samples/base_train_set"

ae = AutoEncoder.AutoEncoder(
    in_path         = inPath,
    save_path       = basedir+"/workdir/auto_encoder_test",
    batch_size      = 256,
    train_epochs    = 10,
    optimizer       = "adam",
    loss_function   = "mean_squared_error",
    eval_metrics    = ["mean_absolute_error", "acc"] )

ae.load_datasets()
ae.build_model()
ae.train_model()
ae.eval_model()

# evaluate stuff
ae.print_classification_examples(10)
ae.plot_metrics()







