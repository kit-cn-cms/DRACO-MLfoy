# python TEST_MODEL_LOAD.py -i /local/scratch/ssd/nshadskiy/2017_nominal -o comparison_densevariational_denseflipout -c ge4j_ge3t -v allVariables_2017_bnn -n BNN --binary -S ttH -q --restorefitdir /home/ycung/Desktop/DRACO-MLfoy/workdir/16-04-2020/QT_BNN_training_ge4j_ge3t/fit_data.csv
# global imports
# so that matplotlib can be used over ssh
import matplotlib #me
matplotlib.use('Agg') #me


import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import tqdm
from copy import deepcopy #me

import tensorflow.keras as keras #me
import tensorflow.keras.models as models #me
import tensorflow as tf #me
import tensorflow_probability as tfp    #me
import tensorflow_probability.python.distributions as tfd   #me

# option handler
import optionHandler
options = optionHandler.optionHandler(sys.argv)

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

# import class for BNN training
import DRACO_Frameworks.DNN.BNN as BNN
import DRACO_Frameworks.DNN.BNN_DenseFlipout as BNN_DP
import DRACO_Frameworks.DNN.data_frame as df

import plot_configs.setupPlots as setup
import utils.generateJTcut as JTcut


options.initArguments()

# load samples
input_samples = df.InputSamples(options.getInputDirectory(), options.getActivatedSamples(), options.getTestPercentage()) #+"_nominal"

# define all samples
input_samples.addSample(options.getDefaultName("ttH"), label = "ttH", normalization_weight = options.getNomWeight())
input_samples.addSample(options.getDefaultName("ttbb") , label = "ttbb" , normalization_weight = options.getNomWeight())
input_samples.addSample(options.getDefaultName("ttcc") , label = "ttcc" , normalization_weight = options.getNomWeight())
input_samples.addSample(options.getDefaultName("ttlf") , label = "ttlf" , normalization_weight = options.getNomWeight())

input_samples_binary = deepcopy(input_samples)

if options.isBinary():
    input_samples_binary.addBinaryLabel(options.getSignal(), options.getBinaryBkgTarget())

###### code for NN output comparison ######   
def get_column(array, i):
    return [row[i] for row in array]


def plot_correlation_two_NNs(NN_pred1, NN_pred2, NN_pred1_std, NN_pred2_std, x_lab, y_lab, save_dir, save_name):
    from matplotlib.colors import LogNorm
    plt.hist2d(NN_pred1, NN_pred2, bins=[50,50], cmin=1, norm=LogNorm())
    plt.plot([0,1],[0,1],'k')
    plt.colorbar()
    plt.xlabel("$\mu$ "+x_lab, fontsize = 16)
    plt.ylabel("$\mu$ "+y_lab, fontsize = 16)
    plt.savefig(save_dir+"/mu_{}.png".format(save_name))
    print "mu_{}.png was created".format(save_name)
    plt.savefig(save_dir+"/mu_{}.pdf".format(save_name))
    print "mu_{}.pdf was created".format(save_name)
    plt.close()

    n_max = np.amax([np.amax(NN_pred1_std), np.amax(NN_pred2_std)])
    plt.hist2d(NN_pred1_std, NN_pred2_std, bins=[50,50], range=[[0.,n_max],[0.,n_max]], cmin=1, norm=LogNorm())
    plt.plot([0,n_max],[0,n_max],'k')
    plt.colorbar()
    plt.xlabel("$\sigma$ "+x_lab, fontsize = 16)
    plt.ylabel("$\sigma$ "+y_lab, fontsize = 16)
    plt.savefig(save_dir+"/std_{}.png".format(save_name))
    print "std_{}.png was created".format(save_name)
    plt.savefig(save_dir+"/std_{}.pdf".format(save_name))
    print "std_{}.pdf was created".format(save_name)
    plt.close()


output_dir = options.getOutputDir()
work_dir = "/home/ycung/Desktop/DRACO-MLfoy/workdir/"

######################################################### initializing BNN/DNN training class ##################################################################
bnnDP_qt = BNN_DP.BNN_Flipout(
    save_path       = options.getOutputDir(),
    input_samples   = input_samples_binary, #changed
    category_name   = options.getCategory(),
    train_variables = options.getTrainVariables(),
    # number of epochs
    train_epochs    = options.getTrainEpochs(),
    # metrics for evaluation (c.f. KERAS metrics)
    eval_metrics    = ["acc"],
    # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
    test_percentage = options.getTestPercentage(),
    # balance samples per epoch such that there amount of samples per category is roughly equal
    balanceSamples  = options.doBalanceSamples(),
    shuffle_seed    = 42,
    evenSel         = options.doEvenSelection(),
    norm_variables  = options.doNormVariables(),
    qt_transformed_variables = options.doQTNormVariables(),
    restore_fit_dir = options.getRestoreFitDir(),
    sys_variation   = False,
    gen_vars        = False)

bnnDP = BNN_DP.BNN_Flipout(
    save_path       = options.getOutputDir(),
    input_samples   = input_samples_binary, #changed
    category_name   = options.getCategory(),
    train_variables = options.getTrainVariables(),
    # number of epochs
    train_epochs    = options.getTrainEpochs(),
    # metrics for evaluation (c.f. KERAS metrics)
    eval_metrics    = ["acc"],
    # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
    test_percentage = options.getTestPercentage(),
    # balance samples per epoch such that there amount of samples per category is roughly equal
    balanceSamples  = options.doBalanceSamples(),
    shuffle_seed    = 42,
    evenSel         = options.doEvenSelection(),
    norm_variables  = options.doNormVariables(),
    qt_transformed_variables = not options.doQTNormVariables(), #changed
    restore_fit_dir = None, #changed
    sys_variation   = False,
    gen_vars        = False)

bnn_qt = BNN.BNN(
    save_path       = options.getOutputDir(),
    input_samples   = input_samples_binary, #changed
    category_name   = options.getCategory(),
    train_variables = options.getTrainVariables(),
    # number of epochs
    train_epochs    = options.getTrainEpochs(),
    # metrics for evaluation (c.f. KERAS metrics)
    eval_metrics    = ["acc"],
    # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
    test_percentage = options.getTestPercentage(),
    # balance samples per epoch such that there amount of samples per category is roughly equal
    balanceSamples  = options.doBalanceSamples(),
    shuffle_seed    = 42,
    evenSel         = options.doEvenSelection(),
    norm_variables  = options.doNormVariables(),
    qt_transformed_variables = options.doQTNormVariables(),
    restore_fit_dir = options.getRestoreFitDir(),
    sys_variation   = False,
    gen_vars        = False)

bnn = BNN.BNN(
    save_path       = options.getOutputDir(),
    input_samples   = input_samples_binary, #changed
    category_name   = options.getCategory(),
    train_variables = options.getTrainVariables(),
    # number of epochs
    train_epochs    = options.getTrainEpochs(),
    # metrics for evaluation (c.f. KERAS metrics)
    eval_metrics    = ["acc"],
    # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
    test_percentage = options.getTestPercentage(),
    # balance samples per epoch such that there amount of samples per category is roughly equal
    balanceSamples  = options.doBalanceSamples(),
    shuffle_seed    = 42,
    evenSel         = options.doEvenSelection(),
    norm_variables  = options.doNormVariables(),
    qt_transformed_variables = not options.doQTNormVariables(), #changed
    restore_fit_dir = None, #changed
    sys_variation   = False,
    gen_vars        = False)
########################################################################################################################

event_classes = ["ttH", "ttbb", "ttcc", "ttlf"]
n_iterations = 100

input_dir_1 = work_dir+"Flipout_QT_BNN_training_50_ge4j_ge3t"
input_dir_2 = work_dir+"16-04-2020/QT_BNN_training_ge4j_ge3t"
input_dir_3 = work_dir+"Flipout_BNN_training_50_ge4j_ge3t"
input_dir_4 = work_dir+"16-04-2020/BNN_training_ge4j_ge3t"



nn1_pred, nn1_pred_std, labels1 = bnnDP_qt.load_trained_model(options.getNetConfig(),input_dir_1, n_iterations)
nn2_pred, nn2_pred_std, labels2 = bnn_qt.load_trained_model(input_dir_2, n_iterations)
nn3_pred, nn3_pred_std, labels3 = bnnDP.load_trained_model(options.getNetConfig(),input_dir_3, n_iterations)
nn4_pred, nn4_pred_std, labels4 = bnn.load_trained_model(input_dir_4, n_iterations)


plot_correlation_two_NNs(nn2_pred, nn1_pred, nn2_pred_std, nn1_pred_std, "binary_BNN_QT_DenseVariational", "binary_BNN_QT_DenseFlipout", output_dir, "BNN_QT_BNN_QT_Flipout_comparison")
plot_correlation_two_NNs(nn4_pred, nn3_pred, nn4_pred_std, nn3_pred_std, "binary_BNN_DenseVariational", "binary_BNN_DenseFlipout", output_dir, "BNN_BNN_Flipout_comparison")

