# python evaluating_misc_models.py -i "/local/scratch/ssd/ycung/new_h5_files_2017_v2" -o comparison_misc_models -c ge4j_ge3t -v new_variableset_25_v2 --binary -S ttH -q -P --restorefitdir /home/ycung/Desktop/DRACO-MLfoy_thesis/workdir/training_ANN_150N/QT_ANN_150N_training_0_ge4j_ge3t/fit_data.pck
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

import csv
import datetime
options.initArguments()

# load samples
input_samples = df.InputSamples(options.getInputDirectory(), options.getActivatedSamples(), options.getTestPercentage()) #+"_nominal"

# define all samples
input_samples.addSample(options.getDefaultName("ttH"), label = "ttH", normalization_weight = options.getNomWeight())
input_samples.addSample(options.getDefaultName("tthf") , label = "tthf" , normalization_weight = options.getNomWeight())
input_samples.addSample(options.getDefaultName("ttcc") , label = "ttcc" , normalization_weight = options.getNomWeight())
input_samples.addSample(options.getDefaultName("ttlf") , label = "ttlf" , normalization_weight = options.getNomWeight())

input_samples_binary = deepcopy(input_samples)

if options.isBinary():
    input_samples_binary.addBinaryLabel(options.getSignal(), options.getBinaryBkgTarget())

###### code for NN output comparison ######   
def get_column(array, i):
    return [row[i] for row in array]

def plot_correlation_two_NNs(NN_pred1, NN_pred2, NN_pred1_std, NN_pred2_std, x_lab, y_lab, save_dir, save_name, privateWork = False):
    from matplotlib.colors import LogNorm
    plt.rc('xtick',labelsize=14)
    plt.rc('ytick',labelsize=14)

    plt.hist2d(NN_pred1, NN_pred2, bins=[50,50], cmin=1, norm=LogNorm())
    plt.plot([0,1],[0,1],'k')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.xlabel(x_lab+" Ausgabewert $\mu$", fontsize = 16)
    plt.ylabel(y_lab+" Ausgabewert $\mu$ ", fontsize = 16)
    if privateWork:
        plt.title(r"$\bf{CMS\ private\ work}$", loc = "left", fontsize = 14)
    plt.tight_layout()
    plt.savefig(save_dir+"/mu_{}.png".format(save_name))
    print "mu_{}.png was created".format(save_name)
    plt.savefig(save_dir+"/mu_{}.pdf".format(save_name))
    print "mu_{}.pdf was created".format(save_name)
    plt.close()

    plt.rc('xtick',labelsize=14)
    plt.rc('ytick',labelsize=14)

    n_max = np.amax([np.amax(NN_pred1_std), np.amax(NN_pred2_std)])
    plt.hist2d(NN_pred1_std, NN_pred2_std, bins=[50,50], range=[[0.,n_max],[0.,n_max]], cmin=1, norm=LogNorm())
    plt.plot([0,n_max],[0,n_max],'k')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.xlabel(x_lab+" Standardabweichung $\sigma$", fontsize = 16)
    plt.ylabel(y_lab+" Standardabweichung $\sigma$ ", fontsize = 16)
    if privateWork:
        plt.title(r"$\bf{CMS\ private\ work}$", loc = "left", fontsize = 14)
    plt.tight_layout()
    plt.savefig(save_dir+"/std_{}.png".format(save_name))
    print "std_{}.png was created".format(save_name)
    plt.savefig(save_dir+"/std_{}.pdf".format(save_name))
    print "std_{}.pdf was created".format(save_name)
    plt.close()
    
output_dir = options.getOutputDir()
work_dir = basedir+"/workdir/"

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

n_iterations = 100

input_dir_1  = work_dir+"Var_BNN_training_150_ge4j_ge3t"
input_dir_2  = work_dir+"Var_QT_BNN_training_150_ge4j_ge3t"
# input_dir_3  = work_dir+"modified_Flipout_V1_BNN_training_150_ge4j_ge3t"
# input_dir_4  = work_dir+"modified_Flipout_V1_QT_BNN_training_150_ge4j_ge3t"


nn1_pred, nn1_pred_std, labels1 = bnn.load_trained_model(input_dir_1, n_iterations, options.getOutputDir())
nn2_pred, nn2_pred_std, labels2 = bnn_qt.load_trained_model(input_dir_2, n_iterations, options.getOutputDir())
# nn3_pred, nn3_pred_std, labels3 = bnnDP.load_trained_model(input_dir_3, n_iterations, options.getOutputDir())
# nn4_pred, nn4_pred_std, labels4 = bnnDP_qt.load_trained_model(input_dir_4, n_iterations, options.getOutputDir())


# # Comparison with and without QT
plot_correlation_two_NNs(nn1_pred, nn2_pred, nn1_pred_std, nn2_pred_std, "(BNN)" + r'$^{\rm{DV}}_{1 \times 150}$', "(BNN)" + r'$^{\rm{DV,\ T}}_{1 \times 150}$', output_dir, "N_BNN_Var_150N_comparison", privateWork=options.isPrivateWork())
# plot_correlation_two_NNs(nn3_pred, nn4_pred, nn3_pred_std, nn4_pred_std, "(BNN)" + r'$^{\rm{DF}}_{1 \times 150}$', "(BNN)" + r'$^{\rm{DF,\ T}}_{1 \times 150}$', output_dir, "BNN_mod_Flip_V1_150N_comparison", privateWork=options.isPrivateWork())

#####################################################################################################################################
######################################################### Comparison 200N ###########################################################
#####################################################################################################################################
# input_dir_1  = work_dir+"Var_BNN_training_200_ge4j_ge3t"
# input_dir_2  = work_dir+"Var_QT_BNN_training_200_ge4j_ge3t"
# input_dir_3  = work_dir+"modified_Flipout_V1_BNN_training_200_ge4j_ge3t"
# input_dir_4  = work_dir+"modified_Flipout_V1_QT_BNN_training_200_ge4j_ge3t"


# nn1_pred, nn1_pred_std, labels1 = bnn.load_trained_model(input_dir_1, n_iterations, options.getOutputDir())
# nn2_pred, nn2_pred_std, labels2 = bnn_qt.load_trained_model(input_dir_2, n_iterations, options.getOutputDir())
# nn3_pred, nn3_pred_std, labels3 = bnnDP.load_trained_model(input_dir_3, n_iterations, options.getOutputDir())
# nn4_pred, nn4_pred_std, labels4 = bnnDP_qt.load_trained_model(input_dir_4, n_iterations, options.getOutputDir())


# # Comparison with and without QT
# plot_correlation_two_NNs(nn1_pred, nn2_pred, nn1_pred_std, nn2_pred_std, "(BNN)" + r'$^{\rm{DV}}_{1 \times 200}$', "(BNN)" + r'$^{\rm{DV,\ T}}_{1 \times 200}$', output_dir, "BNN_Var_200N_comparison", privateWork=options.isPrivateWork())
# plot_correlation_two_NNs(nn3_pred, nn4_pred, nn3_pred_std, nn4_pred_std, "(BNN)" + r'$^{\rm{DF}}_{1 \times 200}$', "(BNN)" + r'$^{\rm{DF,\ T}}_{1 \times 200}$', output_dir, "BNN_mod_Flip_V1_200N_comparison", privateWork=options.isPrivateWork())

# #####################################################################################################################################
# ######################################################### Comparison 100N ###########################################################
# #####################################################################################################################################

# input_dir_1  = work_dir+"Var_BNN_training_100_ge4j_ge3t"
# input_dir_2  = work_dir+"Var_QT_BNN_training_100_ge4j_ge3t"
# input_dir_3  = work_dir+"modified_Flipout_V1_BNN_training_100_ge4j_ge3t"
# input_dir_4  = work_dir+"modified_Flipout_V1_QT_BNN_training_100_ge4j_ge3t"


# nn1_pred, nn1_pred_std, labels1 = bnn.load_trained_model(input_dir_1, n_iterations, options.getOutputDir())
# nn2_pred, nn2_pred_std, labels2 = bnn_qt.load_trained_model(input_dir_2, n_iterations, options.getOutputDir())
# # nn3_pred, nn3_pred_std, labels3 = bnnDP.load_trained_model(input_dir_3, n_iterations, options.getOutputDir())
# # nn4_pred, nn4_pred_std, labels4 = bnnDP_qt.load_trained_model(input_dir_4, n_iterations, options.getOutputDir())


# # Comparison with and without QT
# plot_correlation_two_NNs(nn1_pred, nn2_pred, nn1_pred_std, nn2_pred_std, "(BNN)" + r'$^{\rm{DV}}_{1 \times 100}$', "(BNN)" + r'$^{\rm{DV,\ T}}_{1 \times 100}$', output_dir, "BNN_Var_100N_comparison", privateWork=options.isPrivateWork())
# plot_correlation_two_NNs(nn3_pred, nn4_pred, nn3_pred_std, nn4_pred_std, "(BNN)" + r'$^{\rm{DF}}_{1 \times 100}$', "(BNN)" + r'$^{\rm{DF,\ T}}_{1 \times 100}$', output_dir, "BNN_mod_Flip_V1_100N_comparison", privateWork=options.isPrivateWork())

