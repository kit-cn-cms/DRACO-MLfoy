# python evaluating_models.py -i "/local/scratch/ssd/ycung/new_h5_files_2017_v2" -o comparison_modified_models -c ge4j_ge3t -v new_variableset_25_v2 --binary -S ttH -q -P --restorefitdir /home/ycung/Desktop/DRACO-MLfoy_thesis/workdir/training_ANN_150N/QT_ANN_150N_training_0_ge4j_ge3t/fit_data.pck
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

input_dir_1  = work_dir+"Var_BNN_training_50_ge4j_ge3t"
input_dir_2  = work_dir+"Var_QT_BNN_training_50_ge4j_ge3t"
input_dir_3  = work_dir+"Flipout_BNN_training_50_ge4j_ge3t"
input_dir_4  = work_dir+"Flipout_QT_BNN_training_50_ge4j_ge3t"
input_dir_5  = work_dir+"modified_Flipout_V1_BNN_training_50_ge4j_ge3t"
input_dir_6  = work_dir+"modified_Flipout_V1_QT_BNN_training_50_ge4j_ge3t"
input_dir_7  = work_dir+"modified_Flipout_V3_BNN_training_50_ge4j_ge3t"
input_dir_8  = work_dir+"modified_Flipout_V3_QT_BNN_training_50_ge4j_ge3t"
input_dir_9  = work_dir+"modified_Var_V1_BNN_training_50_ge4j_ge3t"
input_dir_10 = work_dir+"modified_Var_V1_QT_BNN_training_50_ge4j_ge3t"
input_dir_11 = work_dir+"modified_Var_V2_BNN_training_50_ge4j_ge3t"
input_dir_12 = work_dir+"modified_Var_V2_QT_BNN_training_50_ge4j_ge3t"



nn1_pred, nn1_pred_std, labels1 = bnn.load_trained_model(input_dir_1, n_iterations, options.getOutputDir())
nn2_pred, nn2_pred_std, labels2 = bnn_qt.load_trained_model(input_dir_2, n_iterations, options.getOutputDir())
nn3_pred, nn3_pred_std, labels3 = bnnDP.load_trained_model(input_dir_3, n_iterations, options.getOutputDir())
nn4_pred, nn4_pred_std, labels4 = bnnDP_qt.load_trained_model(input_dir_4, n_iterations, options.getOutputDir())
nn5_pred, nn5_pred_std, labels5 = bnnDP.load_trained_model(input_dir_5, n_iterations, options.getOutputDir())
nn6_pred, nn6_pred_std, labels6 = bnnDP_qt.load_trained_model(input_dir_6, n_iterations, options.getOutputDir())
nn7_pred, nn7_pred_std, labels7 = bnnDP.load_trained_model(input_dir_7, n_iterations, options.getOutputDir())
nn8_pred, nn8_pred_std, labels8 = bnnDP_qt.load_trained_model(input_dir_8, n_iterations, options.getOutputDir())
nn9_pred, nn9_pred_std, labels9 = bnn.load_trained_model(input_dir_9, n_iterations, options.getOutputDir())
nn10_pred, nn10_pred_std, labels10 = bnn_qt.load_trained_model(input_dir_10, n_iterations, options.getOutputDir())
nn11_pred, nn11_pred_std, labels11 = bnn.load_trained_model(input_dir_11, n_iterations, options.getOutputDir())
nn12_pred, nn12_pred_std, labels12 = bnn_qt.load_trained_model(input_dir_12, n_iterations, options.getOutputDir())



# Comparison with and without QT
plot_correlation_two_NNs(nn1_pred, nn2_pred, nn1_pred_std, nn2_pred_std, "BNN" + r'$^{\rm{DV}}_{1 \times 50}$', "BNN" + r'$^{\rm{DV,T}}_{1 \times 50}$', output_dir, "BNN_Var_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn3_pred, nn4_pred, nn3_pred_std, nn4_pred_std, "BNN" + r'$^{\rm{DP}}_{1 \times 50}$', "BNN" + r'$^{\rm{DP,T}}_{1 \times 50}$', output_dir, "BNN_Flip_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn5_pred, nn6_pred, nn5_pred_std, nn6_pred_std, "BNN" + r'$^{\rm{DP-Mod 1}}_{1 \times 50}$', "BNN" + r'$^{\rm{DP-Mod 1,T}}_{1 \times 50}$', output_dir, "modV1_BNN_Flip_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn7_pred, nn8_pred, nn7_pred_std, nn8_pred_std, "BNN" + r'$^{\rm{DP-Mod 2}}_{1 \times 50}$', "BNN" + r'$^{\rm{DP-Mod 2,T}}_{1 \times 50}$', output_dir, "modV3_BNN_Flip_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn9_pred, nn10_pred, nn9_pred_std, nn10_pred_std, "BNN" + r'$^{\rm{DV-Mod A}}_{1 \times 50}$', "BNN" + r'$^{\rm{DV-Mod A,T}}_{1 \times 50}$', output_dir, "modV1_BNN_Var_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn11_pred, nn12_pred, nn11_pred_std, nn12_pred_std, "BNN" + r'$^{\rm{DV-Mod A}}_{1 \times 50}$', "BNN" + r'$^{\rm{DV-Mod B,T}}_{1 \times 50}$', output_dir, "modV2_BNN_Var_comparison", privateWork=options.isPrivateWork())

# Comparison Default Var with mod Var V1, mod Var V2, Flip, mod Flip V1, mod Flip V3 (with and without QT)
plot_correlation_two_NNs(nn1_pred, nn9_pred, nn1_pred_std, nn9_pred_std, "BNN" + r'$^{\rm{DV}}_{1 \times 50}$', "BNN" + r'$^{\rm{DV-Mod A}}_{1 \times 50}$', output_dir, "BNN_Var_modV1_Var_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn1_pred, nn11_pred, nn1_pred_std, nn11_pred_std, "BNN" + r'$^{\rm{DV}}_{1 \times 50}$', "BNN" + r'$^{\rm{DV-Mod B}}_{1 \times 50}$', output_dir, "BNN_Var_modV2_Var_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn1_pred, nn3_pred, nn1_pred_std, nn3_pred_std, "BNN" + r'$^{\rm{DV}}_{1 \times 50}$', "BNN" + r'$^{\rm{DP}}_{1 \times 50}$', output_dir, "BNN_Var_Flip_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn1_pred, nn5_pred, nn1_pred_std, nn5_pred_std, "BNN" + r'$^{\rm{DV}}_{1 \times 50}$', "BNN" + r'$^{\rm{DP-Mod 1}}_{1 \times 50}$', output_dir, "BNN_Var_Flip_modV1_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn1_pred, nn7_pred, nn1_pred_std, nn7_pred_std, "BNN" + r'$^{\rm{DV}}_{1 \times 50}$', "BNN" + r'$^{\rm{DP-Mod 2}}_{1 \times 50}$', output_dir, "BNN_Var_Flip_modV2_comparison", privateWork=options.isPrivateWork())

plot_correlation_two_NNs(nn2_pred, nn10_pred, nn2_pred_std, nn10_pred_std, "BNN" + r'$^{\rm{DV,T}}_{1 \times 50}$', "BNN" + r'$^{\rm{DV-Mod A,T}}_{1 \times 50}$', output_dir, "QT_BNN_Var_modV1_Var_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn2_pred, nn12_pred, nn2_pred_std, nn12_pred_std, "BNN" + r'$^{\rm{DV,T}}_{1 \times 50}$', "BNN" + r'$^{\rm{DV-Mod B,T}}_{1 \times 50}$', output_dir, "QT_BNN_Var_modV2_Var_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn2_pred, nn4_pred, nn2_pred_std, nn4_pred_std, "BNN" + r'$^{\rm{DV,T}}_{1 \times 50}$', "BNN" + r'$^{\rm{DP,T}}_{1 \times 50}$', output_dir, "QT_BNN_Var_Flip_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn2_pred, nn6_pred, nn2_pred_std, nn6_pred_std, "BNN" + r'$^{\rm{DV,T}}_{1 \times 50}$', "BNN" + r'$^{\rm{DP-Mod 1,T}}_{1 \times 50}$', output_dir, "QT_BNN_Var_Flip_modV1_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn2_pred, nn8_pred, nn2_pred_std, nn8_pred_std, "BNN" + r'$^{\rm{DV,T}}_{1 \times 50}$', "BNN" + r'$^{\rm{DP-Mod 2,T}}_{1 \times 50}$', output_dir, "QT_BNN_Var_Flip_modV2_comparison", privateWork=options.isPrivateWork())

# Comparison Default Flip with mod Var V1, mod Var V2, mod Flip V1, mod Flip V3 (with and without QT)
plot_correlation_two_NNs(nn3_pred, nn9_pred, nn3_pred_std, nn9_pred_std, "BNN" + r'$^{\rm{DP}}_{1 \times 50}$', "BNN" + r'$^{\rm{DV-Mod A}}_{1 \times 50}$', output_dir, "BNN_Flip_modV1_Var_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn3_pred, nn11_pred, nn3_pred_std, nn11_pred_std, "BNN" + r'$^{\rm{DP}}_{1 \times 50}$', "BNN" + r'$^{\rm{DV-Mod B}}_{1 \times 50}$', output_dir, "BNN_Flip_modV2_Var_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn3_pred, nn5_pred, nn3_pred_std, nn5_pred_std, "BNN" + r'$^{\rm{DP}}_{1 \times 50}$', "BNN" + r'$^{\rm{DP-Mod 1}}_{1 \times 50}$', output_dir, "BNN_Flip_modV1_Flip_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn3_pred, nn7_pred, nn3_pred_std, nn7_pred_std, "BNN" + r'$^{\rm{DP}}_{1 \times 50}$', "BNN" + r'$^{\rm{DP-Mod 2}}_{1 \times 50}$', output_dir, "BNN_Flip_modV3_Flip_comparison", privateWork=options.isPrivateWork())

plot_correlation_two_NNs(nn4_pred, nn10_pred, nn4_pred_std, nn10_pred_std, "BNN" + r'$^{\rm{DP,T}}_{1 \times 50}$', "BNN" + r'$^{\rm{DV-Mod A,T}}_{1 \times 50}$', output_dir, "QT_BNN_Flip_modV1_Var_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn4_pred, nn12_pred, nn4_pred_std, nn12_pred_std, "BNN" + r'$^{\rm{DP,T}}_{1 \times 50}$', "BNN" + r'$^{\rm{DV-Mod B,T}}_{1 \times 50}$', output_dir, "QT_BNN_Flip_modV2_Var_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn4_pred, nn6_pred, nn4_pred_std, nn6_pred_std, "BNN" + r'$^{\rm{DP,T}}_{1 \times 50}$', "BNN" + r'$^{\rm{DP-Mod 1,T}}_{1 \times 50}$', output_dir, "QT_BNN_Flip_modV1_Flip_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn4_pred, nn8_pred, nn4_pred_std, nn8_pred_std, "BNN" + r'$^{\rm{DP,T}}_{1 \times 50}$', "BNN" + r'$^{\rm{DP-Mod 2,T}}_{1 \times 50}$', output_dir, "QT_BNN_Flip_modV3_Flip_comparison", privateWork=options.isPrivateWork())

# Comparison mod Var V1, with mod Var V2, mod Flip V1, mod Flip V3 (with and without QT)
plot_correlation_two_NNs(nn9_pred, nn11_pred, nn9_pred_std, nn11_pred_std, "BNN" + r'$^{\rm{DV-Mod A}}_{1 \times 50}$', "BNN" + r'$^{\rm{DV-Mod B}}_{1 \times 50}$', output_dir, "BNN_modV1_modV2_Var_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn9_pred, nn5_pred, nn9_pred_std, nn5_pred_std, "BNN" + r'$^{\rm{DV-Mod A}}_{1 \times 50}$', "BNN" + r'$^{\rm{DP-Mod 1}}_{1 \times 50}$', output_dir, "BNN_modV1_Var_modV1_Flip_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn9_pred, nn7_pred, nn9_pred_std, nn7_pred_std, "BNN" + r'$^{\rm{DV-Mod A}}_{1 \times 50}$', "BNN" + r'$^{\rm{DP-Mod 2}}_{1 \times 50}$', output_dir, "BNN_modV1_Var_modV3_Flip_comparison", privateWork=options.isPrivateWork())

plot_correlation_two_NNs(nn10_pred, nn12_pred, nn10_pred_std, nn12_pred_std, "BNN" + r'$^{\rm{DV-Mod A,T}}_{1 \times 50}$', "BNN" + r'$^{\rm{DV-Mod B,T}}_{1 \times 50}$', output_dir, "QT_BNN_modV1_modV2_Var_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn10_pred, nn6_pred, nn10_pred_std, nn6_pred_std, "BNN" + r'$^{\rm{DV-Mod A,T}}_{1 \times 50}$', "BNN" + r'$^{\rm{DP-Mod 1,T}}_{1 \times 50}$', output_dir, "QT_BNN_modV1_Var_modV1_Flip_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn10_pred, nn8_pred, nn10_pred_std, nn8_pred_std, "BNN" + r'$^{\rm{DV-Mod A,T}}_{1 \times 50}$', "BNN" + r'$^{\rm{DP-Mod 2,T}}_{1 \times 50}$', output_dir, "QT_BNN_modV1_Var_modV3_Flip_comparison", privateWork=options.isPrivateWork())

# Comparison mod Var V2, mod Flip V1, mod Flip V3 (with and without QT)
plot_correlation_two_NNs(nn11_pred, nn5_pred, nn11_pred_std, nn5_pred_std, "BNN" + r'$^{\rm{DV-Mod A}}_{1 \times 50}$', "BNN" + r'$^{\rm{DP-Mod 1}}_{1 \times 50}$', output_dir, "BNN_modV2_Var_modV1_Flip_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn11_pred, nn7_pred, nn11_pred_std, nn7_pred_std, "BNN" + r'$^{\rm{DV-Mod A}}_{1 \times 50}$', "BNN" + r'$^{\rm{DP-Mod 2}}_{1 \times 50}$', output_dir, "BNN_modV2_Var_modV3_Flip_comparison", privateWork=options.isPrivateWork())

plot_correlation_two_NNs(nn12_pred, nn6_pred, nn12_pred_std, nn6_pred_std, "BNN" + r'$^{\rm{DV-Mod A,T}}_{1 \times 50}$', "BNN" + r'$^{\rm{DP-Mod 1,T}}_{1 \times 50}$', output_dir, "QT_BNN_modV2_Var_modV1_Flip_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn12_pred, nn8_pred, nn12_pred_std, nn8_pred_std, "BNN" + r'$^{\rm{DV-Mod A,T}}_{1 \times 50}$', "BNN" + r'$^{\rm{DP-Mod 2,T}}_{1 \times 50}$', output_dir, "QT_BNN_modV2_Var_modV3_Flip_comparison", privateWork=options.isPrivateWork())

# Comparison mod Flip V1, mod Flip V3 (with and without QT)
plot_correlation_two_NNs(nn5_pred, nn7_pred, nn5_pred_std, nn7_pred_std, "BNN" + r'$^{\rm{DV-Mod A}}_{1 \times 50}$', "BNN" + r'$^{\rm{DP-Mod 2}}_{1 \times 50}$', output_dir, "BNN_modV1_Flip_modV3_Flip_comparison", privateWork=options.isPrivateWork())
plot_correlation_two_NNs(nn5_pred, nn8_pred, nn5_pred_std, nn8_pred_std, "BNN" + r'$^{\rm{DV-Mod A,T}}_{1 \times 50}$', "BNN" + r'$^{\rm{DP-Mod 2,T}}_{1 \times 50}$', output_dir, "QT_BNN_modV1_Flip_modV3_Flip_comparison", privateWork=options.isPrivateWork())