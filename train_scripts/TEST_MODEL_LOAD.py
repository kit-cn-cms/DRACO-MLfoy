# python TEST_MODEL_LOAD.py -i /local/scratch/ssd/nshadskiy/2017_nominal -o comparison_24-06-20 -c ge4j_ge3t -v allVariables_2017_bnn --binary -n BNN -S ttH -q --restorefitdir /home/ycung/Desktop/DRACO-MLfoy/workdir/16-04-2020/QT_BNN_training_ge4j_ge3t/fit_data.csv
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
import DRACO_Frameworks.DNN.BNN_DenseFlipout as BNN_DP #TODO #DEBUG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
import DRACO_Frameworks.DNN.DNN as DNN
import DRACO_Frameworks.DNN.data_frame as df

import plot_configs.setupPlots as setup
import utils.generateJTcut as JTcut

import csv
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


def ann_calc_mean_std(model, input_dir, n_NNs=1):
    pred_list = []
    for i in range(n_NNs):
        print "ITERATIONS" +str(model)+ ": " + str(i+1) + "/" + str(n_NNs)
        preds, event_class, test_labels = model.load_trained_model(input_dir+"_"+str(i)+"_"+options.getCategory()) #TODO: compare with bnn_calc_mean
        pred_list.append(preds)
    test_preds = np.concatenate(pred_list, axis=1)

    from sklearn.metrics import roc_auc_score
    mean_roc_auc_score = roc_auc_score(test_labels, np.mean(test_preds, axis=1))

    ''' save roc_auc_score to csv file'''
    filename = basedir+"/workdir/roc_auc_score.csv"
    file_exists = os.path.isfile(filename)
    with open(filename, "a+") as f:
        headers = ["project_name", "roc_auc_score"]
        csv_writer = csv.DictWriter(f,delimiter=',', lineterminator='\n',fieldnames=headers)
        if not file_exists:
            csv_writer.writeheader()
        csv_writer.writerow({"project_name": options.getOutputDir().split("/")[-1]+"__"+input_dir.split("workdir/")[-1], "roc_auc_score": mean_roc_auc_score})
        print("saved roc_auc_score to "+str(filename))

    print("\nROC-AUC score: {}".format(np.mean(mean_roc_auc_score)))

    return np.mean(test_preds, axis=1), np.std(test_preds, axis=1)
    
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

dnn_qt = DNN.DNN(
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
    qt_transformed_variables = options.doQTNormVariables(), #changed
    restore_fit_dir = options.getRestoreFitDir()) #changed

# initializing DNN training class 
dnn = DNN.DNN(
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
    restore_fit_dir = None) #changed
########################################################################################################################

event_classes = ["ttH", "ttbb", "ttcc", "ttlf"]
n_iterations = 1

input_dir_1 = work_dir+"Flipout_QT_BNN_training_0_ge4j_ge3t"
input_dir_2 = work_dir+"Flipout_BNN_training_0_ge4j_ge3t"
input_dir_3 = work_dir+"Variational_QT_BNN_training_0_ge4j_ge3t"
input_dir_4 = work_dir+"Variational_BNN_training_0_ge4j_ge3t"
input_dir_5 = work_dir+"training_ANN_v2/QT_ANN_v2_training"
input_dir_6 = work_dir+"training_ANN_v2/ANN_v2_training"

nn1_pred, nn1_pred_std, labels1 = bnnDP_qt.load_trained_model(options.getNetConfig(),input_dir_1, n_iterations)
nn2_pred, nn2_pred_std, labels2 = bnnDP.load_trained_model(options.getNetConfig(), input_dir_2, n_iterations)
nn3_pred, nn3_pred_std, labels3 = bnn_qt.load_trained_model(input_dir_3, n_iterations)
nn4_pred, nn4_pred_std, labels4 = bnn.load_trained_model(input_dir_4, n_iterations)
nn5_pred, nn5_pred_std = ann_calc_mean_std(model=dnn_qt, input_dir=input_dir_5, n_NNs=n_iterations)
nn6_pred, nn6_pred_std = ann_calc_mean_std(model=dnn, input_dir=input_dir_6, n_NNs=n_iterations)


#Comparison ANN
plot_correlation_two_NNs(nn6_pred, nn5_pred, nn6_pred_std, nn5_pred_std, "binary_ANN", "binary_ANN_QT", output_dir, "ANN_[100]_comparison")

#Comparison Flipout
plot_correlation_two_NNs(nn2_pred, nn1_pred, nn2_pred_std, nn1_pred_std, "binary_BNN_DenseFlipout", "binary_BNN_DenseFlipout_QT", output_dir, "DenseFlipout_comparison")

#Comparison Variational
plot_correlation_two_NNs(nn4_pred, nn3_pred, nn4_pred_std, nn3_pred_std, "binary_BNN_DenseVariational", "binary_BNN_DenseVariational_QT", output_dir, "Variational_comparison")

#Comparison Flipout and ANN (with and without QT)
plot_correlation_two_NNs(nn2_pred, nn6_pred, nn2_pred_std, nn6_pred_std, "binary_BNN_DenseFlipout", "binary_ANN", output_dir, "Flipout_ANN_comparison")
plot_correlation_two_NNs(nn1_pred, nn5_pred, nn1_pred_std, nn5_pred_std, "binary_BNN_DenseFlipout_QT", "binary_ANN_QT", output_dir, "Flipout_ANN_QT_comparison")

#Comparison Flipout und Variational (with and without QT)
plot_correlation_two_NNs(nn2_pred, nn4_pred, nn2_pred_std, nn4_pred_std, "binary_BNN_DenseFlipout", "binary_BNN_DenseVariational", output_dir, "Flipout_Variational_comparison")
plot_correlation_two_NNs(nn1_pred, nn3_pred, nn1_pred_std, nn3_pred_std, "binary_BNN_DenseFlipout_QT", "binary_BNN_DenseVariational_QT", output_dir, "Flipout_Variational_QT_comparison")

# Does not work as file probably too large
# print "Only saving backup left..."

# ''' save predictions and stds'''
# filename = output_dir+"/backup_values.csv"
# backup_values = np.asarray([nn1_pred, nn1_pred_std, nn2_pred, nn2_pred_std, nn3_pred, nn3_pred_std, nn4_pred, nn4_pred_std, nn5_pred, nn5_pred_std, nn6_pred, nn6_pred_std])
# np.savetxt(filename, backup_values, delimiter=",")
# print("saved backup_values to "+str(filename))

