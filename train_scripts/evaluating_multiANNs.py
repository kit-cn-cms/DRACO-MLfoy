# -*- coding: utf-8 -*-

# python evaluating_multiANNs.py -i "/local/scratch/ssd/ycung/new_h5_files_2017_v2" -o comparison_multiANNs -c ge4j_ge3t -v new_variableset_25_v2 -S ttH -q -P --restorefitdir /home/ycung/Desktop/DRACO-MLfoy_thesis/workdir/training_ANN_150N/QT_ANN_150N_training_0_ge4j_ge3t/fit_data.pck
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
import DRACO_Frameworks.DNN.DNN as DNN
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
class plotConfusionMatrix:
    def __init__(self, event_classes, event_category, plotdir):
        self.event_classes     = event_classes
        self.event_category    = event_category
        self.plotdir           = plotdir
        self.ROCScore          = None

    def plot(self,array ,suffix="", norm_matrix = True, privateWork = False):  
        # initialize Histogram
        cm = setup.setupConfusionMatrix(
            matrix      = array,
            ncls        = len(self.event_classes),
            xtitle      = "Ausgegebene Klasse", 
            ytitle      = "Tats#ddot{a}chliche Klasse",
            binlabel    = self.event_classes)

        canvas = setup.drawConfusionMatrixOnCanvas(cm, "confusion matrix", self.event_category, self.ROCScore, privateWork = privateWork)
        setup.saveCanvas(canvas, self.plotdir+"/mean_confusionMatrix"+suffix+".pdf")

def get_column(array, i):
    return [row[i] for row in array]

def plot_correlation_two_NNs(NN_pred1, NN_pred2, NN_pred1_std, NN_pred2_std, x_lab, y_lab, save_dir, save_name, privateWork = False, current_class = ""):
    from matplotlib.colors import LogNorm
    plt.rc('xtick',labelsize=14)
    plt.rc('ytick',labelsize=14)

    plt.hist2d(NN_pred1, NN_pred2, bins=[50,50], cmin=1, norm=LogNorm())
    plt.plot([0,1],[0,1],'k')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.xlabel(x_lab+" Ausgabewert $\mu$", fontsize = 16)
    plt.ylabel(y_lab+" Ausgabewert $\mu$", fontsize = 16)

    if privateWork:
        plt.title(r"$\bf{CMS\ private\ work}$", loc = "left", fontsize = 14)
        plt.title(current_class, loc = "right", fontsize = 16)

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
    plt.ylabel(y_lab+" Standardabweichung $\sigma$", fontsize = 16)
    if privateWork:
        plt.title(r"$\bf{CMS\ private\ work}$", loc = "left", fontsize = 14)
        plt.title(current_class, loc = "right", fontsize = 16)

    plt.tight_layout()
    plt.savefig(save_dir+"/std_{}.png".format(save_name))
    print "std_{}.png was created".format(save_name)
    plt.savefig(save_dir+"/std_{}.pdf".format(save_name))
    print "std_{}.pdf was created".format(save_name)
    plt.close()

def count_decimal_place(number):
    number = str(number)
    if not '.' in number:
        return 0
    return int(len(number) - number.index('.') - 1)

def multi_ann_calc_mean_std(model, input_dir, n_NNs=1):
    pred_list = {}
    test_preds_mean = []
    test_preds_std = []
    total_eval_duration = 0
    total_pred_duration = 0
    model_eval_list = []

    for j in range(n_NNs):
        print "ITERATIONS MultiANN: " + input_dir.split("workdir/")[-1]+ ": " + str(j+1) + "/" + str(n_NNs)
        preds, event_class, test_labels, eval_duration, pred_duration, model_eval = model.load_trained_model(input_dir+"_"+str(j)+"_"+options.getCategory()) 
        for sample_name in range(len(preds[0])):
            if event_class[sample_name] not in pred_list.keys():
                pred_list[event_class[sample_name]] = get_column(preds, sample_name)
                pred_list[event_class[sample_name]] = np.reshape(pred_list[event_class[sample_name]],(-1,1))
            else:
                pred_list[event_class[sample_name]]  = np.concatenate((pred_list[event_class[sample_name]], np.reshape(get_column(preds,sample_name),(-1,1))), axis=1)
        
        model_eval_list.append(model_eval)
        total_eval_duration += eval_duration
        total_pred_duration += pred_duration

    for i in range(len(pred_list.keys())):
        test_preds_mean.append(np.mean(pred_list[event_class[i]], axis = 1))
        test_preds_std.append(np.std(pred_list[event_class[i]], axis = 1))


    from sklearn.metrics import roc_auc_score
    test_preds_mean_reshaped = np.array([list(a) for a in zip(test_preds_mean[0],test_preds_mean[1],test_preds_mean[2], test_preds_mean[3])])
    mean_roc_auc_score = roc_auc_score(test_labels, test_preds_mean_reshaped)

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

    dict_eval_metrics = {}
    dict_eval_metrics["model_test_loss"] = np.mean(model_eval_list, axis=0)[0]
    for im, metric in enumerate(model.eval_metrics):
        dict_eval_metrics["model_test_"+str(metric)] = np.mean(model_eval_list, axis = 0)[im+1]
    
    import collections
    dict_eval_metrics = collections.OrderedDict(sorted(dict_eval_metrics.items()))

    ''' save eval metrics to csv file'''
    filename = basedir+"/workdir/eval_metrics.csv"
    file_exists = os.path.isfile(filename)
    with open(filename, "a+") as f:
        headers = np.concatenate((["project_name"], dict_eval_metrics.keys()))
        csv_writer = csv.DictWriter(f,delimiter=',', lineterminator='\n',fieldnames=headers)
        if not file_exists:
            csv_writer.writeheader()
        
        row = {"project_name": options.getOutputDir().split("/")[-1]+"__"+input_dir.split("workdir/")[-1]}
        row.update(dict_eval_metrics)
        csv_writer.writerow(row)
        print("saved eval metrics to "+str(filename))

    ''' save eval duration loaded model to csv file'''
    filename = basedir+"/workdir/eval_duration_loaded_model.csv"
    file_exists = os.path.isfile(filename)
    with open(filename, "a+") as f:
        headers = ["project_name", "eval_duration (hh:mm:ss)", "total_pred_duration (hh:mm:ss)", "mean_pred_duration (hh:mm:ss/npreds)"]
        csv_writer = csv.DictWriter(f,delimiter=',', lineterminator='\n',fieldnames=headers)
        if not file_exists:
            csv_writer.writeheader()
        csv_writer.writerow({"project_name": options.getOutputDir().split("/")[-1]+"__"+input_dir.split("workdir/")[-1], "eval_duration (hh:mm:ss)": datetime.timedelta(seconds = total_eval_duration),
                                "total_pred_duration (hh:mm:ss)": datetime.timedelta(seconds = total_pred_duration), "mean_pred_duration (hh:mm:ss/npreds)": datetime.timedelta(seconds = total_pred_duration/float(n_NNs))})
        print("saved eval duration loaded model to "+str(filename))

    return test_preds_mean, test_preds_std, event_class

def plot_confusion_matrix_correlation(input_dir, output_dir, event_classes):
    data_with_qt, data = {}, {}
    with open(input_dir, "r") as f:
        csv_reader = csv.DictReader(f, delimiter=",")
        column_names = csv_reader.fieldnames
        for row in csv_reader:    
            for i in column_names[1:]:
                key = row[column_names[0]].split("_")[-1]+"-"+i.split("_")[-1]
                if "QT" in row[column_names[0]]:
                    if key not in data_with_qt.keys():
                       data_with_qt[key] = []
                    data_with_qt[key].append(float(row[i]))
                                
                elif "QT" not in row[column_names[0]]:
                    if key not in data.keys():
                       data[key] = []
                    data[key].append(float(row[i]))

        mean, mean_with_qt = {}, {}
        for key in data_with_qt.keys():
            mean[key] = np.mean(data[key])
            mean_with_qt[key] = np.mean(data_with_qt[key])

            interested_decimal_place = 1./1000.  
            min_value = min(np.amin(data_with_qt[key]), np.amin(data[key]))
            max_value = max(np.amax(data_with_qt[key]), np.amax(data[key]))
            d = max_value - min_value
            nbins= round(d/interested_decimal_place+1.)
            width_bins = d/nbins
            bins = np.linspace(min_value - width_bins/2., max_value + width_bins/2., nbins+1)

            plt.hist(data[key], bins=bins, label='without QT', color="deepskyblue", alpha=0.3)
            plt.hist(data_with_qt[key], bins=bins, label="with QT", color="orangered", alpha=0.4)
            
            plt.xlabel("confusion matrix value",  fontsize = 16)
            plt.xticks(np.arange(round(min_value,count_decimal_place(interested_decimal_place)), round(max_value,count_decimal_place(interested_decimal_place)), step=interested_decimal_place*2), rotation=45)
            plt.ylabel("events",  fontsize = 16)
            plt.axvline(mean[key], color="deepskyblue", linestyle='dashed', linewidth=1, label="mean: " + str(round(mean[key], count_decimal_place(interested_decimal_place))))
            plt.axvline(mean_with_qt[key], color="orangered", linestyle='dashed', linewidth=1, label="mean: " + str(round(mean_with_qt[key], count_decimal_place(interested_decimal_place))))
            plt.title(key, loc="center",  fontsize = 16)
            plt.legend(loc="upper right")
            plt.savefig(output_dir+"/"+str(key)+".png")
            # print key+" mean: " + str(mean[key])
            # print key+"_with_QT: mean: " + str(mean_with_qt[key])
            # print "DELTA: " + str(round(mean[key]-mean_with_qt[key], count_decimal_place(interested_decimal_place)))
            # print "__________________________________________________________________"
            print key+".pdf was created"
            plt.close()
        
        #reshape array "mean" and "mean_with_qt"
        mean_2 = np.empty((len(event_classes), len(event_classes)))
        mean_2_with_qt = np.empty((len(event_classes), len(event_classes)))
        
        for i in event_classes:
            for j in mean.keys():
                if j.split("-")[0] == i:
                    for k in event_classes:
                        mean_2[event_classes.index(i)][event_classes.index(k)] = mean[i+"-"+k]
                        mean_2_with_qt[event_classes.index(i)][event_classes.index(k)] = mean_with_qt[i+"-"+k]

        ranking = sorted(mean, key=mean.get, reverse=True)
        ranking_with_qt = sorted(mean_with_qt, key=mean_with_qt.get, reverse=True)

        print "Ranking: without_qt" + "---------" + "with_qt"
        for i in range(len(ranking)):
            print str(i) +": "+ranking[i] + "---------" + ranking_with_qt[i]
    
    return mean_2, mean_2_with_qt

    
output_dir = options.getOutputDir()
work_dir = basedir+"/workdir/"

######################################################### initializing BNN/DNN training class ##################################################################
dnn = DNN.DNN(
    save_path       = options.getOutputDir(),
    input_samples   = input_samples, #changed
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

dnn_qt = DNN.DNN(
    save_path       = options.getOutputDir(),
    input_samples   = input_samples, #changed
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
########################################################################################################################

event_classes = ["ttH", "tthf", "ttcc", "ttlf"]
math_dict = {
    "tthf": r"$\mathregular{t\overline{t}+hf}$",
    "ttcc": r"$\mathregular{t\overline{t}+cc}$",
    "ttlf": r"$\mathregular{t\overline{t}+lf}$",
    "ttH": r"$\mathregular{t\overline{t}+H}$",
}
n_iterations = 100 

input_dir_1 = work_dir+"training_modified_multiANNs/modified_MultiANN_training"
input_dir_2 = work_dir+"training_modified_multiANNs/modified_MultiANN_QT_training"

nn1_pred, nn1_pred_std, event_class1 = multi_ann_calc_mean_std(model=dnn, input_dir=input_dir_1, n_NNs=n_iterations)
nn2_pred, nn2_pred_std, event_class2 = multi_ann_calc_mean_std(model=dnn_qt, input_dir=input_dir_2, n_NNs=n_iterations)

# Comparison multiAnns (with and without QT)
for i in event_class1:
    plot_correlation_two_NNs(nn1_pred[event_class1.index(i)], nn2_pred[event_class1.index(i)], nn1_pred_std[event_class1.index(i)], nn2_pred_std[event_class1.index(i)], "(Multi-ANN)"+ r'$^{\rm{D}}_{1 \times 150}$', "(Multi-ANN)" +r'$^{\rm{D,T}}_{1 \times 150}$', output_dir, "MultiANN_comparison_"+i, privateWork=options.isPrivateWork(), current_class = math_dict[i])


# #compare confusion matrix calues with and without qt
# category_label = JTcut.getJTlabel (options.getCategory())
# values, values_with_qt = plot_confusion_matrix_correlation(work_dir+"confusion_matrix.csv", output_dir, event_classes)
# confusion_class = plotConfusionMatrix(event_classes=event_classes, event_category=category_label, plotdir=output_dir)
# confusion_class.plot(values, privateWork=options.isPrivateWork())
# confusion_class.plot(values_with_qt, suffix="_qt", privateWork=options.isPrivateWork())
# confusion_class.plot(values_with_qt-values, suffix="_diff", privateWork=options.isPrivateWork())