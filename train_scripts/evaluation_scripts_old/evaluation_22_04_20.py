#cmd: python evaluation_V2.py -i /local/scratch/ssd/nshadskiy/2017_nominal -o comparison_22-04-2020_V2 -c ge4j_ge3t -v allVariables_2017_bnn --binary -S ttH -q --restorefitdir /home/ycung/Desktop/DRACO-MLfoy/workdir/16-04-2020/QT_BNN_training_ge4j_ge3t/fit_data.csv
# change restore_fit_dir (all models should use same fit info, so that they have the exact same datasets)
# TODO: change n_NNs=n_iterations and in THIS script, 
# TODO: REMOVE wrong entries in best_epoch and confusion_matrix.csv

# global imports
# so that matplotlib can be used over ssh
import matplotlib #me
matplotlib.use('Agg') #me

from copy import deepcopy #me
import csv #me
import operator #me

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import tqdm

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

# import class for DNN training
import DRACO_Frameworks.DNN.DNN as DNN
import DRACO_Frameworks.DNN.BNN as BNN
import DRACO_Frameworks.DNN.data_frame as df

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
            xtitle      = "predicted class",
            ytitle      = "true class",
            binlabel    = self.event_classes)

        canvas = setup.drawConfusionMatrixOnCanvas(cm, "confusion matrix", self.event_category, self.ROCScore, privateWork = privateWork)
        setup.saveCanvas(canvas, self.plotdir+"/mean_confusionMatrix"+suffix+".pdf")

def get_column(array, i):
    return [row[i] for row in array]

def ann_calc_mean_std(model, input_dir, n_NNs=1):
    pred_list = []
    for i in range(n_NNs):
        preds, event_class = model.load_trained_model(input_dir+"_"+str(i)+"_"+options.getCategory()) #TODO: compare with bnn_calc_mean
        pred_list.append(preds)
    test_preds = np.concatenate(pred_list, axis=1)
    return np.mean(test_preds, axis=1), np.std(test_preds, axis=1)

def multi_ann_calc_mean_std(model, input_dir, n_NNs=1):
    pred_list = {}
    test_preds_mean = []
    test_preds_std = []
    for j in range(n_NNs):
        preds, event_class = model.load_trained_model(input_dir+"_"+str(j)+"_"+options.getCategory()) #TODO: compare with bnn_calc_mean
        for sample_name in range(len(preds[0])):
            if event_class[sample_name] not in pred_list.keys():
                pred_list[event_class[sample_name]] = get_column(preds, sample_name)
                pred_list[event_class[sample_name]] = np.reshape(pred_list[event_class[sample_name]],(-1,1))
            else:
                pred_list[event_class[sample_name]]  = np.concatenate((pred_list[event_class[sample_name]], np.reshape(get_column(preds,sample_name),(-1,1))), axis=1)
    for i in range(len(pred_list.keys())):
        test_preds_mean.append(np.mean(pred_list[event_class[i]], axis = 1))
        test_preds_std.append(np.std(pred_list[event_class[i]], axis = 1))

    return test_preds_mean, test_preds_std, event_class

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

def count_decimal_place(number):
    number = str(number)
    if not '.' in number:
        return 0
    return int(len(number) - number.index('.') - 1)

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
    
def comparison_best_epoch(input_dir):
    with open(input_dir, "r") as f:
        csv_reader = csv.DictReader(f, delimiter=",")
        column_names = csv_reader.fieldnames
        best_epochs = {"Multi_QT": [], "ANN_QT": [], "Multi": [], "ANN": []}
        for row in csv_reader: 
            if not "TEST" in row[column_names[0]]: 
                if "QT" in row[column_names[0]]: 
                    if "Multi" in row[column_names[0]]: 
                        best_epochs["Multi_QT"].append(int(row[column_names[1]]))
                    else: 
                        best_epochs["ANN_QT"].append(int(row[column_names[1]]))

                elif not "QT" in row[column_names[0]]: 
                    if "Multi" in row[column_names[0]]: 
                        best_epochs["Multi"].append(int(row[column_names[1]]))
                    else: 
                        best_epochs["ANN"].append(int(row[column_names[1]]))
        
        for key in best_epochs.keys():
            print key+" mean_best_epochs: " + str(np.mean(best_epochs[key]))


######################################################### initializing BNN/DNN training class ##################################################################
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

########################################################################################################################
output_dir = options.getOutputDir()
work_dir = "/home/ycung/Desktop/DRACO-MLfoy/workdir/"
event_classes = ["ttH", "ttbb", "ttcc", "ttlf"]

n_iterations = 30

input_dir_1 = work_dir+"16-04-2020/BNN_training_ge4j_ge3t"
nn1_pred, nn1_pred_std, labels1 = bnn.load_trained_model(input_dir_1, n_iterations)

input_dir_2 = work_dir+"16-04-2020/QT_BNN_training_ge4j_ge3t"
nn2_pred, nn2_pred_std, labels2 = bnn_qt.load_trained_model(input_dir_2, n_iterations)

input_dir_3 = work_dir+"training_ANN/ANN_training"
nn3_pred, nn3_pred_std = ann_calc_mean_std(model=dnn, input_dir=input_dir_3, n_NNs=n_iterations)

input_dir_4 = work_dir+"training_ANN/QT_ANN_training"
nn4_pred, nn4_pred_std = ann_calc_mean_std(model=dnn_qt, input_dir=input_dir_4, n_NNs=n_iterations)


# compare BNN and ANN without quantile transformation
plot_correlation_two_NNs(nn1_pred, nn3_pred, nn1_pred_std, nn3_pred_std, "BNN", "ANN", output_dir, "BNN_ANN_comparison")

# compare BNN and ANN with quantile transformation
plot_correlation_two_NNs(nn2_pred, nn4_pred, nn2_pred_std, nn4_pred_std, "BNN_QT", "ANN_QT", output_dir, "BNN_ANN_QT_comparison")

values, values_with_qt = plot_confusion_matrix_correlation(work_dir + "confusion_matrix_modified.csv", work_dir, event_classes)
confusion_class = plotConfusionMatrix(event_classes=event_classes, event_category=options.getCategory(), plotdir=work_dir)
confusion_class.plot(values)
confusion_class.plot(values_with_qt, suffix="_qt")
confusion_class.plot(values_with_qt-values, suffix="_diff")

comparison_best_epoch(work_dir+"best_epoch_modified.csv")

#ROC Values mean
print "without QT"
print (0.787*2 + 0.788*7+0.789*8 + 0.790*3)/20.
print "with QT"
print (0.787*3+0.788*7+0.790+0.789*8+0.786)/20.

###########################################################################################################################################################################################################################

###### code for evaluation of systematic uncertainties ######

# make plots of the output distribution for one single event

#plot_event_distribution(out_dir_1)

#plot_3_diff_hist(out_dir_1)
#plot_mixing_diff_hist(out_dir_1)

#plot_2_uncert_diff(out_dir_1)
#plot_1_uncert_diff(out_dir_1)


###### evaliation code for BNN uncertainty dependency ######

# calculating the statistical uncertainty dependence
# prior = ["0p001", "0p0025", "0p01", "0p025", "0p1", "0p25", "1", "2p5","10", "25", "100", "250", "1000"]
# prior_ = [0.001, 0.0025, 0.01, 0.025, 0.1, 0.25, 1., 2.5, 10., 25., 100., 250., 1000.]

# #ME
# prior = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]
# prior_ = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# stds = []
# stds_std = []

# for i in prior:
#     # initializing BNN training class
#     bnn = BNN.BNN(
#         save_path       = options.getOutputDir(),
#         input_samples   = input_samples,
#         category_name   = options.getCategory(),
#         train_variables = options.getTrainVariables(),
#         # number of epochs
#         train_epochs    = options.getTrainEpochs(),
#         # metrics for evaluation (c.f. KERAS metrics)
#         eval_metrics    = ["acc"],
#         # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
#         test_percentage = options.getTestPercentage(),
#         # balance samples per epoch such that there amount of samples per category is roughly equal
#         balanceSamples  = options.doBalanceSamples(),
#         shuffle_seed    = 42,
#         evenSel         = options.doEvenSelection(),
#         norm_variables  = options.doNormVariables())

#     # bnn_pred, bnn_pred_std, labels = bnn.load_trained_model("/home/nshadskiy/Documents/draco-bnns/workdir/bnn_prior_{}_nomCSV_ge4j_ge3t".format(i))
#     bnn_pred, bnn_pred_std, labels = bnn.load_trained_model("/home/nshadskiy/Documents/DRACO-MLfoy/workdir/bnn{}_prior_1_nomCSV_ge4j_ge3t".format(i))
#     stds.append(stats.mode(bnn_pred_std)[0])
#     stds_std.append(stats.tstd(bnn_pred_std))


# plt.errorbar(prior_, stds, yerr=stds_std, fmt='o')
# #plt.xlabel("prior width", fontsize = 16)
# plt.xlabel("trained on sample size in %", fontsize = 16) # ~ 830 000 events
# #plt.xscale("log")
# plt.ylabel("mean of event $\sigma$", fontsize = 16)
# plt.savefig("/home/nshadskiy/Documents/draco-bnns/workdir/bnn_stats_std.pdf")
# print "bnns_std.pdf was created"