# global imports
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats


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
input_samples = df.InputSamples(options.getInputDirectory(), options.getActivatedSamples(), options.getTestPercentage())

# define all samples
input_samples.addSample(options.getDefaultName("ttH"), label = "ttH", normalization_weight = options.getNomWeight())
input_samples.addSample(options.getDefaultName("ttbb") , label = "ttbb" , normalization_weight = options.getNomWeight())
input_samples.addSample(options.getDefaultName("tt2b") , label = "tt2b" , normalization_weight = options.getNomWeight())
#input_samples.addSample(options.getDefaultName("ttb")  , label = "ttb"  , normalization_weight = options.getNomWeight())
input_samples.addSample(options.getDefaultName("ttcc") , label = "ttcc" , normalization_weight = options.getNomWeight())
input_samples.addSample(options.getDefaultName("ttlf") , label = "ttlf" , normalization_weight = options.getNomWeight())

if options.isBinary():
    input_samples.addBinaryLabel(options.getSignal(), options.getBinaryBkgTarget())

# # initializing BNN training class
# bnn = BNN.BNN(
#     save_path       = options.getOutputDir(),
#     input_samples   = input_samples,
#     category_name   = options.getCategory(),
#     train_variables = options.getTrainVariables(),
#     # number of epochs
#     train_epochs    = options.getTrainEpochs(),
#     # metrics for evaluation (c.f. KERAS metrics)
#     eval_metrics    = ["acc"],
#     # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
#     test_percentage = options.getTestPercentage(),
#     # balance samples per epoch such that there amount of samples per category is roughly equal
#     balanceSamples  = options.doBalanceSamples(),
#     shuffle_seed    = 42,
#     evenSel         = options.doEvenSelection(),
#     norm_variables  = options.doNormVariables())

# # initializing BNN training class
# bnn2 = BNN.BNN(
#     save_path       = options.getOutputDir(),
#     input_samples   = input_samples,
#     category_name   = options.getCategory(),
#     train_variables = options.getTrainVariables(),
#     # number of epochs
#     train_epochs    = options.getTrainEpochs(),
#     # metrics for evaluation (c.f. KERAS metrics)
#     eval_metrics    = ["acc"],
#     # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
#     test_percentage = options.getTestPercentage(),
#     # balance samples per epoch such that there amount of samples per category is roughly equal
#     balanceSamples  = options.doBalanceSamples(),
#     shuffle_seed    = 42,
#     evenSel         = options.doEvenSelection(),
#     norm_variables  = options.doNormVariables())

# out_dir = "/home/nshadskiy/Documents/draco-bnns/workdir/bnn_2nodes_ge4j_ge3t"
# bnn_pred, bnn_pred_std, labels = bnn.load_trained_model(out_dir)

# # out_dir2 = "/home/nshadskiy/Documents/draco-bnns/workdir/bnn_untrained1_ge4j_ge3t"
# # bnn_pred2, bnn_pred_std2, labels2 = bnn2.load_trained_model(out_dir2)


# model = bnn.get_model()
# # model2 = bnn2.get_model()

# layers = model.layers
# # layers2 = model2.layers
# #print "Layers: " , layers
# layers.pop(0)
# # layers2.pop(0)
# for layer in layers:
#     print layer.name
#     print "---------------------------"
#     print "posterior: ", layer.get_weights()[0]
#     print "prior:     ", layer.get_weights()[1]
#     print "---------------------------"

# for layer in layers2:
#     print layer.name
#     print "---------------------------"
#     print "posterior: ", layer.get_weights()[0]
#     print "prior:     ", layer.get_weights()[1]
#     print "---------------------------"

# posterior1 = layers[-1].get_weights()[0]
# prior1 = layers[-1].get_weights()[1]
# post_weights1 = posterior1[0:20]
# post_bias1 = posterior1[20]
# post_weights1_std = np.log(np.exp(np.log(np.expm1(1.))+posterior1[21:41])+1)
# post_bias1_std = np.log(np.exp(np.log(np.expm1(1.))+posterior1[41])+1)
# pri_weights1 = posterior1[0:20]
# pri_bias1 = posterior1[20]
# pri_weights1_std = np.log(np.exp(np.log(np.expm1(1.))+posterior1[21:41])+1)
# pri_bias1_std = np.log(np.exp(np.log(np.expm1(1.))+posterior1[41])+1)



# posterior2 = layers2[-1].get_weights()[0]
# prior2 = layers2[-1].get_weights()[1]
# post_weights2 = posterior2[0:20]
# post_bias2 = posterior2[20]
# post_weights2_std = np.log(np.exp(np.log(np.expm1(1.))+posterior2[21:41])+1)
# post_bias2_std = np.log(np.exp(np.log(np.expm1(1.))+posterior2[41])+1)
# pri_weights2 = posterior2[0:20]
# pri_bias2 = posterior2[20]

# print "trained prosterior:       ", post_weights1, post_bias1
# print "trained prosterior std:   ", post_weights1_std, post_bias1_std
# print "untrained prosterior:     ", post_weights2, post_bias2
# print "untrained prosterior std: ", post_weights2_std, post_bias2_std

# print "trained prior:     ", pri_weights1, pri_bias1
# print "untrained prior:   ", pri_weights2, pri_bias2
# print "trained prior std: ", pri_weights1_std, pri_bias1_std




# calculating the statistical uncertainty dependence
prior = ["0p001", "0p0025", "0p01"]#, "0p025", "0p1", "0p25", "1", "2p5","10", "25", "100", "250", "1000"]
prior_ = [0.001, 0.0025, 0.01]#, 0.025, 0.1, 0.25, 1., 2.5, 10., 25., 100., 250., 1000.]
stds = []
stds_std = []

for i in prior:
    # initializing BNN training class
    bnn = BNN.BNN(
        save_path       = options.getOutputDir(),
        input_samples   = input_samples,
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
        norm_variables  = options.doNormVariables())

    bnn_pred, bnn_pred_std, labels = bnn.load_trained_model("/home/nshadskiy/Documents/draco-bnns/workdir/bnn_prior_{}_nomCSV_ge4j_ge3t".format(i))
    #std_hist, bin_edges = np.histogram(bnn_pred_std, bins=100, range=(0.,0.25), density=False)
    #std_mode = stats.mode(bnn_preds_std)
    stds.append(stats.mode(bnn_preds_std))
    stds_std.append(stats.stdev(bnn_pred_std))


plt.errorbar(prior_, stds, yerr=stds_std, fmt='o')
plt.xlabel("prior width", fontsize = 16)
plt.xscale("log")
plt.ylabel("mean of event $\sigma$", fontsize = 16)
plt.savefig("/home/nshadskiy/Documents/draco-bnns/workdir/bnn_prior_std_t.pdf")
print "bnns_std.pdf was created"
