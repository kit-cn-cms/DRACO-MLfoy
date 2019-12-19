# global imports
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


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
input_samples.addSample(options.getDefaultName("ttb")  , label = "ttb"  , normalization_weight = options.getNomWeight())
input_samples.addSample(options.getDefaultName("ttcc") , label = "ttcc" , normalization_weight = options.getNomWeight())
input_samples.addSample(options.getDefaultName("ttlf") , label = "ttlf" , normalization_weight = options.getNomWeight())

if options.isBinary():
    input_samples.addBinaryLabel(options.getSignal(), options.getBinaryBkgTarget())

# initializing DNN training class
dnn = DNN.DNN(
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

out_dir_1 = "/home/nshadskiy/Documents/draco-bnns/workdir/ANNs_1"
pred_list = []
for i in range(50):
    i+=1
    preds = dnn.load_trained_model(out_dir_1 + "/ann_{}_ge4j_ge3t".format(i))
    pred_list.append(preds)

test_preds = np.concatenate(pred_list, axis=1)
ann_pred_vector = np.mean(test_preds, axis=1)
ann_pred_vector_std = np.std(test_preds, axis=1)

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

# # build DNN model
# bnn.build_model(options.getNetConfig())

# # # perform the training
# bnn.train_model()

# # # evalute the trained model
# bnn_pred, bnn_pred_std = bnn.eval_model()

# # # save information
# bnn.save_model(sys.argv, filedir, options.getNetConfigName())

# # plotting
# if options.doPlots():
#     # plot the evaluation metrics
#     bnn.plot_metrics(privateWork = options.isPrivateWork())

#     if options.isBinary():
#         # plot output node
#         bin_range = options.getBinaryBinRange()
#         bnn.plot_binaryOutput(
#             log         = options.doLogPlots(),
#             privateWork = options.isPrivateWork(),
#             printROC    = options.doPrintROC(),
#             nbins       = 15,
#             bin_range   = bin_range,
#             name        = options.getName(),
#             sigScale    = options.getSignalScale())

# out_dir_1 = "/home/nshadskiy/Documents/draco-bnns/workdir/bnn_1_ge4j_ge3t"
# bnn_pred1, bnn_pred_std1 = bnn.load_trained_model(out_dir_1)

out_dir_2 = "/home/nshadskiy/Documents/draco-bnns/workdir/bnn_4_ge4j_ge3t"
bnn_pred, bnn_pred_std = bnn.load_trained_model(out_dir_2)

from matplotlib.colors import LogNorm
plt.hist2d(ann_pred_vector, bnn_pred, bins=[50,50], cmin=1, norm=LogNorm())
#plt.hist2d(bnn_pred1, bnn_pred, bins=[50,50], cmin=1, norm=LogNorm())
plt.plot([0,1],[0,1],'k')
plt.colorbar()
plt.xlabel("$\mu$ ANNs", fontsize = 16)
#plt.xlabel("$\mu$ BNN 1", fontsize = 16)
plt.ylabel("$\mu$ BNN", fontsize = 16)
plt.savefig(out_dir_2+"/bnn_ann_mu_l2-3_dout30.png")
#plt.savefig(out_dir_2+"/bnn1_bnn2_mu.png")
print "bnn_ann_mu.png was created"
plt.savefig(out_dir_2+"/bnn_ann_mu_l2-3_dout30.pdf")
#plt.savefig(out_dir_2+"/bnn1_bnn2_mu.pdf")
print "bnn_ann_mu.pdf was created"
plt.close()

n_max = np.amax([np.amax(ann_pred_vector_std), np.amax(bnn_pred_std)])
#n_max = np.amax([np.amax(bnn_pred_std1), np.amax(bnn_pred_std)])
plt.hist2d(ann_pred_vector_std, bnn_pred_std, bins=[50,50], range=[[0.,n_max],[0.,n_max]], cmin=1, norm=LogNorm())
#plt.hist2d(bnn_pred_std1, bnn_pred_std, bins=[50,50], range=[[0.,n_max],[0.,n_max]], cmin=1, norm=LogNorm())
plt.plot([0,n_max],[0,n_max],'k')
plt.colorbar()
plt.xlabel("$\sigma$ ANNs", fontsize = 16)
#plt.xlabel("$\sigma$ BNN 1", fontsize = 16)
plt.ylabel("$\sigma$ BNN", fontsize = 16)
plt.savefig(out_dir_2+"/bnn_ann_std_l2-3_dout30.png")
#plt.savefig(out_dir_2+"/bnn1_bnn2_std.png")
print "bnn_ann_std.png was created"
plt.savefig(out_dir_2+"/bnn_ann_std_l2-3_dout30.pdf")
#plt.savefig(out_dir_2+"/bnn1_bnn2_std.pdf")
print "bnn_ann_std.pdf was created"
plt.close()
