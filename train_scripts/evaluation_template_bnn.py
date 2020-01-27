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
#input_samples.addSample(options.getDefaultName("ttb")  , label = "ttb"  , normalization_weight = options.getNomWeight())
input_samples.addSample(options.getDefaultName("ttcc") , label = "ttcc" , normalization_weight = options.getNomWeight())
input_samples.addSample(options.getDefaultName("ttlf") , label = "ttlf" , normalization_weight = options.getNomWeight())

if options.isBinary():
    input_samples.addBinaryLabel(options.getSignal(), options.getBinaryBkgTarget())

def anns_calc_mean_std(n_NNs, input_dir):
    pred_list = []
    for i in range(n_NNs):
        i+=1
        print "Iteration: {}".format(i)
        preds = dnn.load_trained_model(input_dir + "/ann_{}_ge4j_ge3t".format(i))
        pred_list.append(preds)
    test_preds = np.concatenate(pred_list, axis=1)
    return np.mean(test_preds, axis=1), np.std(test_preds, axis=1)


def plot_correlation_two_NNs(NN_pred1, NN_pred2, NN_pred1_std, NN_pred2_std, x_lab, y_lab, save_dir, save_name):
    from matplotlib.colors import LogNorm
    plt.hist2d(NN_pred1, NN_pred2, bins=[50,50], cmin=1, norm=LogNorm())
    plt.plot([0,1],[0,1],'k')
    plt.colorbar()
    plt.xlabel("$\mu$ "+x_lab, fontsize = 16)
    plt.ylabel("$\mu$ "+y_lab, fontsize = 16)
    plt.savefig(save_dir+"/plots/mu_{}.png".format(save_name))
    print "mu_{}.png was created".format(save_name)
    plt.savefig(save_dir+"/plots/mu_{}.pdf".format(save_name))
    print "mu_{}.pdf was created".format(save_name)
    plt.close()

    n_max = np.amax([np.amax(NN_pred1_std), np.amax(NN_pred2_std)])
    plt.hist2d(NN_pred1_std, NN_pred2_std, bins=[50,50], range=[[0.,n_max],[0.,n_max]], cmin=1, norm=LogNorm())
    plt.plot([0,n_max],[0,n_max],'k')
    plt.colorbar()
    plt.xlabel("$\sigma$ "+x_lab, fontsize = 16)
    plt.ylabel("$\sigma$ "+y_lab, fontsize = 16)
    plt.savefig(save_dir+"/plots/std_{}.png".format(save_name))
    print "std_{}.png was created".format(save_name)
    plt.savefig(save_dir+"/plots/std_{}.pdf".format(save_name))
    print "std_{}.pdf was created".format(save_name)
    plt.close()


# # initializing DNN training class
# dnn = DNN.DNN(
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

# l2, dout = "7", "30"
# out_dir_1 = "/home/nshadskiy/Documents/draco-bnns/workdir/ANNs_l2-{}_dout{}".format(l2, dout)
# nn1_pred, nn1_pred_std = anns_calc_mean_std(n_NNs=10, input_dir=out_dir_1)


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

bnn_prior_1 = "3"
out_dir_1 = "/home/nshadskiy/Documents/draco-bnns/workdir/bnn_t{}_p005_ge4j_ge3t".format(bnn_prior_1)
nn1_pred, nn1_pred_std, labels1 = bnn.load_trained_model(out_dir_1)


# bnn_prior_2 = "t"
# out_dir_2 = "/home/nshadskiy/Documents/draco-bnns/workdir/bnn_prior_{}_ge4j_ge3t".format(bnn_prior_2)
# nn2_pred, nn2_pred_std, labels2 = bnn.load_trained_model(out_dir_2)


# nn2_sig_pred, nn2_sig_pred_std, nn1_sig_pred, nn1_sig_pred_std, nn2_bkg_pred, nn2_bkg_pred_std, nn1_bkg_pred, nn1_bkg_pred_std = [],[],[],[],[],[],[],[]
# for i in range(len(labels2)):
#     if labels2[i]==1:
#         nn1_sig_pred.append(nn1_pred[i])
#         nn1_sig_pred_std.append(nn1_pred_std[i])
#         nn2_sig_pred.append(nn2_pred[i])
#         nn2_sig_pred_std.append(nn2_pred_std[i])
#     elif labels2[i]==0:
#         nn1_bkg_pred.append(nn1_pred[i])
#         nn1_bkg_pred_std.append(nn1_pred_std[i])
#         nn2_bkg_pred.append(nn2_pred[i])
#         nn2_bkg_pred_std.append(nn2_pred_std[i])
#     else:
#         print "--wrong event--"


# plot_correlation_two_NNs(nn1_pred, nn2_pred, nn1_pred_std, nn2_pred_std, "BNN 1", "BNN 2", out_dir_2, "bnn1_bnn2_prior_{}_{}".format(bnn_prior_1, bnn_prior_2))
# plot_correlation_two_NNs(nn1_sig_pred, nn2_sig_pred, nn1_sig_pred_std, nn2_sig_pred_std, "BNN 1 signal", "BNN 2 signal", out_dir_2, "sig_bnn1_bnn2_prior_{}_{}".format(bnn_prior_1, bnn_prior_2))
# plot_correlation_two_NNs(nn1_bkg_pred, nn2_bkg_pred, nn1_bkg_pred_std, nn2_bkg_pred_std, "BNN 1 background", "BNN 2 background", out_dir_2, "bkg_bnn1_bnn2_prior_{}_{}".format(bnn_prior_1, bnn_prior_2))

# plot_correlation_two_NNs(nn1_pred, nn2_pred, nn1_pred_std, nn2_pred_std, "ANNs", "BNN", out_dir_2, "anns_bnn_l2_{}_prior_{}".format(l2, bnn_prior_2))
# plot_correlation_two_NNs(nn1_sig_pred, nn2_sig_pred, nn1_sig_pred_std, nn2_sig_pred_std, "ANNs signal", "BNN signal", out_dir_2, "sig_anns_bnn_l2_{}_prior_{}".format(l2, bnn_prior_2))
# plot_correlation_two_NNs(nn1_bkg_pred, nn2_bkg_pred, nn1_bkg_pred_std, nn2_bkg_pred_std, "ANNs background", "BNN background", out_dir_2, "bkg_anns_bnn_l2_{}_prior_{}".format(l2, bnn_prior_2))
