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

l2, dout = 7, 30

out_dir_1 = "/home/nshadskiy/Documents/draco-bnns/workdir/ANNs_l2-{}_dout{}".format(l2, dout)
pred_list = []
for i in range(50):
    i+=1
    print "Iteration: {}".format(i)
    preds = dnn.load_trained_model(out_dir_1 + "/ann_{}_ge4j_ge3t".format(i))
    pred_list.append(preds)

test_preds = np.concatenate(pred_list, axis=1)
ann_pred = np.mean(test_preds, axis=1)
ann_pred_std = np.std(test_preds, axis=1)

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

sigma = "0p001"

out_dir_2 = "/home/nshadskiy/Documents/draco-bnns/workdir/bnn_prior_{}_ge4j_ge3t".format(sigma)
#out_dir_2 = "/home/nshadskiy/Documents/draco-bnns/workdir/bnn_trained_wo_bias_ge4j_ge3t"
bnn_pred, bnn_pred_std, labels = bnn.load_trained_model(out_dir_2)

bnn_sig_pred, bnn_sig_pred_std, ann_sig_pred, ann_sig_pred_std, bnn_bkg_pred, bnn_bkg_pred_std, ann_bkg_pred, ann_bkg_pred_std = [],[],[],[],[],[],[],[]
for i in range(len(labels)):
    if labels[i]==1:
        bnn_sig_pred.append(bnn_pred[i])
        bnn_sig_pred_std.append(bnn_pred_std[i])
        ann_sig_pred.append(ann_pred[i])
        ann_sig_pred_std.append(ann_pred_std[i])
    elif labels[i]==0:
        bnn_bkg_pred.append(bnn_pred[i])
        bnn_bkg_pred_std.append(bnn_pred_std[i])
        ann_bkg_pred.append(ann_pred[i])
        ann_bkg_pred_std.append(ann_pred_std[i])
    else:
        print "--wrong event--"

print len(bnn_sig_pred)+len(bnn_bkg_pred)
print len(ann_sig_pred)+len(ann_bkg_pred)

from matplotlib.colors import LogNorm
plt.hist2d(ann_pred, bnn_pred, bins=[50,50], cmin=1, norm=LogNorm())
plt.plot([0,1],[0,1],'k')
plt.colorbar()
plt.xlabel("$\mu$ ANNs", fontsize = 16)
plt.ylabel("$\mu$ BNN", fontsize = 16)
plt.savefig(out_dir_2+"/plots/bnn_ann_mu_l2-{}_dout{}.png".format(l2, dout))
print "bnn_ann_mu.png was created"
plt.savefig(out_dir_2+"/plots/bnn_ann_mu_l2-{}_dout{}.pdf".format(l2, dout))
print "bnn_ann_mu.pdf was created"
plt.close()

n_max = np.amax([np.amax(ann_pred_std), np.amax(bnn_pred_std)])
plt.hist2d(ann_pred_std, bnn_pred_std, bins=[50,50], range=[[0.,n_max],[0.,n_max]], cmin=1, norm=LogNorm())
plt.plot([0,n_max],[0,n_max],'k')
plt.colorbar()
plt.xlabel("$\sigma$ ANNs", fontsize = 16)
plt.ylabel("$\sigma$ BNN", fontsize = 16)
plt.savefig(out_dir_2+"/plots/bnn_ann_std_l2-{}_dout{}.png".format(l2, dout))
print "bnn_ann_std.png was created"
plt.savefig(out_dir_2+"/plots/bnn_ann_std_l2-{}_dout{}.pdf".format(l2, dout))
print "bnn_ann_std.pdf was created"
plt.close()

# only signal events plotted

plt.hist2d(ann_sig_pred, bnn_sig_pred, bins=[50,50], cmin=1, norm=LogNorm())
plt.plot([0,1],[0,1],'k')
plt.colorbar()
plt.xlabel("$\mu$ ANNs signal", fontsize = 16)
plt.ylabel("$\mu$ BNN signal", fontsize = 16)
plt.savefig(out_dir_2+"/plots/bnn_ann_mu_sig_l2-{}_dout{}.png".format(l2, dout))
print "bnn_ann_mu_sig.png was created"
plt.savefig(out_dir_2+"/plots/bnn_ann_mu_sig_l2-{}_dout{}.pdf".format(l2, dout))
print "bnn_ann_mu_sig.pdf was created"
plt.close()

n_max = np.amax([np.amax(ann_sig_pred_std), np.amax(bnn_sig_pred_std)])
plt.hist2d(ann_sig_pred_std, bnn_sig_pred_std, bins=[50,50], range=[[0.,n_max],[0.,n_max]], cmin=1, norm=LogNorm())
plt.plot([0,n_max],[0,n_max],'k')
plt.colorbar()
plt.xlabel("$\sigma$ ANNs signal", fontsize = 16)
plt.ylabel("$\sigma$ BNN signal", fontsize = 16)
plt.savefig(out_dir_2+"/plots/bnn_ann_std_sig_l2-{}_dout{}.png".format(l2, dout))
print "bnn_ann_std_sig.png was created"
plt.savefig(out_dir_2+"/plots/bnn_ann_std_sig_l2-{}_dout{}.pdf".format(l2, dout))
print "bnn_ann_std_sig.pdf was created"
plt.close()

# only background events plotted

plt.hist2d(ann_bkg_pred, bnn_bkg_pred, bins=[50,50], cmin=1, norm=LogNorm())
plt.plot([0,1],[0,1],'k')
plt.colorbar()
plt.xlabel("$\mu$ ANNs background", fontsize = 16)
plt.ylabel("$\mu$ BNN background", fontsize = 16)
plt.savefig(out_dir_2+"/plots/bnn_ann_mu_bkg_l2-{}_dout{}.png".format(l2, dout))
print "bnn_ann_mu_bkg.png was created"
plt.savefig(out_dir_2+"/plots/bnn_ann_mu_bkg_l2-{}_dout{}.pdf".format(l2, dout))
print "bnn_ann_mu_bkg.pdf was created"
plt.close()

n_max = np.amax([np.amax(ann_bkg_pred_std), np.amax(bnn_bkg_pred_std)])
plt.hist2d(ann_bkg_pred_std, bnn_bkg_pred_std, bins=[50,50], range=[[0.,n_max],[0.,n_max]], cmin=1, norm=LogNorm())
plt.plot([0,n_max],[0,n_max],'k')
plt.colorbar()
plt.xlabel("$\sigma$ ANNs background", fontsize = 16)
plt.ylabel("$\sigma$ BNN background", fontsize = 16)
plt.savefig(out_dir_2+"/plots/bnn_ann_std_bkg_l2-{}_dout{}.png".format(l2, dout))
print "bnn_ann_std_bkg.png was created"
plt.savefig(out_dir_2+"/plots/bnn_ann_std_bkg_l2-{}_dout{}.pdf".format(l2, dout))
print "bnn_ann_std_bkg.pdf was created"
plt.close()
