# global imports
import numpy as np
import rootpy.plotting as rp
import os
import sys
import socket
import matplotlib.pyplot as plt
from matplotlib import gridspec
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

import math
import DRACO_Frameworks.DNN_Aachen.DNN_Aachen as DNN_Aachen
import DRACO_Frameworks.DNN_Aachen.variable_info as variable_info
import DRACO_Frameworks.DNN_Aachen.data_frame as data_frame
import pyrootsOfTheCaribbean.plot_configs.plotting_styles as pltstyle

category_vars = {
    "4j_ge3t": variable_info.variables_4j_3b,
    "5j_ge3t": variable_info.variables_5j_3b,
    "ge6j_ge3t": variable_info.variables_6j_3b}            
categories = {
    "4j_ge3t":   "(N_Jets == 4 and N_BTagsM >= 3)",
    "5j_ge3t":   "(N_Jets == 5 and N_BTagsM >= 3)",
    "ge6j_ge3t": "(N_Jets >= 6 and N_BTagsM >= 3)",
    }
prenet_targets = [
    #"GenAdd_BB_inacceptance",
    #"GenAdd_B_inacceptance",
    "GenHiggs_BB_inacceptance",
    "GenHiggs_B_inacceptance",
    "GenTopHad_B_inacceptance",
    "GenTopHad_QQ_inacceptance",
    "GenTopHad_Q_inacceptance",
    "GenTopLep_B_inacceptance"
    ]

event_classes = ["ttHbb", "ttbb", "tt2b", "ttb", "ttcc", "ttlf"]

if "naf" in socket.gethostname():
    workpath = "/nfs/dust/cms/user/vdlinden/DRACO-MLfoy/workdir/"
else:
    workpath = "/ceph/vanderlinden/DRACO-MLfoy/workdir/"


inPath = workpath+"/AachenDNN_files"

try:
    key = sys.argv[1]
    smearing = eval(sys.argv[2])
    if smearing: print("doing smearing (x->x+rnd.normal(0,sigma))")
    else:        print("doing scaling  (x->x*rnd.normal(1,sigma))")
except:
    print("first  argument: JT cagegory")
    print("second argument: smearing/scaling (1/0)")
    exit()

outpath = workpath+"/AachenDNN_v2_"+str(key)+"/"
checkpoint_path = outpath + "/checkpoints/trained_main_net.h5py"

result_dir = "/ceph/vanderlinden/DRACO-MLfoy/WiggleStudies/results/KSscans"
if smearing:    result_dir += "_smearing/"
else:           result_dir += "_scaling/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
result_dir += "/"+str(key)+"/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

dnn_aachen = DNN_Aachen.DNN(
    in_path             = inPath,
    save_path           = outpath,
    event_classes       = event_classes,
    event_category      = categories[key],
    train_variables     = category_vars[key],
    prenet_targets      = prenet_targets,
    train_epochs        = 500,
    early_stopping      = 20,
    eval_metrics        = ["acc"],
    additional_cut      = None)


def gen_discrs( predictions ):
    discr_nodes = [[] for _ in event_classes]
    for evt in predictions:
        pred_class = np.argmax(evt)
        pred_value = evt[pred_class]
        discr_nodes[pred_class].append(pred_value)

    return discr_nodes



dnn_aachen.load_trained_model()
data = dnn_aachen.data.get_test_data(as_matrix=False)
prediction_before = dnn_aachen.main_net.predict(data.values)

discriminators_before = gen_discrs(prediction_before)
before_hists = []

pltstyle.init_plot_style()
bins = 40
bin_range = [0.,1.]
for i_node in range(len(event_classes)):
    node_values = discriminators_before[i_node]
    h = rp.Hist(bins, *bin_range, title = "before smearing")
    h.markersize = 0
    h.legendstyle = "F"
    h.fillstyle = "solid"
    h.fillcolor = "orangered"
    h.linecolor = "black"
    h.fill_array( node_values )
    before_hists.append(h)


n_samples = 10
sigma = 0.2
# smearing/scaling
def func(x):
    if smearing: return x + np.random.normal(0,sigma)
    else:        return x*np.random.normal(1,sigma)

plot_dir = result_dir + "/KS_cuts_{:.3f}".format(sigma).replace(".","_")
discrs_after = []
for _ in range(n_samples):
    print("generating new {} predictions ...".format("smeared" if smearing else "scaled"))
    data_new = data.applymap(func)

    prediction_after = dnn_aachen.main_net.predict(data_new.values)
    discr = gen_discrs(prediction_after)
    discrs_after.append(discr)

for i_node in range(len(event_classes)):
    new_hists = []
    for n in range(n_samples):
        node_values = discrs_after[n][i_node]
        new_h = rp.Hist(bins,*bin_range, title = "after smearing (s = {:.2f})".format(sigma))
        new_h.markersize = 0
        new_h.drawstyle = "shape"
        new_h.legendstyle = "L"
        new_h.fillstyle = "hollow"
        new_h.linestyle = "solid"
        new_h.linecolor = "teal"
        new_h.linewidth = 2
        new_h.fill_array( node_values )

        if n == 0:
            new_h_plot = new_h.Clone()
        new_hists.append(new_h)

    # loop over bins and fill average to plot hist with errors
    for iBin in range(new_h_plot.GetNbinsX()+1):
        bin_contents = [h.GetBinContent(iBin) for h in new_hists]
        new_h_plot.SetBinContent(iBin, np.mean(bin_contents))
        new_h_plot.SetBinError(iBin, np.std(bin_contents))

    # get unsmeared histogram
    old_h = before_hists[i_node]
    old_h_plot = old_h.Clone()

    # plot histogram
    canvas = pltstyle.init_canvas(ratiopad = True)
    stack = rp.HistStack( [old_h_plot], stacked = True, drawstyle = "HIST X0")
    max_val = max(stack.GetMaximum(), new_h.GetMaximum())
    stack.SetMaximum( max_val*1.3)
    
    rp.utils.draw([stack]+[new_h_plot], pad = canvas.cd(1),
        xtitle = "discriminator output for {} node".format( event_classes[i_node] ), ytitle = "Events")
    legend = pltstyle.init_legend([old_h_plot,new_h_plot])
    pltstyle.add_category_label(canvas.cd(1), categories[key])

    x_vals = []
    ks_probs_per_bin = []
    ks_error_per_bin = []
    for i_bin in range(new_h.GetNbinsX()):
        #if i_bin==0: continue # underflow

        # set bincontent to zero
        for iSample in range(n_samples):
            new_hists[iSample].SetBinContent(i_bin,0)
        old_h.SetBinContent(i_bin,0)

        x_val = old_h.GetBinCenter(i_bin)
        x_vals.append(x_val)
        
        ks_scores = []
        for iSample in range(n_samples):
            if new_hists[iSample].Integral() == 0 or old_h.Integral() == 0: continue
            
            ks = old_h.KolmogorovTest( new_hists[iSample])#, "N" )
            ks_scores.append(ks)

        if len(ks_scores) == 0:
            ks_probs_per_bin.append(1.)
            ks_error_per_bin.append(0.)
        else:
            ks_probs_per_bin.append( np.mean(ks_scores) )
            ks_error_per_bin.append( np.std(ks_scores) )

    # init ratio plot with line1
    line1 = rp.Graph(50)
    for i, x in enumerate(np.linspace(0.,1.,50)):
        line1.SetPoint(i,x,1)
    line1.markersize = 0
    line1.linecolor = "black"
    line1.SetMaximum(1.3)
    line1.SetMinimum(-0.3)
    line1.GetXaxis().SetTitle("{} node output".format( event_classes[i_node] ))
    line1.GetYaxis().SetTitle("KS prob")
    line1.GetXaxis().SetLimits(0.,1.)

    # adjust label size
    line1.GetYaxis().SetLabelSize(
        line1.GetYaxis().GetLabelSize()*2.4)
    line1.GetXaxis().SetLabelSize(
        line1.GetXaxis().GetLabelSize()*2.4)
    line1.GetYaxis().SetNdivisions(503)
    line1.GetXaxis().SetNdivisions(510)

    # adjust title size
    line1.GetYaxis().SetTitleSize(
        line1.GetYaxis().GetTitleSize()*2.)
    line1.GetXaxis().SetTitleSize(
        line1.GetXaxis().GetTitleSize()*2.4)
    line1.GetYaxis().CenterTitle(1)
    line1.GetYaxis().SetTitleOffset(0.4)

    canvas.cd(2)
    line1.Draw()

    # add KS score graph
    ks_plt = rp.Graph(len(x_vals), drawstyle = "E")
    for i, (x,y,yerr) in enumerate(zip(x_vals, ks_probs_per_bin, ks_error_per_bin)):
        ks_plt.SetPoint(i,x,y)
        ks_plt.SetPointError(i,0.,0.,yerr,yerr)

    ks_plt.SetLineColor(1)
    ks_plt.markersize = 1
    ks_plt.markercolor = "black"
    ks_plt.DrawCopy("same E0 P")

    save_name = plot_dir + "_{}_node.pdf".format( event_classes[i_node] )
    pltstyle.save_canvas(canvas,save_name)
