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
    "4j_4t": variable_info.variables_4j_3b,
    "5j_ge3t": variable_info.variables_5j_3b,
    "ge6j_ge3t": variable_info.variables_6j_3b}            
categories = {
    "4j_ge3t":   "(N_Jets == 4 and N_BTagsM >= 3)",
    "4j_4t":   "(N_Jets == 4 and N_BTagsM == 4)",
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

result_dir = "/ceph/vanderlinden/DRACO-MLfoy/WiggleStudies/results/ACCscans"
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

n_samples = 10
sigma = 0.2
# smearing/scaling
def func(x):
    if smearing: return x + np.random.normal(0,sigma)
    else:        return x*np.random.normal(1,sigma)

# generate a number of smeared histograms
unchangeds = []
changeds = []
plot_dir = result_dir + "/ACC_scan_{:.3f}".format(sigma).replace(".","_")
for n in range(n_samples):
    print("generating new {} predictions ...".format("smeared" if smearing else "scaled"))
    data_new = data.applymap(func)
    
    pred_after = dnn_aachen.main_net.predict(data_new.values)

    # take unsmeared predictions 
    # for each entry check prediction is same for smeared prediction
    unchanged_predictions = [[] for _ in range(len(event_classes))]
    changed_predictions = [[] for _ in range(len(event_classes))]
    for i in range(len(prediction_before)):
        # predicted class of current event
        pred_class = np.argmax(prediction_before[i])
        # discriminator value
        max_val = prediction_before[i][ pred_class ]
        if np.argmax(pred_after[i]) == pred_class:
            # same was predicted for wiggle
            unchanged_predictions[pred_class].append(max_val)
        else:
            # other was predicted for wiggle
            changed_predictions[pred_class].append(max_val)
    unchangeds.append(unchanged_predictions)
    changeds.append(changed_predictions)

    
for i_node in range(len(event_classes)):
    # loop over samples
    unchanged_hists = []
    changed_hists = []
    for i_sample in range(n_samples):
        # generate unchanged histogram
        values = unchangeds[i_sample][i_node]
        unc_h = rp.Hist(bins, *bin_range, title = "unchanged prediction")
        unc_h.markersize = 0
        unc_h.legendstyle = "F"
        unc_h.fillstyle = "solid"
        unc_h.fillcolor = "green"
        unc_h.linecolor = "black"
        unc_h.fill_array(values)
        unchanged_hists.append( unc_h )

        # generate changed histogram
        values = changeds[i_sample][i_node]
        ch_h = rp.Hist(bins, *bin_range, title = "changed prediction")
        ch_h.markersize = 0
        ch_h.legendstyle = "F"
        ch_h.fillstyle = "solid"
        ch_h.fillcolor = "darkred"
        ch_h.linecolor = "black"
        ch_h.fill_array(values)
        changed_hists.append(ch_h)

        # save one hist for plotting
        if i_sample == 0:
            unc_plot = unc_h.Clone()
            ch_plot = ch_h.Clone()

    # set bincontents of plot histogram to means of all samples
    for iBin in range(unc_plot.GetNbinsX()+1):
        bin_contents = [h.GetBinContent(iBin) for h in unchanged_hists]
        unc_plot.SetBinContent(iBin, np.mean(bin_contents))
        unc_plot.SetBinError(iBin, np.std(bin_contents))
    for iBin in range(ch_plot.GetNbinsX()+1):
        bin_contents = [h.GetBinContent(iBin) for h in changed_hists]
        ch_plot.SetBinContent(iBin, np.mean(bin_contents))
        ch_plot.SetBinError(iBin, np.std(bin_contents))

    # generate upper plot
    canvas = pltstyle.init_canvas(ratiopad = True)
    stack = rp.HistStack([unc_plot, ch_plot], stacked = True, drawstyle = "HIST E1 X0")
    max_val = stack.GetMaximum()
    stack.SetMaximum(max_val*1.3)

    rp.utils.draw([stack], pad = canvas.cd(1),
        xtitle = "discriminator output for {} node".format( event_classes[i_node] ), ytitle = "Events")
    legend = pltstyle.init_legend([unc_plot, ch_plot])
    pltstyle.add_category_label(canvas.cd(1), categories[key])

    # generate lower plot
    line1 = rp.Graph(50)
    for i, x in enumerate(np.linspace(0.,1.,50)):
        line1.SetPoint(i,x,1)
    line1.markersize = 0
    line1.linecolor = "black"
    line1.SetMaximum(1.3)
    line1.SetMinimum(-0.3)
    line1.GetXaxis().SetTitle("{} node output".format( event_classes[i_node] ))
    line1.GetYaxis().SetTitle("accuracy in bin")
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

    # add values
    r = rp.Graph(unc_plot.GetNbinsX(), drawstyle = "E")
    for iBin in range(unc_plot.GetNbinsX()+1):
        x = unc_plot.GetBinCenter(iBin)
        unc_val = unc_plot.GetBinContent(iBin)
        unc_err = unc_plot.GetBinError(iBin)
        ch_val = ch_plot.GetBinContent(iBin)
        ch_err = ch_plot.GetBinError(iBin)
        if unc_val == 0 or ch_val == 0:
            ratio = 0
            ratio_err = 0
        else:

            ratio = 1.*unc_val/(unc_val+ch_val)
            ratio_err = np.sqrt(
                (1./(unc_val+ch_val)-unc_val/(unc_val+ch_val)**2)**2*unc_err**2 + \
                (unc_val/(unc_val+ch_val)**2)**2*ch_err)

        r.SetPoint(iBin,x,ratio)
        r.SetPointError(iBin,0.,0.,ratio_err,ratio_err)

    r.SetLineColor(1)
    r.makersize = 1
    r.markercolor = "black"
    r.DrawCopy("same E0 P")

    save_name = plot_dir + "_{}_node.pdf".format(event_classes[i_node])
    pltstyle.save_canvas(canvas, save_name)
        
