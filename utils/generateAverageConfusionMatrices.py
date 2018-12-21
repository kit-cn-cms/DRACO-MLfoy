# global imports
import os
import sys
import numpy as np
import pandas as pd
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

import generateJTcut as JTcut
import pyrootsOfTheCaribbean.plot_configs.setupPlots as setup


JTcategory  = sys.argv[1]

event_classes = ["ttHbb", "ttbb", "tt2b", "ttb", "ttcc", "ttlf"]

cmdir = basedir+"/workdir/confusionMatrixData/"
inPath1 = cmdir+"/topVariablesLoose_"+str(JTcategory)+".h5"
inPath2 = cmdir+"/topVariablesTight_"+str(JTcategory)+".h5"
names = ["loose variable selection", "tight variable selection"]

# generate confusion matrix variable list
labels = ["{}_in_{}_node".format(pred, true) for true in event_classes for pred in event_classes]

def get_average_conf_matrix(path, labels, nclasses):
    df = pd.read_hdf(path, "data")[labels+["ROC"]]
    print("n matrices: "+str(df.shape[0]))
    average_matrix = []
    stddevs = []
    for label in labels:
        average_matrix.append( df[label].mean() )
        stddevs.append( df[label].std() )

    average_matrix = np.array(average_matrix).reshape(nclasses,nclasses)
    stddevs = np.array(stddevs).reshape(nclasses,nclasses)

    roc_mean = df["ROC"].mean()
    roc_std = df["ROC"].std()
    
    return average_matrix, stddevs, roc_mean, roc_std

def norm_matrix(cm, cm_std):
    # norm matrix and its errors
    new_matrix = np.empty( (len(event_classes), len(event_classes)), dtype = np.float64)
    new_errors = np.empty( (len(event_classes), len(event_classes)), dtype = np.float64)
    for yit in range(len(event_classes)):
        evt_sum = float(sum(cm[yit,:]))
        for xit in range(len(event_classes)):
            new_matrix[yit,xit] = cm[yit,xit]/(evt_sum+1e-9)
            new_errors[yit,xit] = cm_std[yit,xit]/(evt_sum+1e-9)
    return new_matrix, new_errors

def plot_matrix(cm, cm_std, roc, roc_std, savedir, name, norm = True):
    if norm:
        cm, cm_std = norm_matrix(cm, cm_std)

    hist = setup.setup2DHistogram(
        matrix   = cm.T, 
        ncls     = len(event_classes), 
        xtitle   = "predicted class", 
        ytitle   = "true class", 
        binlabel = event_classes,
        errors   = cm_std)

    JTLabel = JTcut.getJTlabel(JTcategory)
    canvas = setup.draw2DHistOnCanvas(hist, "ConfusionMatrix", JTLabel, ROC = roc, ROCerr = roc_std)
    setup.printTitle(canvas, name)
    setup.saveCanvas(canvas, savedir)

# generate average matrix for first 
cm1, cm_std1, roc1, roc_std1 = get_average_conf_matrix(inPath1, labels, nclasses = len(event_classes))
savename = cmdir+"/confusionMatrix_{}_{}.pdf".format("loose", JTcategory)
plot_matrix(cm1, cm_std1, roc1, roc_std1, savename, name = names[0])

# generate average matrix for second
cm2, cm_std2, roc2, roc_std2 = get_average_conf_matrix(inPath2, labels, nclasses = len(event_classes))
savename = cmdir+"/confusionMatrix_{}_{}.pdf".format("tight", JTcategory)
plot_matrix(cm2, cm_std2, roc2, roc_std2, savename, name = names[1])


def get_diff_matrix(cm2, cm1, std2, std1):
    cm2, std2 = norm_matrix(cm2, std2)
    cm1, std1 = norm_matrix(cm1, std1)
    diff = np.empty( (cm2.shape[0], cm2.shape[1]), dtype = np.float64)
    std  = np.empty( (cm2.shape[0], cm2.shape[1]), dtype = np.float64)

    for yit in range(cm2.shape[0]):
        for xit in range(cm2.shape[1]):
            diff[xit,yit] = cm2[xit,yit] - cm1[xit,yit]
            std[xit,yit] = np.sqrt( std2[xit,yit]**2 + std1[xit,yit]**2 )

    return diff, std
        


# generate difference matrix
diff_cm, diff_cm_std = get_diff_matrix(cm2, cm1, cm_std2, cm_std1)
diff_roc = roc2 - roc1
diff_roc_std = np.sqrt(roc_std2**2 + roc_std1**2)
name = "difference (tight-loose)"
savename = cmdir+"/confusionMatrix_{}_{}.pdf".format("difference", JTcategory)
plot_matrix(diff_cm, diff_cm_std, diff_roc, diff_roc_std, savename, name = name, norm = False)
