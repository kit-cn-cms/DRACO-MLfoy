# global imports
import os
import sys

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

# import classes for training
import DRACO_Frameworks.ttbarMatcher.ttbarMatcher as ttbarMatcher
import DRACO_Frameworks.ttbarMatcher.data_frame as df

# absolute path to folder with input dataframes
inPath   = "/ceph/vanderlinden/MLFoyTrainData/ttbarMatcher/"

# naming for input files
naming = "_input.h5"

# define input objects
input_features = df.InputFeatures()
input_features.addObject("ttMatchInputJet",     length = "N_ttMatchInputJets",    max = 10)
input_features.addObject("ttMatchInputLepton",  length = "N_ttMatchInputLeptons", max = 1)
input_features.addObject("ttMatchInputMET")
# define input variables
input_features.addVariables(["E", "Px", "Py", "Pz"])# "M", "CSV", "Constituents"])
input_features.generateVariableSet()

# define output objects
target_features = df.TargetFeatures()
target_features.addObjects("ttMatchTargetHadTop")
target_features.addObjects("ttMatchTargetLepTop")
# define target variables
target_features.addVariables(["E", "Px", "Py", "Pz"])
target_features.generateTargets()

# load samples
input_samples = df.InputSamples(inPath, max_events = 100000)

# define the input samples one by one
input_samples.addSample("ttbar"+naming, label = "ttbar")

# path to output directory (adjust NAMING)
savepath = basedir+"/workdir/"+"ttbarMatcher"+"/"

# initializing train class
matcher = ttbarMatcher.ttbarMatcher(
    save_path       = savepath,
    input_samples   = input_samples,
    input_features  = input_features,
    target_features = target_features,
    shuffle_inputs  = True,
    loss_function   = "mschrode_kit_cool_loss",
    feature_scaling = 500.,
    n_epochs        = 10,
    val_percentage  = 0.2,
    test_percentage = 0.01)

matcher.build_model()

matcher.train_model()

matcher.eval_model()
