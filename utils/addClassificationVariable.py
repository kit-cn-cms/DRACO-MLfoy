import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import sys
import optparse
import json
import pandas as pd
import keras
import glob

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

usage="usage=%prog [options]"
parser = optparse.OptionParser(usage=usage)

parser.add_option("-i", "--input", dest = "inputDir",
    help = "DIR of trained net data")

parser.add_option("-v", "--variableName", dest = "variableName",
    help = "name of variable to be added")

parser.add_option("-s", "--sample", dest = "sample",
    help = "sample to be split or folder of all samples to be split")

parser.add_option("--overwrite", dest = "overwrite", default = False, action = "store_true",
    help = "overwrite existing h5 files with added variables, otherwise create _new.h5")

(options, args) = parser.parse_args()
if not os.path.isabs(options.inputDir):
    inPath = basedir+"/workdir/"+options.inputDir
elif os.path.exists(options.inputDir):
    inPath = options.inputDir
else:
    sys.exit("ERROR: Input Directory does not exist!")
cpPath = inPath+"/checkpoints/"

if not options.sample.endswith(".h5"):
    samples = glob.glob(options.sample+"/*.h5")
else:   
    samples = [options.sample]

# load checkpoints
modelFile = cpPath+"/trained_model.h5py"
configFile = cpPath+"/net_config.json"
normFile = cpPath+"/variable_norm.csv"

with open(configFile) as f:
    config = f.read()
config = json.loads(config)

# load csv norm file
normCSV = pd.read_csv(normFile, index_col = 0)
# load input variables
variables = config["trainVariables"]

# build DNN from checkpoints
model = keras.models.load_model(modelFile)
model.summary()

# load sample
def evaluate_sample(sample, model, variables, normCSV, varName, overwrite = False):
    # load sample
    print("="*30)
    print("handling sample {}".format(sample))
    with pd.HDFStore(sample, mode = "r") as store:
        df = store.select("data")

    variable_df = df[variables]
    # apply norm csv
    for v in list(normCSV.index.values):
        mean = float(normCSV["mu"].loc[v])
        std  = float(normCSV["std"].loc[v])
        variable_df[v] = (variable_df[v]-mean)/(std)

    # calculate binary output value
    prediction_vector = model.predict(
        variable_df[variables].values)
    df[varName] = pd.Series(prediction_vector[:,0], index = df.index)

    # save signal/bkg like events into directories
    if overwrite:
        sample_name = sample
    else:
        sample_name = sample.replace(".h5", "_new.h5")
    df.to_hdf(sample_name, key = "data", mode = "w")
    print("wrote new sample at {}".format(sample_name))

for sample in samples:
    evaluate_sample(sample, model, variables, normCSV, options.variableName, overwrite = options.overwrite)



