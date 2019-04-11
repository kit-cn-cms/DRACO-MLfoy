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

parser.add_option("-t", "--threshold", dest = "threshold",
    help = "threshold of sample split", default = 0.5)

parser.add_option("-s", "--sample", dest = "sample",
    help = "sample to be split or folder of all samples to be split")

parser.add_option("-o", "--output", dest = "outDir",
    help = "DIR of split samples output")

(options, args) = parser.parse_args()
if not os.path.isabs(options.inputDir):
    inPath = basedir+"/workdir/"+options.inputDir
elif os.path.exists(options.inputDir):
    inPath = options.inputDir
else:
    sys.exit("ERROR: Input Directory does not exist!")
cpPath = inPath+"/checkpoints/"

if not os.path.isabs(options.outDir):
    outPath = basedir+"/workdir/"+options.outDir
else:
    outPath = options.outDir

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
def evaluate_sample(sample, model, variables, normCSV, outPath, threshold):
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
    df["prediction"] = pd.Series(prediction_vector[:,0], index = df.index)

    # split sample
    df_low = df.query("prediction < "+str(threshold))
    df_high = df.query("prediction >= "+str(threshold))
    print("number of events in low {}".format(df_low.shape[0]))
    print("number of events in high {}".format(df_high.shape[0]))

    # output paths
    if outPath.endswith("/"):
        outPath = outPath[:-1]
    high_path = outPath+"_high/"
    low_path = outPath+"_low/"
    # generate new directory for hdf5 files
    if not os.path.exists(high_path):
        os.makedirs(high_path)
    if not os.path.exists(low_path):
        os.makedirs(low_path)
    
    # save signal/bkg like events into directories
    sample_name = sample.split("/")[-1]
    df_low.to_hdf(low_path+"/"+sample_name, key = "data", mode = "w")
    df_high.to_hdf(high_path+"/"+sample_name, key = "data", mode = "w")
    

for sample in samples:
    evaluate_sample(sample, model, variables, normCSV, outPath, options.threshold)



