import os
import sys
import optparse
import pandas as pd
import glob

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

usage="usage=%prog [options]"
parser = optparse.OptionParser(usage=usage)

parser.add_option("-v", "--variableName", dest = "variableName",
    default = "CSV[0],CSV[1],CSV[2],CSV[3],CSV[4],CSV[5]",
    help = "name of csv variables, comma separated")

parser.add_option("-s", "--sample", dest = "sample",
    help = "h5 file or folder of all h5 files to add the variable to")

parser.add_option("--overwrite", dest = "overwrite", default = False, action = "store_true",
    help = "overwrite existing h5 files with added variables, otherwise create _new.h5")

(options, args) = parser.parse_args()
if not options.sample.endswith(".h5"):
    samples = glob.glob(options.sample+"/*.h5")
else:   
    samples = [options.sample]

def func(x):
    return (x>=0.0494)*1 + (x>=0.2770)*1 + (x>=0.7264)*1

# load sample
def evaluate_sample(sample, varName, overwrite = False):
    # load sample
    print("="*30)
    print("handling sample {}".format(sample))
    with pd.HDFStore(sample, mode = "r") as store:
        df = store.select("data")

    for variable in varName.split(","):
        print("\tadding {} to dataframe".format("discrete_"+variable))
        df["discrete_"+variable] = df[variable].apply(lambda x: func(x))

    # save signal/bkg like events into directories
    if overwrite:
        sample_name = sample
    else:
        sample_name = sample.replace(".h5", "_new.h5")
    df.to_hdf(sample_name, key = "data", mode = "w")
    print("wrote new sample at {}".format(sample_name))

for sample in samples:
    evaluate_sample(sample, options.variableName, overwrite = options.overwrite)



