import pandas as pd
import os
import sys
import optparse

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

usage = ""
parser = optparse.OptionParser(usage=usage)

parser.add_option("-o", "--output", dest="outName", default="mergedHDFfiles",
    help="Path and name of the new file where the merged data shall be saved", metavar="outName")

parser.add_option("-i", "--input", dest="inFiles", default=None,
    help="comma separated list with all .h5 files to add", metavar="inFiles")

parser.add_option("-id", "--inputDirectory", dest="inDirectory", default=".",
    help="path to the directory where the input files are", metavar="inDirectory")

parser.add_option("-od", "--outputDirectory", dest="outDirectory", default=".",
    help="path to the directory where the input files are", metavar="outDirectory")

parser.add_option("-n", "--naming", dest="naming", default="_dnn.h5",
    help="adds naming to input, so only ttlf... must be given", metavar="naming")

(options, args) = parser.parse_args()

if not os.path.isabs(options.inDirectory):
    sys.path.append(basedir+options.inDirectory+"/")
else:
    sys.path.append(options.inDirectory+"/")

inList=options.inFiles.split(",")

if os.path.exists(options.outName+".h5"):
    print("renaming file {}".format(options.outName))
    os.rename(options.outName,options.outName+".old")

with pd.HDFStore(options.outDirectory+"/"+options.outName+".h5", "a") as store:
    for h5file in inList:
        if options.inDirectory:
            tmp_df=pd.read_hdf(os.path.join(options.inDirectory+"/",h5file+options.naming))
        else:
            tmp_df=pd.read_hdf(h5file)
        tmp_df["class_label"] = pd.Series([options.outName]*tmp_df.shape[0], index = tmp_df.index) 
        store.append("data", tmp_df, index=False)
