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

parser.add_option("-o", "--output", dest="outFile", default="mergedHDFfiles",
    help="Path and name of the new file where the merged data shall be saved", metavar="outFile")

parser.add_option("-i", "--input", dest="inFiles", default=None,
    help="comma separated list with all .h5 files to add", metavar="inFiles")

parser.add_option("-d", "--directory", dest="directory", default=".",
    help="adds directory from path to systems working directory", metavar="directory")

parser.add_option("-n", "--naming", dest="naming", default="_dnn.h5",
    help="adds naming to input, so only ttlf... must be given", metavar="naming")

(options, args) = parser.parse_args()

if not os.path.isabs(options.directory):
    sys.path.append(basedir+options.directory+"/")
else:
    sys.path.append(directory+"/")

inList=options.inFiles.split(",")

if os.path.exists(options.outFile):
    print("renaming file {}".format(options.outFile))
    os.rename(options.outFile,options.outFile+".old")

with pd.HDFStore(options.outFile, "a") as store:
    for h5file in inList:
        if options.directory:  
            store.append("data", pd.read_hdf(os.path.join(options.directory+"/",h5file+options.naming)), index=False)
        else:
            store.append("data", pd.read_hdf(h5file), index=False)
