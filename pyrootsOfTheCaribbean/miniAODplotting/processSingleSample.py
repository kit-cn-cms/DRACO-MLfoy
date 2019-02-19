# this file is called by each NAF batch job for each mAOD file
import optparse
import os
import sys
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(filedir)

parser = optparse.OptionParser("usage=%prog [options] processors")
parser.add_option("-f",dest="path_to_sample",
    help = "absolute path to mAOD-type sample")
parser.add_option("-o",dest="output_path",
    help = "absolute path to output hdf5 file")
parser.add_option("-s",dest="sample_type",default="ttH",
    help = "type of sample; needed for matching specific particles")
parser.add_option("-w",dest="cross_section_weight",default=1.,
    help = "cross section weight of the sample to norm the yields to 1/fb")
(opts, args) = parser.parse_args()

print("processing sample: {}".format(opts.path_to_sample))
print("output:            {}".format(opts.output_path))
print("sample type:       {}".format(opts.sample_type))
print("XS weight:         {}".format(opts.cross_section_weight))
print("-"*50)
print("using the following variable processors:")
for proc in args:
    print(proc)

import sampleProcessor as processor
# call the sample processor function which handles the processing
processor.processSample(
    sample      = opts.path_to_sample,
    out_path    = opts.output_path,
    sample_type = opts.sample_type,
    XSWeight    = float(opts.cross_section_weight),
    processors  = args)


