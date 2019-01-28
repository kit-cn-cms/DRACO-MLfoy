# this file is called by each NAF batch job for each mAOD file
import os
import sys
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(filedir)

# path to mAOD type root file
sample_file     = sys.argv[1]
# directory for output hdf5 file
output_dir      = sys.argv[2]
# type of sample (ttH/ttZ/...)
sample_type     = sys.argv[3]
# cross section weight for sample
XSWeight        = float(sys.argv[4])

import sampleProcessor as processor
# call the sample processor function which handles the processing
processor.processSample(sample_file, output_dir, sample_type, XSWeight)

