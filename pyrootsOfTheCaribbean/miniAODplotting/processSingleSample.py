import os
import sys
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(filedir)

sample_file = sys.argv[1]
output_dir = sys.argv[2]
sample_type = sys.argv[3]

import sampleProcessor as processor

processor.processSample(sample_file, output_dir, sample_type)

