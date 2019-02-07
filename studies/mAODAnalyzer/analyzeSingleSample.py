import sys
import os
import sampleProcessor

sample_path = sys.argv[1]
output_file = sys.argv[2]

sampleProcessor.analyze(sample_path, output_file)
