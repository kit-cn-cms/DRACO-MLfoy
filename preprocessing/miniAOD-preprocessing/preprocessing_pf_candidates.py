# script to be executed by batch jobs
import miniAOD_preprocessing_pf_candidates as preproc
import numpy as np
import sys


inFile = sys.argv[1]

outFile = sys.argv[2]

preproc.load_data(
    inFile      = inFile,
    outFile     = outFile)
