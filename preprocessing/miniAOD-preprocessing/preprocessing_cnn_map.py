# script to be executed by batch jobs
import miniAOD_preprocessing as preproc
import numpy as np
import sys

# pixel size definition
d_eta = 0.1
d_phi = 0.1*180./np.pi
# range definitions
eta_range = [-2.5,2.5]
phi_range = [-180., 180.]

# pixel converter
n_pix_eta = int( (eta_range[1]-eta_range[0])/d_eta )
n_pix_phi = int( (phi_range[1]-phi_range[0])/d_phi )
print("creating image with size "+str(n_pix_eta)+" x "+str(n_pix_phi)+" pixels")
hdfConfig = preproc.HDFConfig(
    # only one information layer atm, thus last entry is 1
    imageSize   = [n_pix_eta, n_pix_phi, 1],
    etaRange    = eta_range,
    phiRange    = phi_range,       
    # pixel intensity linear or logarithmic
    logNorm     = False)

inFile = sys.argv[1]
outFile = sys.argv[2]

print("using infile:  "+str(inFile))
print("using outfile: "+str(outFile))

preproc.load_data(
    inFile      = inFile,
    outFile     = outFile,
    hdfConfig   = hdfConfig)
