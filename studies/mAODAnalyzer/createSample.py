import optparse
import os
import sys
import glob
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
studiesdir = os.path.dirname(filedir)
basedir = os.path.dirname(studiesdir)
sys.path.append(basedir)

import pyrootsOfTheCaribbean.miniAODplotting.NAFSubmit as NAFSubmit
import sample_configs.mAODsamples_unskimmed as sampleConfig
import sampleProcessor

# parser
parser = optparse.OptionParser(usage="%prog [options]")
parser.add_option("-s","--sample",dest="samplename",default="ttZJets",metavar="SAMPLENAME",
    help="name of sample to process")
parser.add_option("--Submit",action="store_true",dest="submit",metavar="SUBMIT",default=True,
    help="submit jobs to naf batch system")
parser.add_option("--noSubmit", action="store_false",dest="submit",metavar="SUBMIT",
    help="omit submission of naf jobs")
(opts,args)=parser.parse_args()

if opts.samplename == "ttZJets":
    samples = sampleConfig.get_ttZJets("samples")
elif opts.samplename == "ttZqq":
    samples = sampleconfig.get_ttZqq("samples")
else:
    print("sample not defined (sorry for hardcoding")
    exit()

# output directory
output_dir = filedir + "/output/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
outfile = output_dir+"/{}.h5".format(opts.samplename)

# create shell scripts
shellscripts, output_parts = sampleProcessor.generate_submit_scripts(samples, output_dir, filedir)

submit = True
if len(sys.argv) > 1:
    if sys.argv[1] == "--noSubmit":
        submit = False
        
# submit them to naf
if submit:
    jobids = NAFSubmit.submitToBatch(output_dir, shellscripts)
    NAFSubmit.monitorJobStatus(jobids)

sampleProcessor.concat_samples(output_parts, outfile)

