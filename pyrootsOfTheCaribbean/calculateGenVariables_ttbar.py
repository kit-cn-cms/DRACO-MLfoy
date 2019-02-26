from ROOT import PyConfig, gROOT
PyConfig.IgnoreCommandLineOptions = True
gROOT.SetBatch(True)
import optparse
import os
import sys
import glob
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(filedir)
sys.path.append(basedir)

import miniAODplotting.sampleProcessor as processor
import utils.NAFSubmit as NAFSubmit
import sample_configs.mAODsamples_unskimmed as sampleConfig

parser = optparse.OptionParser(usage="%prog [options]")
parser.add_option("-o","--output",dest="out_dir",metavar="OUTPUT",
    default = "/miniAODGenLevelData/ttbarSystem/",
    help = "output directory relative to workdir")
parser.add_option("--noSubmit",dest="doSubmit",action="store_false",default=True,metavar="SUBMIT",
    help = "dont submit files to batch system - just write shell scripts and exit")
parser.add_option("--onlyConcat",dest="onlyConcat",action="store_true",default=False,metavar="ONLYCONCAT",
    help = "only concatenate the outputfiles, dont submit stuff")
(opts, args) = parser.parse_args()


samples = {}
samples["ttZJets"] = {
    "data":     sampleConfig.get_ttZJets("samples"),
    "XSWeight": sampleConfig.get_ttZJets("XSWeight")}

samples["ttZll"] = {
    "data":     sampleConfig.get_ttZll("samples"),
    "XSWeight": sampleConfig.get_ttZll("XSWeight")}

samples["ttZqq"] = {
    "data":     sampleConfig.get_ttZqq("samples"),
    "XSWeight": sampleConfig.get_ttZqq("XSWeight")}

samples["ttHbb"] = {
    "data":     sampleConfig.get_ttHbb("samples"),
    "XSWeight": sampleConfig.get_ttHbb("XSWeight")}

samples["ttSL"] = {
    "data":     sampleConfig.get_ttSL("samples"),
    "XSWeight": sampleConfig.get_ttSL("XSWeight")}

# output directory
output_dir = basedir + "/workdir/"+opts.out_dir+"/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

variableProcessors = [
    "ttbarAngles",
    "ttbarAngularDifferences_FullAngles",
    "ttbarAngularDifferences_LeptonBottom",
    "ttbarAngularDifferences_TopDecays",
    #"ttbarHelicityVariables"
    "ttXAngles",
    ]

# create shell scripts
shellscripts, sample_parts = processor.generate_submit_scripts(samples, output_dir, filedir, variableProcessors)

# submit them to naf
if opts.doSubmit and not opts.onlyConcat:
    jobids = NAFSubmit.submitToBatch(output_dir, shellscripts)
    NAFSubmit.monitorJobStatus(jobids)

# concatenate shell scripts
if opts.doSubmit:
    processor.concat_samples(sample_parts, output_dir)



