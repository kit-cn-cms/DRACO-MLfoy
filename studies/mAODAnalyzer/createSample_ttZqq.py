import os
import sys
import glob
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
studiesdir = os.path.dirname(filedir)
basedir = os.path.dirname(studiesdir)
sys.path.append(basedir)

import pyrootsOfTheCaribbean.miniAODplotting.NAFSubmit as NAFSubmit
import sampleProcessor

samples = ["root://xrootd-cms.infn.it//"+str(i) for i in [
    "/store/mc/RunIIFall17MiniAODv2/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/270000/F225C5E3-FECD-E811-9141-6CC2173DAD00.root",
    "/store/mc/RunIIFall17MiniAODv2/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/270000/E2877F62-00CE-E811-9246-6CC2173DBF20.root",
    "/store/mc/RunIIFall17MiniAODv2/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/270000/DC931BE5-FECD-E811-A1BD-6CC2173D9AB0.root",
    "/store/mc/RunIIFall17MiniAODv2/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/270000/D8E2D601-FACD-E811-B759-6CC2173DA2F0.root",
    "/store/mc/RunIIFall17MiniAODv2/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/270000/D0C17C59-02CE-E811-B09C-6CC2173DCB90.root",
    "/store/mc/RunIIFall17MiniAODv2/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/270000/C28FEACF-F4CD-E811-AF15-6CC2173DA9E0.root",
    "/store/mc/RunIIFall17MiniAODv2/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/270000/982A80C8-F0CD-E811-BB30-6CC2173D9300.root",
    "/store/mc/RunIIFall17MiniAODv2/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/270000/900F992A-F8CD-E811-9953-6CC2173DA2F0.root",
    "/store/mc/RunIIFall17MiniAODv2/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/270000/70326EC6-01CE-E811-BAE9-6CC2173DA9E0.root",
    "/store/mc/RunIIFall17MiniAODv2/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/270000/68320DC6-01CE-E811-9542-6CC2173DA9E0.root",
    "/store/mc/RunIIFall17MiniAODv2/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/270000/5E6888E0-FECD-E811-944A-6CC2173DA930.root",
    "/store/mc/RunIIFall17MiniAODv2/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/270000/5CCB6B52-02CE-E811-AB05-6CC2173DBD00.root",
    "/store/mc/RunIIFall17MiniAODv2/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/270000/54358862-00CE-E811-85DA-6CC2173D9300.root",
    "/store/mc/RunIIFall17MiniAODv2/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/270000/4C05D8DD-FECD-E811-9B69-6CC2173DBF20.root",
    "/store/mc/RunIIFall17MiniAODv2/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/270000/321D00E0-FECD-E811-8BF3-6CC2173D9300.root",
    "/store/mc/RunIIFall17MiniAODv2/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v2/270000/2A9ECFDD-FECD-E811-B5A6-6CC2173DA2F0.root",
    ]]

output_dir = filedir + "/output/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
outfile = output_dir+"/ttZqq.h5"

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

