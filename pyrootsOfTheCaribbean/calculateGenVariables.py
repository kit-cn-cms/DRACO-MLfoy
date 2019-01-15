import os
import sys
import glob
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(filedir)

import miniAODplotting.sampleProcessor as processor
import miniAODplotting.NAFSubmit as NAFSubmit

samples = {}
ttH = "/pnfs/desy.de/cms/tier2/store/user/mwassmer/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/KIT_tthbb_sl_skims_MC_new_pmx_94X/181019_155115/0000/*.root"
samples["ttH"] = ttH

ttZ = "/pnfs/desy.de/cms/tier2/store/user/mwassmer/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/KIT_tthbb_sl_skims_MC_v2_94X/181022_163656/0000/*.root"
samples["ttZ"] = ttZ

output_dir = filedir + "/miniAODGenLevelData/"
h5_files = glob.glob(output_dir+"/*.h5")
for f in h5_files: os.remove(f)

shellscripts = processor.generate_submit_scripts(samples, output_dir, filedir)
jobids = NAFSubmit.submitToBatch(output_dir, shellscripts)



