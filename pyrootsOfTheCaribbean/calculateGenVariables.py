import os
import sys
import glob
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(filedir)

import miniAODplotting.sampleProcessor as processor
import miniAODplotting.NAFSubmit as NAFSubmit

samples = {}
samples["ttH"] = {"data": "/pnfs/desy.de/cms/tier2/store/user/mwassmer/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/KIT_tthbb_sl_skims_MC_new_pmx_94X/181019_155115/0000/*.root",
                "XSWeight": 7.00166976289E-05}

samples["ttZ"] = {"data": "/pnfs/desy.de/cms/tier2/store/user/mwassmer/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/KIT_tthbb_sl_skims_MC_v2_94X/181022_163656/0000/*.root",
                "XSWeight": 0.0013842102}

output_dir = basedir + "/workdir/miniAODGenLevelData/"
h5_files = glob.glob(output_dir+"/*.h5")
for f in h5_files: os.remove(f)

shellscripts, sample_parts = processor.generate_submit_scripts(samples, output_dir, filedir)
jobids = NAFSubmit.submitToBatch(output_dir, shellscripts)
NAFSubmit.monitorJobStatus(jobids)

processor.concat_samples(sample_parts, output_dir)



