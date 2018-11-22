import NAFSubmit
import os
import glob


ttH = [
        {"name":    "ttHbb",
         "ntuples": "/nfs/dust/cms/user/kelmorab/ttH_2018/ntuples_forDNN_v2/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root",
         "MEM":     "/nfs/dust/cms/user/vdlinden/MEM_2017/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8_MEM/*.root",
         "mAOD":    "/pnfs/desy.de/cms/tier2/store/user/mwassmer/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/KIT_tthbb_sl_skims_MC_new_pmx_94X/181019_155115/0000/*.root",},

        {"name":    "ttHNobb",
         "ntuples": "/nfs/dust/cms/user/kelmorab/ttH_2018/ntuples_forDNN_v2/ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root",
         "MEM":     "/nfs/dust/cms/user/vdlinden/MEM_2017/ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8/*.root",
         "mAOD":    "/pnfs/desy.de/cms/tier2/store/user/mwassmer/ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8/KIT_tthbb_sl_skims_MC_new_pmx_94X/181020_135500/0000/*.root",},
        ]

ttbar = [
        {"name":    "TTToSL",
         "ntuples": "/nfs/dust/cms/user/kelmorab/ttH_2018/ntuples_forDNN_v2/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root",
         "MEM":     "/nfs/dust/cms/user/vdlinden/MEM_2017/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_MEM/*.root",
         "mAOD":    "/pnfs/desy.de/cms/tier2/store/user/mwassmer/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/KIT_tthbb_sl_skims_MC_new_pmx_94X/181019_154801/0000/*.root",},

        {"name":    "TTToHad",
         "ntuples": "/nfs/dust/cms/user/kelmorab/ttH_2018/ntuples_forDNN_v2/TTToHadronic_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root",
         "MEM":     "/nfs/dust/cms/user/vdlinden/MEM_2017/TTToHadronic_TuneCP5_13TeV-powheg-pythia8_MEM/*.root",
         "mAOD":    "/pnfs/desy.de/cms/tier2/store/user/mwassmer/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/KIT_tthbb_sl_skims_MC_new_pmx_94X/181019_154333/0000/*.root",},

        {"name":    "TTToLep",
         "ntuples": "/nfs/dust/cms/user/kelmorab/ttH_2018/ntuples_forDNN_v2/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root",
         "MEM":     "/nfs/dust/cms/user/vdlinden/MEM_2017/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_MEM/*.root",
         "mAOD":    "/pnfs/desy.de/cms/tier2/store/user/mwassmer/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/KIT_tthbb_sl_skims_MC_new_pmx_94X/181019_153955/0000/*.root",}
        ]


def preprocess_single_sample(sample, basedir, data_type, test_run = False):
    mAOD_files = glob.glob(sample["mAOD"])
    sample_name = sample["name"]
    n_files = len(mAOD_files)
    
    print("="*100)
    print("now handling miniAOD files from "+str(sample_name))
    print("number of files: "+str(n_files))
    print("="*100)

    shell_scripts = NAFSubmit.writeShellScripts(
        workdir   = basedir,
        inFiles   = mAOD_files,
        nameBase  = sample_name,
        data_type = data_type,
        test_run  = test_run)
    
    return shell_scripts










# !!! change this !!!
basedir = "/nfs/dust/cms/user/vdlinden/DRACO-MLfoy/workdir/miniAOD_files/"
data_type = "cnn_map"
test_run = False

scripts_to_submit = []
for sample in ttH:
    scripts = preprocess_single_sample( sample, basedir, data_type, test_run = test_run )
    scripts_to_submit += scripts

for sample in ttbar:
    scripts = preprocess_single_sample( sample, basedir, data_type, test_run = test_run )
    scripts_to_submit += scripts

print(len(scripts_to_submit))


print("submitting scripts to batch...")
jobIDs = NAFSubmit.submitToBatch( 
    workdir            = basedir,
    list_of_shells     = scripts_to_submit )
NAFSubmit.monitorJobStatus(jobIDs)
