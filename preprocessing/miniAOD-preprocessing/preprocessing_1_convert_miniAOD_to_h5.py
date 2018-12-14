import NAFSubmit
import os
import glob
import miniAOD_preprocessing as preproc

# !!! adjustable options !!!
basedir = "/nfs/dust/cms/user/vdlinden/DRACO-MLfoy/workdir/miniAOD_files/"
data_type = "cnn_map" # unused option but needs to be set
test_run = False


# sample definitions
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

all_samples = ttH+ttbar



# =================================================================================================
def generate_submit_scripts(samples, basedir, data_type, test_run):
    scripts_to_submit = []
    for sample in samples:
        scripts = preprocess_single_sample( sample, basedir, data_type, test_run = test_run )
        scripts_to_submit += scripts

    print("total number of files: "+str(len(scripts_to_submit)))
    return scripts_to_submit




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

def concat_output_files(samples, out_path):
    print("concatenate output files ...")
    out_path = basedir + "/CNN_files"

    sample_dict = {}
    for sample in samples:
        name = sample["name"]
        sample_dict[name] = basedir+"out_files/"+str(name)+"_*.h5"

    for sample in sample_dict:
        print("{} files: {}".format(sample, sample_dict[sample]))

    preproc.add_output_files(
        sample_dict = sample_dict,
        out_path    = out_path)
# =================================================================================================




# write one shell script per root file
shell_scripts = generate_submit_scripts(all_samples, basedir, data_type, test_run)

print("submitting scripts to batch...")
jobIDs = NAFSubmit.submitToBatch( 
    workdir            = basedir,
    list_of_shells     = shell_scripts )
# monitor jobs
NAFSubmit.monitorJobStatus(jobIDs)
# concatenate the generated output files
concat_output_files(all_samples, basedir)

