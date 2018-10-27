import NAFSubmit
import os

ttHbb_files = [
    "/pnfs/desy.de/cms/tier2/store/user/mwassmer/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/KIT_tthbb_sl_skims_MC_94X/180617_093100/0000/Skim_1.root",
    "/pnfs/desy.de/cms/tier2/store/user/mwassmer/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/KIT_tthbb_sl_skims_MC_94X/180617_093100/0000/Skim_2.root",
    "/pnfs/desy.de/cms/tier2/store/user/mwassmer/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/KIT_tthbb_sl_skims_MC_94X/180617_093100/0000/Skim_3.root",
    "/pnfs/desy.de/cms/tier2/store/user/mwassmer/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/KIT_tthbb_sl_skims_MC_94X/180617_093100/0000/Skim_4.root",
    ]


ttZqq_files = [
    "/pnfs/desy.de/cms/tier2/store/user/pkeicher/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/KIT_tthbb_sl_skims_MC_94X/180618_082211/0000/Skim_1.root",
    "/pnfs/desy.de/cms/tier2/store/user/pkeicher/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/KIT_tthbb_sl_skims_MC_94X/180618_082211/0000/Skim_2.root",
    "/pnfs/desy.de/cms/tier2/store/user/pkeicher/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/KIT_tthbb_sl_skims_MC_94X/180618_082211/0000/Skim_3.root",
    "/pnfs/desy.de/cms/tier2/store/user/pkeicher/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/KIT_tthbb_sl_skims_MC_94X/180618_082211/0000/Skim_4.root",
    ]

# !!! change this !!!
basedir = "/nfs/dust/cms/user/vdlinden/DRACO-MLfoy/workdir"
data_type = "pf_cands" # "cnn_map"

print("writing shell scripts for ttH")
ttH_scripts = NAFSubmit.writeShellScripts( 
    workdir     = basedir,
    inFiles     = ttHbb_files, 
    nameBase    = "ttHbb_pfc",
    data_type   = data_type)

print("writing shell scripts for ttZ")
ttZ_scripts = NAFSubmit.writeShellScripts(
    workdir     = basedir,
    inFiles     = ttZqq_files,
    nameBase    = "ttZqq_pfc",
    data_type   = data_type)

print("submitting scripts to batch...")
NAFSubmit.submitToBatch( 
    workdir            = basedir,
    list_of_shells     = ttH_scripts + ttZ_scripts )

