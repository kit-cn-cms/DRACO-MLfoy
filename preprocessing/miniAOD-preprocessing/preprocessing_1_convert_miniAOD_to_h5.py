import NAFSubmit
import os

ttHbb_files = [
    "/pnfs/desy.de/cms/tier2/store/user/mwassmer/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/KIT_tthbb_sl_skims_MC_94X/180617_093100/0000/Skim_1.root",
    "/pnfs/desy.de/cms/tier2/store/user/mwassmer/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/KIT_tthbb_sl_skims_MC_94X/180617_093100/0000/Skim_2.root",
    ]


ttZqq_files = [
    "/pnfs/desy.de/cms/tier2/store/user/pkeicher/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/KIT_tthbb_sl_skims_MC_94X/180618_082211/0000/Skim_1.root",
    "/pnfs/desy.de/cms/tier2/store/user/pkeicher/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/KIT_tthbb_sl_skims_MC_94X/180618_082211/0000/Skim_2.root",
    ]

# !!! change this !!!
basedir = "/nfs/dust/cms/user/vdlinden/DRACO-MLfoy/workdir"

print("writing shell scripts for ttH")
ttH_scripts = NAFSubmit.writeShellScripts( 
    workdir     = basedir,
    inFiles     = ttHbb_files, 
    nameBase    = "ttHbb")

print("writing shell scripts for ttZ")
ttZ_scripts = NAFSubmit.writeShellScripts(
    workdir     = basedir,
    inFiles     = ttZqq_files,
    nameBase    = "ttZqq")

print("submitting scripts to batch...")
NAFSubmit.submitToBatch( 
    workdir            = basedir,
    list_of_shells     = ttH_scripts + ttZ_scripts )

