import root2pandas


# define a base event selection which is applied for all Samples
base_selection = "\
( \
(N_Jets >= 4 and N_BTagsM >= 3 and Evt_Pt_MET > 20. and Weight_GEN_nom > 0.) \
and (\
(N_LooseMuons == 0 and N_TightElectrons == 1 and (Triggered_HLT_Ele35_WPTight_Gsf_vX == 1 or Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX == 1)) \
or \
(N_LooseElectrons == 0 and N_TightMuons == 1 and Muon_Pt > 29. and Triggered_HLT_IsoMu27_vX == 1) \
) \
)"


# define other additional selections
ttH_selection = "(Evt_Odd == 1)"

ttbar_selection = "(\
abs(Weight_scale_variation_muR_0p5_muF_0p5) <= 100 and \
abs(Weight_scale_variation_muR_0p5_muF_1p0) <= 100 and \
abs(Weight_scale_variation_muR_0p5_muF_2p0) <= 100 and \
abs(Weight_scale_variation_muR_1p0_muF_0p5) <= 100 and \
abs(Weight_scale_variation_muR_1p0_muF_1p0) <= 100 and \
abs(Weight_scale_variation_muR_1p0_muF_2p0) <= 100 and \
abs(Weight_scale_variation_muR_2p0_muF_0p5) <= 100 and \
abs(Weight_scale_variation_muR_2p0_muF_1p0) <= 100 and \
abs(Weight_scale_variation_muR_2p0_muF_2p0) <= 100 \
)"




# initialize dataset class
dataset = root2pandas.Dataset(
    outputdir   = "/nfs/dust/cms/user/vdlinden/MLfoy/workdir/DNNInputFiles/",
    naming      = "_dnn",
    addCNNmap   = False,
    addMEM      = True)

# add base event selection
dataset.addBaseSelection(base_selection)



# add samples to dataset
dataset.addSample(
    sampleName  = "ttHbb",
    ntuples     = "/nfs/dust/cms/user/kelmorab/ttH_2018/ntuples_forDNN_v4/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root",
    selections  = ttH_selection,
    MEMs        = "/nfs/dust/cms/user/vdlinden/MEM_2017/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8_MEM/*.root",
    CNNmaps     = "/nfs/dust/cms/user/vdlinden/DRACO-MLfoy/workdir/miniAOD_files/CNN_files/ttHbb.h5")
    
dataset.addSample(
    sampleName  = "ttHNobb",
    ntuples     = "/nfs/dust/cms/user/kelmorab/ttH_2018/ntuples_forDNN_v4/ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root",
    selections  = ttH_selection,
    MEMs        = "/nfs/dust/cms/user/vdlinden/MEM_2017/ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8/*.root",
    CNNmaps     = "/nfs/dust/cms/user/vdlinden/DRACO-MLfoy/workdir/miniAOD_files/CNN_files/ttHNobb.h5")



dataset.addSample(
    sampleName  = "TTToSL",
    ntuples     = "/nfs/dust/cms/user/kelmorab/ttH_2018/ntuples_forDNN_v4/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root",
    selections  = ttbar_selection,
    MEMs        = "/nfs/dust/cms/user/vdlinden/MEM_2017/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_MEM/*.root",
    CNNmaps     = "/nfs/dust/cms/user/vdlinden/DRACO-MLfoy/workdir/miniAOD_files/CNN_files/TTToSL.h5")

dataset.addSample(
    sampleName  = "TTToHad",
    ntuples     = "/nfs/dust/cms/user/kelmorab/ttH_2018/ntuples_forDNN_v4/TTToHadronic_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root",
    selections  = ttbar_selection,
    MEMs        = "/nfs/dust/cms/user/vdlinden/MEM_2017/TTToHadronic_TuneCP5_13TeV-powheg-pythia8_MEM/*.root",
    CNNmaps     = "/nfs/dust/cms/user/vdlinden/DRACO-MLfoy/workdir/miniAOD_files/CNN_files/TTToHad.h5")

dataset.addSample(
    sampleName  = "TTToLep",
    ntuples     = "/nfs/dust/cms/user/kelmorab/ttH_2018/ntuples_forDNN_v4/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root",
    selections  = ttbar_selection,
    MEMs        = "/nfs/dust/cms/user/vdlinden/MEM_2017/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_MEM/*.root",
    CNNmaps     = "/nfs/dust/cms/user/vdlinden/DRACO-MLfoy/workdir/miniAOD_files/CNN_files/TTToLep.h5")



# initialize variable list 
variableListModule = 
dataset.initVariableList(



