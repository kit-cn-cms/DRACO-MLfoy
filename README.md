# DRACO-MLfoy

Collection of Machine learning frameworks for DNNs, Regressions, Adversaries, CNNs and Others (DRACO)

## ttH 2017 Analysis (with Aachen DNNs)
This branch saves the status with which the training of DNNs was performed for the first Status update of the ttH Analysis with 2017 Data

### preprocessing
`preprocessing/root2pandas/convert_root_2_dataframe.py`
Used samples:
```
/nfs/dust/cms/user/kelmorab/ttH_2018/ntuples_forDNN_v2/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root
/nfs/dust/cms/user/kelmorab/ttH_2018/ntuples_forDNN_v2/ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root
/nfs/dust/cms/user/kelmorab/ttH_2018/ntuples_forDNN_v2/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root
/nfs/dust/cms/user/kelmorab/ttH_2018/ntuples_forDNN_v2/TTToHadronic_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root
/nfs/dust/cms/user/kelmorab/ttH_2018/ntuples_forDNN_v2/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root
```
Used selection:
```
(N_Jets >= 4 and N_BTagsM >= 3 and Evt_Pt_MET > 20. and Weight_GEN_nom > 0.) 
and (
(N_LooseMuons == 0 and N_TightElectrons == 1 and (Triggered_HLT_Ele35_WPTight_Gsf_vX == 1 or Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX == 1)) 
or 
(N_LooseElectrons == 0 and N_TightMuons == 1 and Muon_Pt > 29. and Triggered_HLT_IsoMu27_vX == 1) 
) 
```
Used ttbar selection:
```
abs(Weight_scale_variation_muR_0p5_muF_0p5) <= 100 and 
abs(Weight_scale_variation_muR_0p5_muF_1p0) <= 100 and 
abs(Weight_scale_variation_muR_0p5_muF_2p0) <= 100 and 
abs(Weight_scale_variation_muR_1p0_muF_0p5) <= 100 and 
abs(Weight_scale_variation_muR_1p0_muF_1p0) <= 100 and
abs(Weight_scale_variation_muR_1p0_muF_2p0) <= 100 and
abs(Weight_scale_variation_muR_2p0_muF_0p5) <= 100 and
abs(Weight_scale_variation_muR_2p0_muF_1p0) <= 100 and
abs(Weight_scale_variation_muR_2p0_muF_2p0) <= 100 
```

### training
`train_scripts/train_Aachen_DNN.py`

`4j,ge3t` category:

`5j,ge3t` category:

`ge6j,ge3t` cateogry:

