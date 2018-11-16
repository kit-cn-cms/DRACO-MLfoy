# ttH 2017 Analysis (with Aachen DNNs)
This branch saves the status with which the training of DNNs was performed for the first Status update of the ttH Analysis with 2017 Data

## preprocessing
`preprocessing/root2pandas/convert_root_2_dataframe.py`

Samples:
```
/nfs/dust/cms/user/kelmorab/ttH_2018/ntuples_forDNN_v2/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root
/nfs/dust/cms/user/kelmorab/ttH_2018/ntuples_forDNN_v2/ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root
/nfs/dust/cms/user/kelmorab/ttH_2018/ntuples_forDNN_v2/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root
/nfs/dust/cms/user/kelmorab/ttH_2018/ntuples_forDNN_v2/TTToHadronic_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root
/nfs/dust/cms/user/kelmorab/ttH_2018/ntuples_forDNN_v2/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root
```
Selection:
```
(N_Jets >= 4 and N_BTagsM >= 3 and Evt_Pt_MET > 20. and Weight_GEN_nom > 0.) 
and (
(N_LooseMuons == 0 and N_TightElectrons == 1 and (Triggered_HLT_Ele35_WPTight_Gsf_vX == 1 or Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX == 1)) 
or 
(N_LooseElectrons == 0 and N_TightMuons == 1 and Muon_Pt > 29. and Triggered_HLT_IsoMu27_vX == 1) 
) 
```
Additional ttbar selection:
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

## training
top level train script: `train_scripts/train_Aachen_DNN.py [category]`

input variables: `preprocessing/root2pandas/variable_info.py`

net confugurations: `DRACO_Frameworks/DNN_Aachen/DNN_Architectures.py`

### `4j_ge3t`
`(N_Jets == 4 and N_BTagsM >= 3)`

### `5j_ge3t`
`(N_Jets == 5 and N_BTagsM >= 3)`

### `ge6j_ge3t`
`(N_Jets >= 6 and N_BTagsM >= 3)`

