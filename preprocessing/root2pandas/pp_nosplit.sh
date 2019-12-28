
#!/bin/bash
export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch
source $VO_CMS_SW_DIR/cmsset_default.sh

cd /nfs/dust/cms/user/pkraemer/CMSSW/9_4_10/CMSSW_9_4_10/src
eval `scramv1 runtime -sh`

cd -

python preprocessing.py -o DNN_Input_Reco_nosplit_allVars -v allVars_expanded
