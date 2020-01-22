#!/bin/bash

for varset in topVariables_validated_decorr_2018vars #AngleFeaturesOnly_S01 #allVars allVars_noReco RecoVarsOnly
do
    #for f in ./${varset}/splitTTToSemileptonic*
    for f in ./${varset}/sp*
    do
        a=$(basename ${f})
        #echo ${f} ./${varset}/sp_v02${a#*sp_v01}
        #mv ${f} ./${varset}/sp${a#*splitTTToSemileptonic}
        mv ${f} ./${varset}/sp_${a#*sp__}
    done
done
