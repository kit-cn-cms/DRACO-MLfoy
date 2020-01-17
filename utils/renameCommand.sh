#!/bin/bash

for varset in allVars allVars_noReco RecoVarsOnly
do
    for f in ./${varset}/splitTTToSemileptonic*
    do
        a=$(basename ${f})
#         echo ${f} ./${varset}/sp${a#*splitTTToSemileptonic}
        mv ${f} ./${varset}/sp${a#*splitTTToSemileptonic}
    done
done