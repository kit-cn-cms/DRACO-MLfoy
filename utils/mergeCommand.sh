#!/bin/bash

# python mergeVariableSets.py --jt=4j_ge3t -o ../variable_sets/MergeAll_S01.py ../variable_sets/allVars_S01.py ../variable_sets/allVars_bin_S01.py ../variable_sets/RecoVarsOnly_S01.py ../variable_sets/RecoVarsOnly_bin_S01.py ../variable_sets/allVars_noReco_S01.py ../variable_sets/allVars_noReco_bin_S01.py


#python mergeVariableSets.py --jt=4j_ge3t,5j_ge3t,ge6j_ge3t,ge4j_3t,ge4j_ge4t -o ../variable_sets/MergeAll_S01.py ../variable_sets/allVars_S01.py ../variable_sets/allVars_bin_S01.py ../variable_sets/RecoVarsOnly_S01.py ../variable_sets/RecoVarsOnly_bin_S01.py ../variable_sets/allVars_noReco_S01.py ../variable_sets/allVars_noReco_bin_S01.py

python mergeVariableSets.py --jt=4j_ge3t,5j_ge3t,ge6j_ge3t,ge4j_3t,ge4j_ge4t -o ../variable_sets/MergeNoRecoReco_S02.py ../variable_sets/RecoVarsOnly_S02.py ../variable_sets/RecoVarsOnly_bin_S02.py ../variable_sets/allVars_noReco_S02.py ../variable_sets/allVars_noReco_bin_S02.py

#python mergeVariableSets.py --jt=4j_ge3t,5j_ge3t,ge6j_ge3t,ge4j_3t,ge4j_ge4t -o ../variable_sets/MergeAllExclBin_S01.py ../variable_sets/allVars_S01.py ../variable_sets/RecoVarsOnly_S01.py ../variable_sets/allVars_noReco_S01.py 

python mergeVariableSets.py --jt=4j_ge3t,5j_ge3t,ge6j_ge3t,ge4j_3t,ge4j_ge4t -o ../variable_sets/MergeNoRecoRecoExclBin_S02.py ../variable_sets/RecoVarsOnly_S02.py ../variable_sets/allVars_noReco_S02.py 


