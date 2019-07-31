#!/usr/bin/env bash

epoches=("50" "100") # "100" "150") #"200") -> overfitting
ada_epoches=("300")
# configs=("ada_weak1" "ada_weak2" "ada_weak3")
# configs=("ada_weak4" "ada_weak4_1" "ada_weak4_2")
configs=("ada_weak6_4_2_opt")
# configs=("ada_weak6_4" "ada_weak6_4_1" "ada_weak6_4_2" "ada_weak6_4_3") #test some loss functions

for i in ${ada_epoches[*]};
do
  for j in ${epoches[*]};
  do
    for conf in ${configs[*]};
    do
      python train_ada_template.py -i /ceph/swieland/ttH/h5Files/LegacyStrategy/Baseline -n _LegacyStrategyStudyBaseline.h5 --trainepochs $j --netconfig $conf --adaboost $i --binary -t -1 --signalclass ttH -c ge6j_ge3t --m2 True
      wait
      # python train_ada_template.py -i /ceph/swieland/ttH/h5Files/LegacyStrategy/Baseline -n _LegacyStrategyStudyBaseline.h5 --trainepochs $j --netconfig $conf --adaboost $i --binary -t -1 --signalclass ttH -c ge6j_ge3t
      # wait
    done
  done
done
