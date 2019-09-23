#!/usr/bin/env bash

epoches=("15") # "100" "150") #"200") -> overfitting
ada_epoches=("10")
# ada_epoches=("100" "200")
# configs=("ada_weak1" "ada_weak2" "ada_weak3")
# configs=("ada_weak4" "ada_weak4_1" "ada_weak4_2")
configs=("ada_weak1_2_3_opt")
# configs=("ada_weak3")

for i in ${ada_epoches[*]};
do
  for j in ${epoches[*]};
  do
    for conf in ${configs[*]};
    do
      # python train_ada_template.py -i /ceph/swieland/ttH/h5Files/LegacyStrategy/Baseline -n _LegacyStrategyStudyBaseline.h5 --trainepochs $j --netconfig $conf --adaboost $i --binary -t -1 --signalclass ttH -c ge6j_ge3t --m2 True
      # wait
      python train_ada_template.py -i /ceph/swieland/ttH/h5Files/LegacyStrategy/Baseline -n _LegacyStrategyStudyBaseline.h5 --trainepochs $j --netconfig $conf --adaboost $i --binary -t -1 --signalclass ttH -c ge6j_ge3t --simultan 2
      wait
    done
  done
done
