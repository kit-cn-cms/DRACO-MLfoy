#!/usr/bin/env bash

epoches=("50" "100" "200")
ada_epoches=("25" "50" "100" "150" "200" "300" "500")

for i in ${ada_epoches[*]};
do
  for j in ${epoches[*]};
  do
    python train_ada_template.py -i /ceph/swieland/ttH/h5Files/LegacyStrategy/Baseline -n _LegacyStrategyStudyBaseline.h5 --trainepochs $j --netconfig binary_config --adaboost $i --binary -t -1 --signalclass ttH -c ge6j_ge3t --m2 True
    wait
    python train_ada_template.py -i /ceph/swieland/ttH/h5Files/LegacyStrategy/Baseline -n _LegacyStrategyStudyBaseline.h5 --trainepochs $j --netconfig binary_config --adaboost $i --binary -t -1 --signalclass ttH -c ge6j_ge3t
    wait
  done
done
