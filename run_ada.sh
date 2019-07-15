#!/usr/bin/env bash

epoches=("50" "100" "200")
ada_epoches=("25" "50" "100" "150" "200" "300" "500")

for i in ${ada_epoches[*]};
do
  for j in ${epoches[*]};
  do
    echo $j
    wait
  done
done
