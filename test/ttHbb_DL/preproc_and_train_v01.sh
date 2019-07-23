#!/bin/bash

TAG=test_230719_cate9

rm -rf workdir/${TAG}
python -B preprocessing/root2pandas/preprocessing.py -o ${TAG} -t liteTreeTTH_step7_cate9

rm -rf workdir/${TAG}_training

python -B train_scripts/train_ttHbb_DL.py -i prova/cate8 -o training_prova/cate8 -v DL_variables -n ttHbb_2017_DL -c ge4j_ge4t -p -R -P
python -B train_scripts/train_ttHbb_DL.py -i ${TAG} -o ${TAG}_training \
  -n ttHbb_2017_DL -c ge4j_ge3t -v DL_variables -p -R 

unset -v TAG
