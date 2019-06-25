#!/bin/bash

TAG=test_190623_nevt10_cate9

rm -rf workdir/${TAG}
python -B preprocessing/root2pandas/preprocessing.py -o ${TAG} -t liteTreeTTH_step7_cate9

rm -rf workdir/${TAG}_training
python -B train_scripts/train_ttHbb_DL_binary.py -i ${TAG} -o ${TAG}_training \
  -n _dnn.h5 -c ge4j_ge3t -v DL_variables --netconfig binary_DL --binary --signalclass ttHbb --balanceSamples

unset -v TAG