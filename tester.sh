#!/bin/bash

rm -rf workdir/test_190623_nevt10_cate9
python -B preprocessing/root2pandas/preprocessing.py -o test_190623_nevt10_cate9 -t liteTreeTTH_step7_cate9

rm -rf workdir/test_190623_nevt10_cate9_training                                                                                                                                  
python -B train_scripts/train_ttHbb_DL_binary.py -i test_190623_nevt10_cate9 -o test_190623_nevt10_cate9_training \
  -n _dnn.h5 -c ge4j_ge3t -v DL_variables --netconfig binary_DL --binary --signalclass ttHbb --balanceSamples
