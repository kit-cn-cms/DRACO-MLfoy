#!/bin/bash

#Preprocessing and multi-class/binary training steps for cate9
TAG=test_230719_cate9

rm -rf workdir/${TAG}
python -B preprocessing/root2pandas/preprocessing_ttHbb_DL.py -o ${TAG}/cate9 -t liteTreeTTH_step7_cate9

rm -rf workdir/${TAG}_training_multiclass
python -B train_scripts/train_ttHbb_DL.py -i ${TAG}/cate9 -o ${TAG}_training_multiclass/cate9 -v variables_ttHbb_DL -n ttHbb_2017_DL -c ge4j_ge3t -p -R -e 50

rm -rf workdir/${TAG}_training_binary
python -B train_scripts/train_ttHbb_DL.py -i ${TAG}/cate9 -o ${TAG}_training_binary/cate9 -v variables_ttHbb_DL -n binary_DL -c ge4j_ge3t -p -R -e 50 --binary -t -1. --signal ttH
unset -v TAG



#Preprocessing and multi-class training steps for cate9
TAG=test_230719_cate8

rm -rf workdir/${TAG}
python -B preprocessing/root2pandas/preprocessing_ttHbb_DL.py -o ${TAG}/cate8 -t liteTreeTTH_step7_cate8

rm -rf workdir/${TAG}_training
python -B train_scripts/train_ttHbb_DL.py -i ${TAG}/cate8 -o ${TAG}_training/cate8 -v variables_ttHbb_DL -n ttHbb_2017_DL -c ge4j_ge4t -p -R -e 50

rm -rf workdir/${TAG}_training_binary
python -B train_scripts/train_ttHbb_DL.py -i ${TAG}/cate8 -o ${TAG}_training_binary/cate8 -v variables_ttHbb_DL -n ttHbb_2017_DL -c ge4j_ge4t -p -R -e 50 --binary -t -1 --signal ttH

unset -v TAG
