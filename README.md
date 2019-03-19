# DRACO-MLfoy

Collection of Machine learning frameworks for DNNs, Regressions, Adversaries, CNNs and Others (DRACO)

## ATTENTION
When working with the combination of
- `ROOT.__version__ = 6.14/04`
- `keras.__version__ = 2.2.4`
- `pandas.__version__ = 0.23.4`
segmentation violations appear when opening hdf-files with `pandas.read_hdf()` or `pandas.HDFStore()`.
To circumvent this, first import everything related to `ROOT`, only then start importing things from `keras`.

## workflow in a nutshell
1. Preprocess input data:
    - to create input features you need data in the ntuple format, these get converted into hdf5 dataframes
    - create a list of input features in `variable_sets/` (look at the README for structural advice)
    - adjust settings in `preprocessing/root2pandas/preprocessing.py`, check README 
    - execute it on the NAF (tested with `CMSSW_9_4_9`)
    - this creates one dataset for each event category you specified in `preprocessing.py`
2. Setup training:
    - move to directory `train_scripts/` and create a config script for your training (for simple DNN training adjust `train_scripts/train_template.py`)
    - look at the README for further instructions
3. Execute training
    - execute the trainig script with your jet-tag category as option (further parser options described in README)
    - after completion of the script you can look at results in the specified output directory


## preprocessing

collection of scripts for preprocessing of ntuples files to hdf5 files as input for DRACOs
- `root2pandas`: generate hdf5 files from ntuples + MEM files      

## train_scripts

collection of top-level scripts for training

## pyrootsOfTheCaribbean

collection of scripts for plotting 
- `plot_configs`: collection of scripts for configuring plots

## variable_sets

lists of input variables used for dnn trainings


## DRACO_Frameworks

collections of frameworks for training and evaluation of different machine learning frameworks

