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
    - adjust settings in `preprocessing/root2pandas/preprocessing.py` 
    - execute it on the NAF (tested with `CMSSW_9_4_9`)
    - this creates one dataset for each event category you specified in `preprocessing.py`, copy these datasets to your GPU machine
2. Setup training:
    - move to directory `train_scripts/` and create a config script for your training (for simple DNN training adjust `train_scripts/DNN_template.py`)
    - for the jet-tag categories `4j_ge3t`, `5j_ge3t`, `ge6j_ge3t` default net architectures and settings are defined in `DRACO_Frameworks/DNN/architecture.py`
3. Execute training:
    - adjust settings to network structure in `DRACO_Frameworks/DNN/architecture.py` or give a selfbuilt model as an argument to `dnn.build_model()`
    - execute the trainig script with your jet-tag category as a first argument
    - after completion of the script you can look at results in the specified output directory


## preprocessing

collection of scripts for preprocessing of ntuples/miniAOD files to hdf5 files as input for DRACOs
- `miniAOD-preprocessing`: generate CNN input maps from miniAOD files
- `root2pandas`: generate hdf5 files from ntuples + MEM files + CNN maps        

## pyrootsOfTheCaribbean

collection of scripts for plotting 
- `miniAOD_visualization`: plot CNN maps from miniAOD files
- `plot_configs`: collection of scripts for configuring plots

## variable_sets

lists of input variables used for dnn trainings

## studies

collection of studies

## train_scripts

collection of top-level scripts for training

## DRACO_Frameworks

collections of frameworks for training and evaluation of different machine learning frameworks

