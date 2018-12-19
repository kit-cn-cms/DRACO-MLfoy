# DRACO-MLfoy

Collection of Machine learning frameworks for DNNs, Regressions, Adversaries, CNNs and Others (DRACO)

## ATTENTION
When working with the combination of
- `ROOT.__version__ = 6.14/04`
- `keras.__version__ = 2.2.4`
- `pandas.__version__ = 0.23.4`
segmentation violations appear when opening hdf-files with `pandas.read_hdf()` or `pandas.HDFStore()`.
To circumvent this, first import everything related to `ROOT`, only then start importing things from `keras`.


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

