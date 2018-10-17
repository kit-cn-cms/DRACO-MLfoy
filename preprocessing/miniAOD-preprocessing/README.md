Preprocessing of miniAOD files to h5 dataframes
===============================================

## Workflow
* add lists of miniAOD-files to `preprocessing_1` and create a single output file for each event class
* by executing `preprocessing_1` dataframes are written by parallel batch jobs
* to adjust the data that is written to the files, adjust the `hdfConfig` in `preprocess_single_file` or add stuff to the `read_event` function in `miniAOD_preprocessing`
* monitor the status of batch jobs via `condor_q` (console command) and check log files after termination
* to create a dataset for training execute `preprocessing_2` with adjusted paths and datasets
* move the training set to your local GPU machine, e.g. `gpumonster` to perform training


## non Parallel creation of hdf5 dataframe
* execute `preprocessing_single_file` with the following arguments:
    1. path to input miniAOD-file
    2. path to output hdf5-file
    3. name of dataframe in file

