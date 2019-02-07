# pyrootsOfTheCaribbean

Collection of scripts and modules for plotting histograms with ROOT

## scripts

### `calculateGenVariables_X.py`

This script is used to calculate variables from information in miniAOD-level samples.
- specify the samples to process (from `<base>/sample_configs`)
- adjust the in/out paths to your liking
- executing the script generates hdf5-files in the specified output directory, containing the variables definied in `miniAODplotting/variableCalculations.py`

Disclaimer: most of the mAOD samples defined in `<base>/sample_configs` are not stored locally on the NAF-system, but on different GRID storages. Thus, you need a valid voms-proxy to access these samples, including an environment variable `X509_USER_PROXY` containing a path to your personal proxy (`export X509_USER_PROXY=/path/to/proxy/file`)

### `plotGenLevelVariables_X.py`

This script is used to generate plots for the variables, stored in the output files produced with `calculateGenVariables.py`.
- adujust paths
- choose plot options
- define which variables to plot
    - either choose a variable set from `<base>/variable_sets` and optionally `additional_variables`
    - or plot all variables in the input files by specifying `variable_set = None` 
- add samples to plot (`signalSamples` are shown as a line, `backgroundSamples` are shown as a filled stack)
- specify categories to plot (look at `<base>/utils/generateJTcut.py` on how to define a category)

### `plot2DGenLevelVariables.py`

This script is used to generate 2D plots showing the correlations of two variables, stored in the output files produced with `calculateGenVariables.py`.
- adjust paths
- choose plot options
- define which variable pairs to plot in `variables` list
- add samples to plot
- specify categories to plot (look at `<base>/utils/generateJTcut.py` on how to define a category)


### `plotInputVariables.py`

This script is used to plot the input variables used for DNN training.
- adjust paths
- choose plot options
- specify a variable set from `<base>/variable_sets` and optionally `additional_variables`
- add samples to plot (`signalSamples` are shown as a line, `backgroundSamples` are shown as a filled stack)
- specify categories to plot (look at `<base>/utils/generateJTcut.py` on how to define a category)

