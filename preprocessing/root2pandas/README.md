# Create input files for NN training

## Before Usage
execute on NAF with `CMSSW_9_4_9` or newer

uproot needs to be installed locally as it is not a part of the CMSSW package
```bash
pip install --user uproot==3.2.7
pip install --user uproot-methods==0.2.6
pip install --user awkward==0.4.2
```

## Adjust settings in `preprocessing.py`
- `base_selection` to define a base event selection which is applied for all Samples 
   
   (default `base_selection = "(N_Jets >= 4 and N_BTagsM >= 3)"`)
- change/add event categories (default event categories are `ttH_categories` and `ttbar_categories`)
```python
	EVENTCATEGORYNAME=root2pandas.EventCategories()
```
- change/add categories of event categories (`SELECTION` can be `None`)
```python
   EVENTCATEGORYNAME.addCategory(CATEGORYNAME, selection = SELECTION)
 ``` 
- `ntuplesPath` as absolute path to ntuples
- `memPath` absolute path to MEMs, usage of MEMs is optional
- change/add samples of the dataset used with 
```python
	dataset.addSample(SampleName  = SAMPLENAME,
    			ntuples     = PATHTONTUPLES,
    			categories  = EVENTCATEGORYNAME,
    			selections  = SELECTION,
    			MEMs        = PATHTOMEMS`
```
- change `additional_variables` for variables needed to preprocess, that are not defined in the selection and not needed for training


## Usage
To execute with default options use
```bash
python preprocessing.py
```
or use the following for options
- `-o DIR` to change the name of the ouput directory, can be either a string or absolute path (default is `InputFeatures`)
- `-v FILE` to change the variable Selection, if the file is in `/variable_sets/` the name is sufficient, else the absolute path is needed (default is `example_variables`)
- `-e INT` to change the maximal number of entries for each batch to restrict memory usage (default is `50000`)
- `-n STR` to change the naming of the output file
- `-m` to activate using MEMs

```bash
python preprocessing.py -o DIR -v FILE -e INT -m -n STR
```

## Concerning MEMs
In the current ntuple setup the MEM likelihood variables are not written directly to ntuples, but are saved separately.

To include MEM variables in the DNN training these need to be read and added separately. For the single-lepton analysis the MEMs are saved in `.root` files with trees called `tree`.

The function `root2pandas.generateMEMdf` reads the MEM values (more specifically `mem_p`) and event IDs (`run`, `lumi`, `event`) from these files and saves the information as separate pandas dataframes.
At later steps in the `preprocessing` these MEM values are added to the output files. Only events where a MEM value was found (matching IDs) are added to the output files.

The MEM variable is saved as `memDBp` in the output files to match the naming scheme used in the single-lepton analysis.

## Concerning friend trees
variables from other trees can be added as long as the structure of the input files is the same. I.e. the file structure
```
basepath/
----/sample/
----/----/file1.root
----/----/file2.root
```
has to be retained. Also the order of events in the files has to be the same.
To add friend trees give a dictionary of friend trees to the `Dataset` class via
```
friendTrees = {
    "ftName": "new/base/path/",
    ...
```
Variables specifically from these friend trees can be obtained via `ftName_ft_varName` in the variable set files or additional variables list.
This is also the name of these variables in the resulting data frames.
Currently only non-vectorized variables are supported

------------------------------------------------------------------------------------------

## Instructions specific to ttHbb dilepton channel (inputs in DESY-format)

- `ntuplesPath` as absolute path to ntuples
- change `additional_variables` for variables needed to preprocess, that are not defined in the selection and not needed for training

## Usage
To execute with default options use
```bash
python preprocessing/root2pandas/preprocessing_ttHbb_DL.py -o cate8 -t liteTreeTTH_step7_cate8
```
or use the following for options
- `-o DIR` to change the name of the ouput directory, can be either a string or absolute path (default is `InputFeatures`)
- `-v FILE` to change the variable Selection, if the file is in `/variable_sets/` the name is sufficient, else the absolute path is needed (for DL analysis default is `variables_ttHbb_DL` )
- `-e STR` to select the tree corresponding to the right category  (default is `liteTreeTTH_step7_cate8`)
- `-e INT` to change the maximal number of entries for each batch to restrict memory usage (default is `50000`)
- `-n STR` to change the naming of the output file
- `-m` to activate using MEMs

```bash
python preprocessing_ttHbb_DL.py -o DIR -v FILE -e INT -m -n STR
```

------------------------------------------------------------------------------------------
