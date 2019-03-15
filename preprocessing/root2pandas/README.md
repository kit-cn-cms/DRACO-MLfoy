# create input files for NN training

## Before Usage
execute on NAF with `CMSSW_9_4_9` 
uproot needs to be installed
```bash
pip install --user uproot
```

## Adjust settings in `preprocessing.py`
- `base_selection` to define a base event selection which is applied for all Samples
- change/add event categories with `EVENTCATEGORYNAME=root2pandas.EventCategories()` (default event categories are `ttH_categories` and `ttbar_categories`)
- change/add categories of event categories with `EVENTCATEGORYNAME.addCategory(CATEGORYNAME, selection = SELECTION)` (`SELECTION` can be `None`)
- absolute path to ntuples and MEM: `ntuplesPath` and `memPath`
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
```bash
python preprocessing.py
```
To execute with default options or use 
- `-o DIR` to change the name of the ouput directory, can be either a string or absolute path (default is `InputFeatures`)
- `-v FILE` to change the variable Selection, if the file is in `/variable_sets/` the name is sufficient, else the absolute path is needed (default is `NewJEC_top20Variables`)
- `-e INT` to change the maximal number of entries for each batch to prevent NAF from crashing (default is `50000`)
- `-n STR` to change the naming of the output file

```bash
python preprocessing.py -o DIR -v FILE -e INT -m BOOL -n STR
```

