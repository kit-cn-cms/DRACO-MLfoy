# Create input files for NN training

## Before Usage
execute on NAF with `CMSSW_9_4_9` or newer

uproot needs to be installed locally as it is not a part of the CMSSW package
```bash
pip install --user uproot
```

## Adjust settings in `preprocessing.py`
- `base_selection` to define a base event selection which is applied for all Samples (default `base_selection = "(N_Jets >= 4 and N_BTagsM >= 3)"`)
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
- `-v FILE` to change the variable Selection, if the file is in `/variable_sets/` the name is sufficient, else the absolute path is needed (default is `NewJEC_top20Variables`)
- `-e INT` to change the maximal number of entries for each batch to prevent NAF from crashing (default is `50000`)
- `-n STR` to change the naming of the output file

```bash
python preprocessing.py -o DIR -v FILE -e INT -m BOOL -n STR
```

