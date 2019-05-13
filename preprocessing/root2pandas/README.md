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
- `ntuplesPath` as absolute path to ntuples
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
python preprocessing/root2pandas/preprocessing.py -o cate8 -t liteTreeTTH_step7_cate8
```
or use the following for options
- `-o DIR` to change the name of the ouput directory, can be either a string or absolute path (default is `InputFeatures`)
- `-v FILE` to change the variable Selection, if the file is in `/variable_sets/` the name is sufficient, else the absolute path is needed (default is `DL_variables`)
- `-e STR` to select the tree corresponding to the right category  (default is `liteTreeTTH_step7_cate8`)
- `-e INT` to change the maximal number of entries for each batch to restrict memory usage (default is `50000`)
- `-n STR` to change the naming of the output file
- `-m` to activate using MEMs

```bash
python preprocessing.py -o DIR -v FILE -e INT -m -n STR
```
