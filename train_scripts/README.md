# Train NNs
## Before Usage
Manually set TensorFlow backend for KERAS, add this for example to your `~/.zshrc`
```bash
export KERAS_BACKEND=tensorflow
```


## Usage
To execute with default options use
```bash
python train_template.py
```
or use the following options
1. Category used for training
	- `-c STR` name of the category `(ge)[nJets]j_(ge)[nTags]t`
	(default is `ge4j_ge4t`)

2. Naming/File Options
	- `-o DIR` to change the name of the ouput directory, can be either a string or absolute path
	(default is `test_training`)
	- `-i DIR` to change the name of the input directory, can be either a string or absolute path
	(default is `InputFeatures`)
	- `-n STR` to adjust the name of the input file generated in preprocessing
	(default is `dnn.h5`)
	- `-v FILE` to change the variable Selection, if the file is in `/variable_sets/` the name is sufficient, else the absolute path is needed
	(default is `DL_variables`)
	- `--signalclass=STR` to change name of the signal class
	(default is `None`)

3. Training Options
	- `-e INT` change number of training epochs
	(default is `1000`)
	- `-s INT` change number of epochs without decrease in validation loss before stopping
	(default is `20`)
	- `--netconfig=STR` STR of the config name in `net_config` dictonary in`net_configs.py` (config in this file will not be used anymore!)

4. Plotting Options
	- `-p` to create plots of the output
	- `-l` to create logarithmic plots
	- `--printroc` to print ROC value for confusion matrix
	- `--privatework` to create private work label


Example:
```bash
python train_template.py -i /path/to/input/files/ -o testRun --netconfig=test_config --plot --printroc -c ge6j_ge3t --epochs=1000
```


Re-evaluate DNN after training with
```bash
python eval_template.py
```
using the option `-i DIR` to specify the path to the already trained network.

The plotting options of `train_template.py` are also avaiable for this script.

Example:
```bash
python eval_template.py -i testRun_ge6j_ge3t -o eval_testRun --plot --printroc
```
