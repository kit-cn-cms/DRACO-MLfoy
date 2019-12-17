# Train NNs
## Before Usage
Manually set TensorFlow backend for KERAS, add this for example to your `~/.zshrc`
```bash
export KERAS_BACKEND=tensorflow
```

## Adjust settings
- add/change samples used in training
```python
input_samples.addSample("SAMPLENAME"+naming, label = "SAMPLENAME", normalization_weight = FLOAT)
```
for example
```python
input_samples.addSample(options.getDefaultName("ttH"), label = "ttH", normalization_weight = options.getNomWeight())
input_samples.addSample(options.getDefaultName("ttbb") , label = "ttbb")
```

To change the network architecture adjust an architecture in the `net_configs.py` or create a new one. To use this config during training specify the `--netconfig=NAMEOFCONFIG` option.


## Usage
To execute with default options use
```bash
python train_template.py
```
or use the following options
1. Category used for training
    - `-c STR` name of the category `(ge/le)[nJets]j_(ge/le)[nTags]t`
    (default is `4j_ge3t`)

2. Sample Options
    - `-o DIR` to change the name of the output directory (absolute path or path relative to `workdir`)
        (default is `test_training`)
    - `-i DIR` to change the name of the input directory where the preprocessed h5 files are stored. Which files in this directory are used for training has to be adjusted in the script itself
        (default is `InputFeatures`)
    - `--naming=STR` to adjust the naming of the input files.
        (default is `_dnn.h5`)
    - `--even` to select only events with `Evt_Odd==0`
    - `--odd` to select only events with `Evt_Odd==1`

3. Training Options
    - `-v FILE` to change the variable selection (absolute path to the variable set file or path relative to `variable_sets` directory)
        (default is `example_variables`)
    - `-e INT` change number of training epochs
        (default is `1000`)
    - `-n STR` STR of the config name in`net_configs.py` file to adjust the architecture of the neural network
    - `--balanceSamples` activates an additional balancing of train samples. With this options the samples which have fewer events are used multiple times in one pass over the training set (epoch). As a default options the sample weights are balanced, such that the sum of train weights is equal for all used samples.
    - `-a` comma separated list of samples to be activated for training. If this option is not used all samples are used as a default.
    - `-u` to NOT perform a normalization of input features to mean zero and std deviation one.

4. Plotting Options
    - `-p` to create plots of the output
    - `-L` to create plots with logarithmic y-axis
    - `-R` to print ROC value for confusion matrix
    - `-P` to create private work label
    - `-S STR` to change the plotted signal class (not part of the background stack plot), possible to do a combination, for example `ttH,ttbb`
        (default is `None`)
    - `-s FLOAT` to scale the signal histograms in the output plots (default is -1 and scales to background integral)

5. Binary Training Options
    - `--binary` activate binary training by defining one signal and one background class. Which samples are set as signal is defined by the `signal` option
    - `-t` target value for background samples during training.
        (default is 0, default for signal 1 and cannot be changed)
    - `--signal=STR` signal class for binary classification, possible to do a combination, for example `ttH,ttbb` (default is `None`)

6. Adversary Training Options
    - `--adversary` activate adversary training with an additional network competing with the classifier. Which samples are set as nominal is defined by `naming` and additional samples by `addsamplenaming`.
    - `--penalty=FLOAT` change the penalty parameter in the adversary loss function (default ist `10`)
    - `--addsamplenaming=STR` to adjust the naming of the input files of additional samples from other generators (default is `_dnn_OL.h5`)


Example:
```bash
python train_template.py -i /path/to/input/files/ -o testRun --netconfig=test_config --plot --printroc -c ge6j_ge3t --epochs=1000 --signalclass=ttHbb,ttbb
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


Compute the average of the weights for different layers and different seeds:

Run train_template.py calling the function get_weights() (not only get_input_weights()) in order to have the "absolute_weight_sum_layer *.csv" for the input layer and each dropout layer.
If you want to compute the average over multiple trainings with different seed, run train_template.py several times and copy all the folders inside one folder (/path/to/trained/networks/)

```bash
train_scripts/average_weights.py -i /path/to/trained/networks/
```
using the option `-i DIR` to specify the path to the directory inside which there are the folders for the already trained networks with different seeds.

using the option `-l INT` to specify the number of layers to consider in the computation.


To compute the first-order derivatives for the DNN Taylor expansion add the funciton "dnn.get_gradients(options.isBinary())" in `train_template.py`. Whenever you add/change an architecture in `net_configs.py`, remember to  change the corresponding TensorFlow architecture in `net_configs_tensorflow.py`. The TensorFlow architecture has to have the same name as the keras one with the additional "_tensorflow" at the end.


## Interface to pyroot plotscripts
The DNNs which are trained with this framework can be evaluated with the `pyroot-plotscripts` framework.
For this purpose, generate a directory (e.g `DNNSet`) containing subdirectories (e.g. `4j_ge3t_dnn`, etc.) for each separately trained DNN.
Copy the content of the `checkpoints` directory created after the DNN training to these subdirectories.
The directory of sets of DNNs can be set in the plotLimits script at the `checkpointFiles` option.
To generate a config for plotting the DNN discriminators and the input features, execute `python util/dNNInterfaces/MLfoyInterface.py -c /PATH/TO/DNNSet/` which generates a file `autogenerated_plotconfig.py`. Move this file to the `configs` directory and rename it if wanted.
Specify this filename in the plotLimits script as `plot_cfg`.
