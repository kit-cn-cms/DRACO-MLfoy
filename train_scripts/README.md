# Train NNs
## Before Usage
```bash
export KERAS_BACKEND=tensorflow
```

## Adjust settings
- add/change samples used in training
```python
input_samples.addSample("SAMPLENAME"+naming, label = "SAMPLENAME", signalSample = BOOL, normalization_weight = FLOAT)
```
for example
```python
input_samples.addSample("ttHbb"+naming, label = "ttHbb", signalSample = True, normalization_weight = 2.)
input_samples.addSample("ttbb"+naming,  label = "ttbb")
```

- change network architecture `dpg_config` if optimizing, otherwise add network architecture to the file `net_configs.py` as dictonary entry in the `config_dict` to not lose it and execute it with the parser option --netconfig=CONFIGNAME
- only change DNN training class `dnn = DNN.DNN([...])` properties 
	- `eval_metrics` 
	- `test_percentage` percentage of samples used to test
  others are changed with parser options!

## Usage
To execute with default options us 
```bash
python train_template.py 
```
or use the following options 
- 