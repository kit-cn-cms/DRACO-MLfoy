# pyrootsOfTheCaribbean
Script to plot the input variables used for NN training.
## Adjust settings
- add/change signal samples to plot
```python
plotter.addSample(
    sampleName      = "NAME",
    sampleFile      = data_dir+"FILENAME",
    plotColor       = ROOT.COLOR,
    signalSample    = True)
```
- add/change background samples to plot
```python
plotter.addSample(
    sampleName      = "NAME",
    sampleFile      = data_dir+"FILENAME",
    plotColor       = ROOT.COLOR)
```
- add/change categories to plot (example category name is `4j_ge3t`)
```python
plotter.addCategory("(ge)[nJets]j_(ge)[nTags]t")
```
- add/change variable plotting in [plot_configs/variableConfig.csv](https://github.com/kit-cn-cms/DRACO-MLfoy/blob/dev_ReleaseVersion/pyrootsOfTheCaribbean/plot_configs/variableConfig.csv)
	- `variablename` name of the variable used in ntuples
	- `minvalue` bin range: minimal value of the plotted histogram
	- `maxvalue` bin range: maximal value of the plotted histogram
	- `numberofbins` number of bins plotted
	- `logoption` logarithmic y-axis range ` to activate, `-` to deactivate
	- `displayname` displayed name of the variable on the plot

## Usage
to execute with default options use
```bash
python plotInputVariables.py
```
or use the following options
1. File Options
	- `-o DIR` to change the name of the ouput directory, can be either a string or absolute path
	(default is `plots_InputFeatures`)
	- `-i DIR` to change the name of the input directory, can be either a string or absolute path 
	(default is `InputFeatures`)
	- `-v FILE` to change the variable Selection to plot, if the file is in `/variable_sets/` the name is sufficient, else the absolute path is needed 
	(default is `example_variables`)
2. Plot Options
	- `-l` to create logarithmic plots
	- `-r` to deactivate additional ratio plots 
	- `--ratiotitle=STR` change the Title of the ratio plot (default is `"#frac{signal}{background}"`)
	- `-k` to deactivate KS score 
3. Scaling Options
	- `--scalesignal` to scale signal (default is `1`), possible options:
		- `1` to scale Signal to background Integral
		- `FLOAT` to scale Signal with float value (for example `--scalesignal=41`)
		- `False` to not scale Signal
	- `--lumiscale` to scale number of events according to luminosity (default is `1`)

4. Private Work Option
	`-p` to activate Private work Option, 
	- will add "private work"- label to plots
	- ATTENTION: deactivates scaling options, therefore changes y-axis title to "normalized to unit area" instead of "Events expected"

Example:
```bash
python plotInputVariables.py -i /path/to/input/files -o testPlots -v test_set  --scalesignal=False --lumiscale=41 --ratiotitle=#frac{ttH}{ttbar}
```

