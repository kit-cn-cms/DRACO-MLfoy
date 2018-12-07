import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# file path
file_path = "/storage/c/vanderlinden/DRACO-MLfoy/workdir/train_samples"
test_sample = file_path + "/base_train_set_test.h5"

variable = "nJets"


df = pd.read_hdf(test_sample, key = "data")

max_jets = max(df[variable].values)
min_jets = min(df[variable].values)
n_bins = max_jets - min_jets + 1
bin_range = [min_jets -0.5, max_jets +0.5]


df["nJets"].hist(bins = n_bins, range = bin_range)
plt_axis = plt.gca()
plt_axis.set_xticks(np.arange(min_jets,max_jets+1,1))
plt.xlabel(variable)
plt.show()
