import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from keras.utils import to_categorical

import matplotlib.pyplot as plt

class DataFrame(object):
    def __init__(self, path_to_input_files, 
                classes, event_category, 
                train_variables, prenet_targets,
                test_percentage = 0.1,
                norm_variables = False):

        ''' takes a path to a folder where one h5 per class is located
            the events are cut according to the event_category
            variables in train_variables are used as input variables
            variables in prenet_targets are used as classes for the pre net
            the dataset is shuffled and split into a test and train sample
                according to test_percentage
            for better training, the variables can be normed to std(1) and mu(0) '''

        # loop over all classes and extract data as well as event weights
        class_dataframes = list()
        for cls in classes:
            class_file = path_to_input_files + "/" + cls + ".h5"
            print("-"*50)
            print("loading class file "+str(class_file))

            with pd.HDFStore( class_file, mode = "r" ) as store:
                cls_df = store.select("data")
                print("number of events before selections: "+str(cls_df.shape[0]))

            # apply event category cut
            cls_df.query(event_category, inplace = True)
            self.event_category = event_category
            print("number of events after selections:  "+str(cls_df.shape[0]))

            # add event weight
            cls_df = cls_df.assign(total_weight = lambda x: x.Weight_XS * x.Weight_CSV * x.Weight_GEN_nom)
            
            weight_sum = sum(cls_df["total_weight"].values)
            class_weight_scale = 1.
            if "ttH" in cls: class_weight_scale *= 1.0
            cls_df = cls_df.assign(train_weight = lambda x: class_weight_scale*x.total_weight/weight_sum)
            print("weight sum of train_weight: "+str( sum(cls_df["train_weight"].values) ))

            # add data to list of dataframes
            class_dataframes.append( cls_df )
            print("-"*50)

        # concatenating all dataframes
        df = pd.concat( class_dataframes )
        del class_dataframes

        # add class_label translation
        index = 0
        self.class_translation = {}
        for cls in classes:
            self.class_translation[cls] = index
            index += 1
        self.classes = classes
        self.index_classes = [self.class_translation[c] for c in classes]

        df["index_label"] = pd.Series( [self.class_translation[c] for c in df["class_label"].values], index = df.index )

        # norm weights to mean(1)
        df["train_weight"] = df["train_weight"]*df.shape[0]/len(classes)

        # save some meta data about net
        self.n_input_neurons = len(train_variables)
        self.n_prenet_output_neurons = len(prenet_targets)
        self.n_output_neurons = len(classes)

        # shuffle dataframe
        df = shuffle(df)

        # norm variables if wanted
        unnormed_df = df
        if norm_variables:
            df[train_variables] = (df[train_variables] - df[train_variables].mean())/df[train_variables].std()

        # split test sample
        n_test_samples = int( df.shape[0]*test_percentage )
        self.df_test = df.head(n_test_samples)
        self.df_train = df.tail(df.shape[0] - n_test_samples )
        self.df_test_unnormed = unnormed_df.head(n_test_samples)
        del df

        # save variable lists
        self.train_variables = train_variables
        self.prenet_targets = prenet_targets
        self.output_classes = classes

    # train data -----------------------------------
    def get_train_data(self, as_matrix = True):
        if as_matrix: return self.df_train[ self.train_variables ].values
        else:         return self.df_train[ self.train_variables ]

    def get_train_weights(self):
        return self.df_train["train_weight"].values

    def get_train_labels(self, as_categorical = True):
        if as_categorical: return to_categorical( self.df_train["index_label"].values )      
        else:              return self.df_train["index_label"].values

    def get_prenet_train_labels(self):
        return self.df_train[ self.prenet_targets ].values
        
    # test data ------------------------------------
    def get_test_data(self, as_matrix = True, normed = True):
        if not normed: return self.df_test_unnormed[ self.train_variables ]
        if as_matrix:  return self.df_test[ self.train_variables ].values
        else:          return self.df_test[ self.train_variables ]

    def get_test_weights(self):
        return self.df_test["total_weight"].values

    def get_test_labels(self, as_categorical = True):
        if as_categorical: return to_categorical( self.df_test["index_label"].values )
        else:              return self.df_test["index_label"].values

    def get_prenet_test_labels(self, as_matrix = True):
        return self.df_test[ self.prenet_targets ].values




    def hist_train_variables(self, signal_hists = [], n_bins = 20, logscale = True):

        # loop over variables
        for var in self.train_variables:
            plt.figure()
            
            # figure out binning
            max_val = max(self.df_train[var].values)
            min_val = min(self.df_train[var].values)
            bin_range = [min_val, max_val]

            # stack backgrounds
            bkg_values = []
            bkg_labels = []
            bkg_weights = []
            for cls in self.output_classes:
                if  cls in signal_hists: continue
                condition = "(class_label == '"+str(cls)+"')"
                bkg_values.append( self.df_train.query(condition)[var].values )
                bkg_weights.append( self.df_train.query(condition)["train_weight"].values )
                bkg_labels.append(cls)

            # plot backgrounds
            plt.hist( bkg_values, weights = bkg_weights,
                bins = n_bins, range = bin_range, 
                label = bkg_labels, stacked = True, histtype = "stepfilled",
                log = logscale, normed = True)

            n_bkgs = len(bkg_labels)

            # loop over signals
            for cls in signal_hists:
                condition = "(class_label == '"+str(cls)+"')"
                sig_values = self.df_train.query(condition)[var].values
                sig_weights = self.df_train.query(condition)["train_weight"].values

                # plot signal
                plt.hist(sig_values, weights = sig_weights,
                    bins = n_bins, range = bin_range,
                    label = cls, histtype = "step", 
                    color = "black", lw = 2, log = logscale, normed = True)

            # annotations
            plt.legend()
            plt.grid()
            plt.xlabel(var)
            plt.title(self.event_category)
            # plt.savefig(str(var)+".pdf")
            
        # plot all
        plt.show()



