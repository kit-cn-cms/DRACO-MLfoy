import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from keras.utils import to_categorical

import matplotlib.pyplot as plt

class DataFrame(object):
    def __init__(self, path_to_input_files,
                classes, event_category,
                train_variables,
                test_percentage = 0.1,
                norm_variables = False,
                additional_cut = None,
                lumi = 41.5):

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

            # add lumi weight
            cls_df = cls_df.assign(lumi_weight = lambda x: x.Weight_XS * x.Weight_GEN_nom * lumi)

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

        df["is_ttH"] = pd.Series( [1 if c=="ttHbb" else 0 for c in df["class_label"].values], index = df.index )
        df["index_label"] = pd.Series( [self.class_translation[c] for c in df["class_label"].values], index = df.index )

        # norm weights to mean(1)
        df["train_weight"] = df["train_weight"]*df.shape[0]/len(classes)

        # save some meta data about net
        self.n_input_neurons = len(train_variables)
        self.n_output_neurons = len(classes)

        # shuffle dataframe
        df = shuffle(df, random_state = 333)

        # norm variables if wanted
        unnormed_df = df.copy()
        if norm_variables:
            norm_csv = pd.DataFrame(index=train_variables, columns=["mu", "std"])
            for v in train_variables:
                norm_csv["mu"][v] = unnormed_df[v].mean()
                norm_csv["std"][v] = unnormed_df[v].std()
            df[train_variables] = (df[train_variables] - df[train_variables].mean())/df[train_variables].std()
            self.norm_csv = norm_csv

        if additional_cut:
            print("events in dataframe before cut "+str(df.shape[0]))
            df.query( additional_cut, inplace = True )
            print("events in dataframe after cut "+str(df.shape[0]))

        self.unsplit_df = df.copy()
        # split test sample
        n_test_samples = int( df.shape[0]*test_percentage )
        self.df_test = df.head(n_test_samples)
        self.df_train = df.tail(df.shape[0] - n_test_samples )
        self.df_test_unnormed = unnormed_df.head(n_test_samples)

        # print some counts
        print("total events after cuts:  "+str(df.shape[0]))
        print("events used for training: "+str(self.df_train.shape[0]))
        print("events used for testing:  "+str(self.df_test.shape[0]))
        del df

        # save variable lists
        self.train_variables = train_variables
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


    # test data ------------------------------------
    def get_test_data(self, as_matrix = True, normed = True):
        if not normed: return self.df_test_unnormed[ self.train_variables ]
        if as_matrix:  return self.df_test[ self.train_variables ].values
        else:          return self.df_test[ self.train_variables ]

    def get_test_weights(self):
        return self.df_test["total_weight"].values
    def get_lumi_weights(self):
        return self.df_test["lumi_weight"].values

    def get_test_labels(self, as_categorical = True):
        if as_categorical: return to_categorical( self.df_test["index_label"].values )
        else:              return self.df_test["index_label"].values

    def get_ttH_flag(self):
        return self.df_test["is_ttH"].values

    # full sample ----------------------------------
    def get_full_df(self):
        return self.unsplit_df[self.train_variables]
