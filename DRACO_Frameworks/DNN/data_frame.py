import pandas as pd
import os
import numpy as np
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.decomposition import PCA

class Sample:
    def __init__(self, path, label, normalization_weight = 1.):
        self.path = path
        self.label = label
        self.normalization_weight = normalization_weight
        

    def load_dataframe(self, event_category, lumi):
        print("-"*50)
        print("loading sample file "+str(self.path))
        with pd.HDFStore( self.path, mode = "r" ) as store:
            df = store.select("data")
            print("number of events before selections: "+str(df.shape[0]))

        # apply event category cut
        df.query(event_category, inplace = True)
        print("number of events after selections:  "+str(df.shape[0]))

        # add event weight
        df = df.assign(total_weight = lambda x: x.Weight_XS * x.Weight_CSV * x.Weight_GEN_nom)

        # assign train weight
        weight_sum = sum(df["total_weight"].values)
        df = df.assign(train_weight = lambda x: x.total_weight/weight_sum)

        # add lumi weight
        df = df.assign(lumi_weight = lambda x: x.Weight_XS * x.Weight_GEN_nom * lumi * self.normalization_weight)

        self.data = df
        print("-"*50)
        
    def getConfig(self):
        config = {}
        config["sampleLabel"] = self.label
        config["samplePath"] = self.path
        config["sampleWeight"] = self.normalization_weight
        return config

    def addPrediction(self, model, train_variables):
        self.prediction_vector = model.predict(
            self.data[train_variables].values)
        
        print("total number of events in sample: "+str(self.data.shape[0]))
        self.predicted_classes = np.argmax( self.prediction_vector, axis = 1 )

        self.lumi_weights = self.data["lumi_weight"].values

class InputSamples:
    def __init__(self, input_path):
        self.input_path = input_path
        self.samples = []

    def addSample(self, sample_path, label, normalization_weight = 1.):
        if not os.path.isabs(sample_path):
            sample_path = self.input_path + "/" + sample_path
        self.samples.append( Sample(sample_path, label, normalization_weight) )
        
    def getClassConfig(self):
        configs = []
        for sample in self.samples: 
            configs.append( sample.getConfig() )
        return configs


class DataFrame(object):
    ''' takes a path to a folder where one h5 per class is located
        the events are cut according to the event_category
        variables in train_variables are used as input variables
        the dataset is shuffled and split into a test and train sample
            according to test_percentage
        for better training, the variables can be normed to std(1) and mu(0) '''

    def __init__(self,
                input_samples,
                event_category,
                train_variables,
                norm_variables = True,
                test_percentage = 0.1,
                lumi = 41.5,
                shuffleSeed = None):

        self.event_category = event_category
        self.lumi = lumi

        self.shuffleSeed = shuffleSeed

        # loop over all input samples and load dataframe
        train_samples = []
        for sample in input_samples.samples:
            sample.load_dataframe(self.event_category, self.lumi)
            train_samples.append(sample.data)
        
        # concatenating all dataframes
        df = pd.concat( train_samples )
        del train_samples

        # add class_label translation
        index = 0
        self.class_translation = {}
        self.classes = []
        for sample in input_samples.samples:
            self.class_translation[sample.label] = index
            self.classes.append(sample.label)
            index += 1
        self.index_classes = [self.class_translation[c] for c in self.classes]

        # add flag for ttH to dataframe
        df["is_ttH"] = pd.Series( [1 if (c=="ttHbb" or c=="ttH") else 0 for c in df["class_label"].values], index = df.index )

        # add index labelling to dataframe
        df["index_label"] = pd.Series( [self.class_translation[c] for c in df["class_label"].values], index = df.index )

        # norm weights to mean(1)
        df["train_weight"] = df["train_weight"]*df.shape[0]/len(self.classes)

        # save some meta data about network
        self.n_input_neurons = len(train_variables)
        self.n_output_neurons = len(self.classes)

        # shuffle dataframe
        if not self.shuffleSeed:
            self.shuffleSeed = np.random.randint(low = 0, high = 2**16)
        df = shuffle(df, random_state = self.shuffleSeed)

        # norm variables if activated
        unnormed_df = df.copy()
        if norm_variables:
            norm_csv = pd.DataFrame(index=train_variables, columns=["mu", "std"])
            for v in train_variables:
                norm_csv["mu"][v] = unnormed_df[v].mean()
                norm_csv["std"][v] = unnormed_df[v].std()
            df[train_variables] = (df[train_variables] - df[train_variables].mean())/df[train_variables].std()
            self.norm_csv = norm_csv

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
        self.output_classes = self.classes
        self.input_samples = input_samples


    # train data -----------------------------------
    def get_train_data(self, as_matrix = True):
        if as_matrix: return self.df_train[ self.train_variables ].values
        else:         return self.df_train[ self.train_variables ]

    def get_train_weights(self):
        return self.df_train["train_weight"].values

    def get_train_labels(self, as_categorical = True):
        if as_categorical: return to_categorical( self.df_train["index_label"].values )
        else:              return self.df_train["index_label"].values

    def get_train_lumi_weights(self):
        return self.df_train["lumi_weight"].values        

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

    def get_class_flag(self, class_label):
        return pd.Series( [1 if c==class_label else 0 for c in self.df_test["class_label"].values], index = self.df_test.index ).values

    def get_ttH_flag(self):
        return self.df_test["is_ttH"].values

    # full sample ----------------------------------
    def get_full_df(self):
        return self.unsplit_df[self.train_variables]
