import pandas as pd
import os
import sys
import numpy as np
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.decomposition import PCA

class Sample:
    def __init__(self, path, label, normalization_weight = 1., train_weight = 1., test_percentage = 0.2, total_weight_expr='x.Weight_XS * x.Weight_CSV * x.Weight_GEN_nom', addSampleSuffix = ""):
        self.path = path
        self.label = label
        self.normalization_weight = normalization_weight
        self.isSignal = None
        self.train_weight = train_weight
        self.test_percentage = test_percentage
        self.min=0.0
        self.max=1.0
        self.total_weight_expr = total_weight_expr
        self.addSampleSuffix = addSampleSuffix

    def load_dataframe(self, event_category, lumi, evenSel = ""):
        print("-"*50)
        print("loading sample file "+str(self.path))
        with pd.HDFStore( self.path, mode = "r" ) as store:
            df = store.select("data")
            print("number of events before selections: "+str(df.shape[0]))

        # apply event category cut
        query = event_category

        if not evenSel == "":
            query+=" and "+evenSel
        df.query(query, inplace = True)
        print("number of events after selections:  "+str(df.shape[0]))
        self.nevents = df.shape[0]

        # add event weight
        df = df.assign(total_weight = lambda x: eval(self.total_weight_expr))

        # assign train weight
        weight_sum = sum(df["total_weight"].values)
        df = df.assign(train_weight = lambda x: x.total_weight/weight_sum*self.train_weight)
        print("sum of train weights: {}".format(sum(df["train_weight"].values)))

        # add lumi weight
        # adjust weights via 1/test_percentage such that yields in plots correspond to complete dataset

        df = df.assign(lumi_weight = lambda x: x.total_weight * lumi * self.normalization_weight / self.test_percentage)

        if self.addSampleSuffix in self.label:
            df["class_label"] = pd.Series([ c + self.addSampleSuffix for c in df["class_label"].values], index = df.index)

        self.data = df
        print("-"*50)

    def getConfig(self):
        config = {}
        config["sampleLabel"] = self.label
        config["samplePath"] = self.path
        config["sampleWeight"] = self.normalization_weight
        config["sampleEvents"] = self.nevents
        config["min"] = self.min
        config["max"] = self.max
        return config

    def addPrediction(self, model, train_variables):
        self.prediction_vector = model.predict(
            self.data[train_variables].values)

        print("total number of events in sample: "+str(self.data.shape[0]))
        self.predicted_classes = np.argmax( self.prediction_vector, axis = 1 )

        self.lumi_weights = self.data["lumi_weight"].values

class InputSamples:
    def __init__(self, input_path, activateSamples = None, test_percentage = 0.2, addSampleSuffix = ""):
        self.binary_classification = False
        self.input_path = input_path
        self.samples = []
        self.activate_samples = activateSamples
        self.addSampleSuffix = addSampleSuffix
        if self.activate_samples:
            self.activate_samples = self.activate_samples.split(",")
        self.test_percentage = float(test_percentage)
        if self.test_percentage <= 0. or self.test_percentage >= 1.:
            sys.exit("fraction of events to be used for testing (test_percentage) set to {}. this is not valid. choose something in range (0.,1.)")

        self.additional_samples = 0

    def addSample(self, sample_path, label, normalization_weight=1., train_weight=1., total_weight_expr='x.Weight_XS * x.Weight_CSV * x.Weight_GEN_nom'):
        if self.activate_samples and not label in self.activate_samples:
            print("skipping sample {}".format(label))
            return
        if not os.path.isabs(sample_path):
            sample_path = self.input_path + "/" + sample_path
        self.samples.append(Sample(sample_path, label, normalization_weight, train_weight, self.test_percentage, total_weight_expr=total_weight_expr, addSampleSuffix = self.addSampleSuffix))

        if bool(self.addSampleSuffix) and label.endswith(self.addSampleSuffix):
            self.additional_samples +=1

    def getClassConfig(self):
        configs = []
        for sample in self.samples:
            configs.append( sample.getConfig() )
        return configs

    def addBinaryLabel(self, signals, bkg_target):
        self.binary_classification = True
        self.signal_classes = signals
        self.bkg_target = float(bkg_target)
        for sample in self.samples:
            if sample.label in signals:
                sample.isSignal = True
            else:
                sample.isSignal = False


class DataFrame(object):
    ''' takes a path to a folder where one h5 per class is located
        the events are cut according to the event_category
        variables in train_variables are used as input variables
        the dataset is shuffled and split into a test and train sample according to test_percentage
        for better training, the variables can be normed to std(1) and mu(0) '''

    def __init__(self,
                input_samples,
                event_category,
                train_variables,
                category_cutString = None,
                category_label     = None,
                norm_variables = True,
                test_percentage = 0.2,
                lumi = 41.5,
                shuffleSeed = None,
                balanceSamples = True,
                evenSel = "",
                addSampleSuffix = ""):

        self.event_category = event_category
        self.lumi = lumi
        self.evenSel = evenSel

        self.shuffleSeed = shuffleSeed
        self.balanceSamples = balanceSamples
        self.addSampleSuffix = addSampleSuffix

        self.binary_classification = input_samples.binary_classification
        if self.binary_classification: self.bkg_target = input_samples.bkg_target

        # loop over all input samples and load dataframe
        train_samples = []
        for sample in input_samples.samples:
            sample.load_dataframe(self.event_category, self.lumi, self.evenSel)
            train_samples.append(sample.data)

        # concatenating all dataframes
        df = pd.concat(train_samples, sort=True)
        del train_samples

        # multiclassification labelling
        if not self.binary_classification:
            # add class_label translation
            index = 0
            self.class_translation = {}
            self.classes = []

            for sample in input_samples.samples:
                self.class_translation[sample.label] = index
                self.classes.append(sample.label)
                index += 1
            self.index_classes = [self.class_translation[c] for c in self.classes]

            # add flag for ttH and ttbb to dataframe
            df["is_ttH"] = pd.Series( [1 if (c=="ttHbb" or c=="ttH") else 0 for c in df["class_label"].values], index = df.index )
            df["is_ttBB"] = pd.Series( [1 if ("ttbb" in c) else 0 for c in df["class_label"].values], index = df.index )

            # add generator flag for adversary training
            if input_samples.additional_samples:
                df["generator_flag"] = pd.Series( [1 if (self.addSampleSuffix in c) else 0 for c in df["class_label"].values], index = df.index )

                df["index_label"] = pd.Series( [self.class_translation[c[:-len(self.addSampleSuffix)]] if (c.endswith(self.addSampleSuffix)) 
                    else self.class_translation[c.replace("ttHbb", "ttH").replace("ttZbb", "ttZ")] for c in df["class_label"].values], index = df.index )
            else:
                df["index_label"] = pd.Series( [self.class_translation[c.replace("ttHbb", "ttH").replace("ttZbb","ttZ")] for c in df["class_label"].values], index = df.index )   
            
            # norm weights to mean(1) 
            # TODO: adjust train_weights for adversary training (processes with different samples from different generators)
            df["train_weight"] = df["train_weight"]*df.shape[0]/len(self.classes)

            # save some meta data about network
            self.n_input_neurons = len(train_variables)
            self.n_output_neurons = len(self.classes)-input_samples.additional_samples

        # binary classification labelling
        else:

            # class translations
            self.class_translation = {}
            self.class_translation["sig"] = 1
            self.class_translation["bkg"] = float(self.bkg_target)

            self.classes = ["sig", "bkg"]
            self.index_classes = [self.class_translation[c] for c in self.classes]

            df["index_label"] = pd.Series( [1 if c.replace("ttHbb","ttH").replace("ttZbb","ttZ") in input_samples.signal_classes else 0 for c in df["class_label"].values], index = df.index)

            # add_bkg_df = None
            if not input_samples.additional_samples:
                bkg_df = df.query("index_label == 0")
            else:
                df["is_ttBB"] = pd.Series( [1 if ("ttbb" in c) else 0 for c in df["class_label"].values], index = df.index )
                df["generator_flag"] = pd.Series( [1 if (self.addSampleSuffix in c) else 0 for c in df["class_label"].values], index = df.index )

                bkg_df = df.query("index_label == 0 and generator_flag == 0")
                add_bkg_df = df.query("index_label == 0 and generator_flag == 1")
                add_bkg_weight = sum(add_bkg_df["train_weight"].values)
                add_bkg_df["train_weight"] = add_bkg_df["train_weight"]/(2*add_bkg_weight)*df.shape[0]
                add_bkg_df["binaryTarget"] = float(self.bkg_target)
            
            sig_df = df.query("index_label == 1")

            sig_weight = sum(sig_df["train_weight"].values)
            bkg_weight = sum(bkg_df["train_weight"].values)

            signal_weight = sum( sig_df["train_weight"].values )
            bkg_weight = sum( bkg_df["train_weight"].values )
            sig_df["train_weight"] = sig_df["train_weight"]/(2*signal_weight)*df.shape[0]
            bkg_df["train_weight"] = bkg_df["train_weight"]/(2*bkg_weight)*df.shape[0]
            #sig_df["class_label"] = "sig"
            #bkg_df["class_label"] = "bkg"
            sig_df["binaryTarget"] = 1.
            bkg_df["binaryTarget"] = float(self.bkg_target)

            if input_samples.additional_samples == 0:
                df = pd.concat([sig_df, bkg_df])
                print("True")
            else:
                print("False")
                df = pd.concat([sig_df, bkg_df, add_bkg_df])

            self.n_input_neurons = len(train_variables)
            self.n_output_neurons = 1

        # shuffle dataframe
        if not self.shuffleSeed:
           self.shuffleSeed = np.random.randint(low = 0, high = 2**16)

        print("using shuffle seed {} to shuffle input data".format(self.shuffleSeed))

        df = shuffle(df, random_state = self.shuffleSeed)

        # norm variables if activated
        unnormed_df = df.copy()
        norm_csv = pd.DataFrame(index=train_variables, columns=["mu", "std"])
        if norm_variables:
            for v in train_variables:
                norm_csv["mu"][v] = unnormed_df[v].mean()
                norm_csv["std"][v] = unnormed_df[v].std()
                if norm_csv["std"][v] == 0.:
                    sys.exit("std deviation of variable {} is zero -- this cannot be used for training".format(v))
        else:
            for v in train_variables:
                norm_csv["mu"][v] = 0.
                norm_csv["std"][v] = 1.
        df[train_variables] = (df[train_variables] - df[train_variables].mean())/df[train_variables].std()
        self.norm_csv = norm_csv

        self.unsplit_df = df.copy()

        # split test sample
        n_test_samples = int( df.shape[0]*test_percentage)
        self.df_test = df.head(n_test_samples)
        self.df_train = df.tail(df.shape[0] - n_test_samples)
        self.df_test_unnormed = unnormed_df.head(n_test_samples)

        # split ttbb test samples into nominal and add sample according to ttbb generator
        if addSampleSuffix:
            self.df_test_nominal = self.df_test[ self.df_test.class_label != "ttbb"+self.addSampleSuffix ]
            self.df_test_additional = self.df_test[ self.df_test.class_label != "ttbb" ]

        # save variable lists
        self.train_variables = train_variables
        self.output_classes = self.classes
        self.input_samples = input_samples

        # sample balancing if activated
        if self.balanceSamples:
           self.balanceTrainSample()

        # print some counts
        print("total events after cuts:  "+str(df.shape[0]))
        print("events used for training: "+str(self.df_train.shape[0]))
        print("events used for testing:  "+str(self.df_test.shape[0]))
        del df

    #     # init non trainable samples
    #     self.non_train_samples = None

    # def get_non_train_samples(self):
    #     # get samples with flag 'isTrainSample == False' and load those
    #     samples = []
    #     for sample in self.input_samples.samples:
    #         if sample.isTrainSample: continue
    #         sample.data[self.train_variables] = (sample.data[self.train_variables] - self.norm_csv["mu"])/(self.norm_csv["std"])
    #         samples.append(sample)
    #     self.non_train_samples = samples



    def balanceTrainSample(self):
        # get max number of events per sample
        maxEvents = 0
        for sample in self.input_samples.samples:
            if maxEvents < sample.nevents:
                maxEvents = sample.nevents

        new_train_dfs = []

        print("balancing train sample ...")

        # multiply train events
        for sample in self.input_samples.samples:
            print("+"*30)

            # get events
            class_label = sample.label

            if self.binary_classification: class_label = '1' if class_label in self.input_samples.signal_classes else '0'

            events = self.df_train.query("(index_label == '{}')".format(class_label))

            # get multiplication factor
            factor = int(maxEvents/sample.nevents)

            print("multiplying {} Events by factor {}".format(sample.label, factor))
            print("number of events before: {}".format(events.shape[0]))
            print("number of events after:  {}".format(events.shape[0]*factor))
            events["train_weight"] = events["train_weight"]/factor
            print("sum of train weights: {}".format(sum(events["train_weight"].values)*factor))
            for _ in range(factor):
                new_train_dfs.append(events)

        self.df_train = pd.concat(new_train_dfs)
        self.df_train = shuffle(self.df_train)

    # train data -----------------------------------
    def get_train_data(self, as_matrix = True):
        if as_matrix: return self.df_train[ self.train_variables ].values
        else:         return self.df_train[ self.train_variables ]

    def get_train_weights(self):
        return self.df_train["train_weight"].values

    def get_train_labels(self, as_categorical = True):
        if self.binary_classification: return self.df_train["binaryTarget"].values
        if as_categorical: return to_categorical( self.df_train["index_label"].values )
        else:              return self.df_train["index_label"].values

    def get_train_lumi_weights(self):
        return self.df_train["lumi_weight"].values

    # test data ------------------------------------
    def get_test_data(self, as_matrix=True, normed=True):
        if not normed: return self.df_test_unnormed[ self.train_variables ]
        if as_matrix:  return self.df_test[ self.train_variables ].values
        else:          return self.df_test[ self.train_variables ]

    def get_test_weights(self):
        return self.df_test["total_weight"].values

    def get_lumi_weights(self):
        return self.df_test["lumi_weight"].values

    def get_test_labels(self, as_categorical = True):
        if self.binary_classification: return self.df_test["binaryTarget"].values
        if as_categorical: return to_categorical( self.df_test["index_label"].values )
        else:              return self.df_test["index_label"].values

    def get_class_flag(self, class_label):
        return pd.Series( [1 if c.replace("ttHbb","ttH").replace("ttZbb","ttZ")==class_label else 0 for c in self.df_test["class_label"].values], index = self.df_test.index ).values

    def get_ttH_flag(self):
        return self.df_test["is_ttH"].values

    # full sample ----------------------------------
    def get_full_df(self):
        return self.unsplit_df[self.train_variables]

    # adversary test data --------------------------
    def get_adversary_labels(self):
        return self.df_train["generator_flag"].values

    def get_adversary_weights_classifier(self):
        return pd.Series([0 if self.df_train["generator_flag"].values[i]==1 else c for i, c in enumerate(self.df_train["train_weight"].values)],index = self.df_train.index).values

    def get_adversary_weights_adversary(self):
        return pd.Series([c if self.df_train["is_ttBB"].values[i]==1 else 0 for i, c in enumerate(self.df_train["train_weight"].values)],index = self.df_train.index).values

    def get_test_data_nominal(self, as_matrix=True, normed=True):
        if not normed: return self.df_test_unnormed[ self.train_variables ]
        if as_matrix:  return self.df_test_nominal[ self.train_variables ].values
        else:          return self.df_test_nominal[ self.train_variables ]

    def get_test_data_additional(self, as_matrix=True, normed=True):
        if not normed: return self.df_test_unnormed[ self.train_variables ]
        if as_matrix:  return self.df_test_additional[ self.train_variables ].values
        else:          return self.df_test_additional[ self.train_variables ]

    def get_test_weights_nominal(self):
        return self.df_test_nominal["total_weight"].values

    def get_test_weights_additional(self):
        return self.df_test_additional["total_weight"].values

    def get_lumi_weights_nominal(self):
        return self.df_test_nominal["lumi_weight"].values

    def get_lumi_weights_additional(self):
        return self.df_test_additional["lumi_weight"].values

    def get_test_labels_nominal(self, as_categorical = True):
        if self.binary_classification: return self.df_test_nominal["binaryTarget"].values
        if as_categorical: return to_categorical( self.df_test_nominal["index_label"].values )
        else:              return self.df_test_nominal["index_label"].values

    def get_test_labels_additional(self, as_categorical = True):
        if self.binary_classification: return self.df_test_additional["binaryTarget"].values
        if as_categorical: return to_categorical( self.df_test_additional["index_label"].values )
        else:              return self.df_test_additional["index_label"].values
