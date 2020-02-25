import pandas as pd
import os
import sys
import numpy as np
import tqdm
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.decomposition import PCA

class Sample:
    def __init__(self, path, label, normalization_weight = 1., train_weight = 1., test_percentage = 0.2, total_weight_expr='x.Weight_XS * x.Weight_CSV * x.Weight_GEN_nom'):
        self.path = path
        self.label = label
        self.normalization_weight = normalization_weight
        self.isSignal = None
        self.train_weight = train_weight
        self.test_percentage = test_percentage
        self.min=0.0
        self.max=1.0
        self.total_weight_expr = total_weight_expr

    def load_dataframe(self, event_category, lumi, evenSel = "", sys_variation=False):
        if not sys_variation:
            print("-"*50)
            print("loading sample file "+str(self.path))
            with pd.HDFStore( self.path, mode = "r" ) as store:
                df = store.select("data")
                samp = int(df.shape[0]*1.0)
                df = df.head(samp)
                print("number of events before selections: "+str(df.shape[0]))
        else:
            df = self.get_mixed_df()
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

        self.data = df
        print("-"*50)

    def get_mixed_df(self):
        abs_path = self.path.split("2017")[0]
        path_nominal = self.path.split("/")[-2]
        path_h5 = self.path.split("/")[-1]
        print("-"*50)
        print("loading sample file "+str(path_nominal))
        with pd.HDFStore( abs_path+path_nominal+"/"+path_h5, mode = "r" ) as store:
            df_nominal = store.select("data")
            print("number of nominal events before mixing: "+str(df_nominal.shape[0]))

        path_up = path_nominal.split("_gen")[0] + "_gen_JESup"
        print("-"*50)
        print("loading sample file "+str(path_up))
        with pd.HDFStore( abs_path+path_up+"/"+path_h5, mode = "r" ) as store:
            df_up = store.select("data")
            print("number of JES up events before mixing: "+str(df_up.shape[0]))

        path_down = path_nominal.split("_gen")[0] + "_gen_JESdown"
        print("-"*50)
        print("loading sample file "+str(path_down))
        with pd.HDFStore( abs_path+path_down+"/"+path_h5, mode = "r" ) as store:
            df_down = store.select("data")
            print("number of JES down events before mixing: "+str(df_down.shape[0]))

        df_mix = pd.DataFrame(columns=df_nominal.columns)
        evt_ids_nominal, evt_ids_up, evt_ids_down = [], [], []
        print "----- event mixing -----"
        #print df_nominal
        for i in tqdm.tqdm(range(df_down.shape[0])):
            evt_id = df_down.index[i]
            if evt_id in df_up.index:
                if evt_id in df_nominal.index:
                    if ((i+1) % 3 == 0):
                        evt_ids_nominal.append(evt_id)
                    elif ((i+1) % 3 == 1):
                        evt_ids_up.append(evt_id)
                    elif ((i+1) % 3 == 2):
                        evt_ids_down.append(evt_id)
        df_mix = df_mix.append(df_nominal.loc[evt_ids_nominal])
        df_mix = df_mix.append(df_up.loc[evt_ids_up])
        df_mix = df_mix.append(df_down.loc[evt_ids_down])
        #print df_mix.index
        return df_mix


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
    def __init__(self, input_path, activateSamples = None, test_percentage = 0.2):
        self.binary_classification = False
        self.input_path = input_path
        self.samples = []
        self.activate_samples = activateSamples
        if self.activate_samples:
            self.activate_samples = self.activate_samples.split(",")
        self.test_percentage = float(test_percentage)
        if self.test_percentage <= 0. or self.test_percentage >= 1.:
            sys.exit("fraction of events to be used for testing (test_percentage) set to {}. this is not valid. choose something in range (0.,1.)")

    def addSample(self, sample_path, label, normalization_weight=1., train_weight=1., total_weight_expr='x.Weight_XS * x.Weight_CSV * x.Weight_GEN_nom'):
        if self.activate_samples and not label in self.activate_samples:
            print("skipping sample {}".format(label))
            return
        if not os.path.isabs(sample_path):
            sample_path = self.input_path + "/" + sample_path

            # sample_path_nominal = self.input_path + "_nominal/" + sample_path
            # sample_path_up = self.input_path + "_JESup/" + sample_path
            # sample_path_down = self.input_path + "_JESdown/" + sample_path
        self.samples.append(Sample(sample_path, label, normalization_weight, train_weight, self.test_percentage, total_weight_expr=total_weight_expr))

        # self.samples.append(Sample(sample_path_nominal, label, normalization_weight, train_weight, self.test_percentage, total_weight_expr=total_weight_expr))
        # self.samples.append(Sample(sample_path_up, label, normalization_weight, train_weight, self.test_percentage, total_weight_expr=total_weight_expr))
        # self.samples.append(Sample(sample_path_down, label, normalization_weight, train_weight, self.test_percentage, total_weight_expr=total_weight_expr))

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
                sys_variation = False,
                gen_vars = False):

        self.event_category = event_category
        self.lumi = lumi
        self.evenSel = evenSel
        self.sys_variation = sys_variation

        self.shuffleSeed = shuffleSeed
        self.balanceSamples = balanceSamples

        self.binary_classification = input_samples.binary_classification
        if self.binary_classification: self.bkg_target = input_samples.bkg_target

        # loop over all input samples and load dataframe
        train_samples = []
        for sample in input_samples.samples:
            sample.load_dataframe(self.event_category, self.lumi, self.evenSel, self.sys_variation)
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

            # add flag for ttH to dataframe
            df["is_ttH"] = pd.Series( [1 if (c=="ttHbb" or c=="ttH") else 0 for c in df["class_label"].values], index = df.index )

#            print(df["class_label"].values)

            # add index labelling to dataframe
            df["index_label"] = pd.Series( [self.class_translation[c.replace("ttHbb", "ttH").replace("ttZbb","ttZ")] for c in df["class_label"].values], index = df.index )

            # norm weights to mean(1)
            df["train_weight"] = df["train_weight"]*df.shape[0]/len(self.classes)

            # save some meta data about network
            self.n_input_neurons = len(train_variables)
            self.n_output_neurons = len(self.classes)

        # binary classification labelling
        else:

            # class translations
            self.class_translation = {}
            self.class_translation["sig"] = 1
            self.class_translation["bkg"] = float(self.bkg_target)

            self.classes = ["sig", "bkg"]
            self.index_classes = [self.class_translation[c] for c in self.classes]

            df["index_label"] = pd.Series( [1 if c.replace("ttHbb","ttH").replace("ttZbb","ttZ") in input_samples.signal_classes else 0 for c in df["class_label"].values], index = df.index)

            sig_df = df.query("index_label == 1")
            bkg_df = df.query("index_label == 0")

            #sig_weight = sum(sig_df["train_weight"].values)
            #bkg_weight = sum(bkg_df["train_weight"].values)

            signal_weight = sum( sig_df["train_weight"].values )
            bkg_weight = sum( bkg_df["train_weight"].values )
            sig_df["train_weight"] = sig_df["train_weight"]/(2*signal_weight)*df.shape[0]
            bkg_df["train_weight"] = bkg_df["train_weight"]/(2*bkg_weight)*df.shape[0]
            #sig_df["class_label"] = "sig"
            #bkg_df["class_label"] = "bkg"
            sig_df["binaryTarget"] = 1.
            bkg_df["binaryTarget"] = float(self.bkg_target)

            df = pd.concat([sig_df, bkg_df])

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

        gen_dict={'Jet_GenJet_Eta[0]':'Jet_Eta[0]','Jet_GenJet_Eta[1]':'Jet_Eta[1]','Jet_GenJet_Eta[2]':'Jet_Eta[2]','Jet_GenJet_Eta[3]':'Jet_Eta[3]',
                  'Jet_GenJet_Pt[0]':'Jet_Pt[0]','Jet_GenJet_Pt[1]':'Jet_Pt[1]','Jet_GenJet_Pt[2]':'Jet_Pt[2]','Jet_GenJet_Pt[3]':'Jet_Pt[3]',
                  'GenTopLep_Lep_Phi[0]':'LooseLepton_Phi[0]','GenTopLep_Lep_Pt[0]':'LooseLepton_Pt[0]','GenTopLep_Lep_E[0]':'LooseLepton_E[0]',
                  'GenTopLep_Lep_Eta[0]':'LooseLepton_Eta[0]','Gen_MET_Pt':'Evt_MET_Pt', 'Gen_Ht_Jets':'Evt_HT_jets'}
        
        #print df['Jet_GenJet_Pt[0]']
        #print df['Jet_Pt[0]']
        if gen_vars:
            for i in gen_dict:
                df[gen_dict[i]]=df[i]
        #print df['Jet_Pt[0]']

        df[train_variables] = (df[train_variables] - df[train_variables].mean())/df[train_variables].std()
        self.norm_csv = norm_csv

        self.unsplit_df = df.copy()

        # split test sample
        n_test_samples = int( df.shape[0]*test_percentage)
        self.df_test = df.head(n_test_samples)
        self.df_train = df.tail(df.shape[0] - n_test_samples)
        self.df_test_unnormed = unnormed_df.head(n_test_samples)

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
            #print events

            # get multiplication factor
            print "maxEvents: ", maxEvents
            print "sample.nevents: ", sample.nevents
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

    def get_all_test_data(self, unnormed=True):
        if unnormed:  return self.df_test_unnormed
        else:         return self.df_test

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
