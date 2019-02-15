import pandas as pd
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

        weight_sum = sum(df["total_weight"].values)
        df = df.assign(train_weight = lambda x: x.total_weight/weight_sum)
        print("weight sum of train_weight: "+str( sum(df["train_weight"].values) ))

        # add lumi weight
        df = df.assign(lumi_weight = lambda x: x.Weight_XS * x.Weight_GEN_nom * lumi * self.normalization_weight)
        print("yield (sum of weights): {}".format(df["lumi_weight"].sum()))

        self.data = df
        print("-"*50)
        

    def addPrediction(self, model, train_variables):
        self.input_values = self.data[train_variables].values
        self.prediction_vector = model.predict(self.input_values)
        self.lumi_weights = self.data["lumi_weight"].values

    def setInputValues(self, values):
        self.input_values = values

    def setPredictionVector(self, vector):
        self.prediction_vector = vector

    def setLumiWeights(self, weights):
        self.lumi_weights = weights

    def calculateLossVector(self, loss_function):
        if loss_function == "mean_squared_error":
            loss = (self.input_values - self.prediction_vector)**2
            self.lossMatrix = loss
            self.lossVector = np.mean(loss, axis = 1)
        else:
            print("invalis loss_function {}".format(loss_function))


class InputSamples:
    def __init__(self, input_path):
        self.input_path = input_path
        self.samples = []
        self.encoder_sample = None

    def addEvalSample(self, sample_path, label, normalization_weight = 1.):
        path = self.input_path + "/" + sample_path
        self.samples.append( Sample(path, label, normalization_weight) )
        
    def addEncoderSample(self, sample_path, label, normalization_weight = 1.):
        if self.encoder_sample:
            print("this auto encoder is designed to train on a single sample, you added more than one encoder samples")
            exit()
        
        path = self.input_path + "/" + sample_path
        self.encoder_sample = Sample(path, label, normalization_weight)

class DataFrame(object):
    def __init__(self,
                input_samples,
                event_category,
                train_variables,
                test_percentage = 0.1,
                norm_variables = False,
                additional_cut = None,
                lumi = 41.5):

        ''' takes a path to a folder where one h5 per class is located
            the events are cut according to the event_category
            variables in train_variables are used as input variables
            the dataset is shuffled and split into a test and train sample
                according to test_percentage
            for better training, the variables can be normed to std(1) and mu(0) '''
        self.event_category = event_category
        self.lumi = lumi

        # get train sample
        self.train_sample = input_samples.encoder_sample
        print(self.train_sample)
        self.train_sample.load_dataframe(self.event_category, self.lumi)

        df = self.train_sample.data

        # norm weights to mean(1)
        df["train_weight"] = df["train_weight"]*df.shape[0]

        # save some meta data about network
        self.n_input_neurons = len(train_variables)
        self.n_output_neurons = len(train_variables)

        # shuffle dataframe
        df = shuffle(df, random_state = 333)

        # norm variables if wanted
        unnormed_df = df.copy()
        if norm_variables:
            norm_csv = pd.DataFrame(index=train_variables, columns=["mu", "std"])
            tmp_train_variables = []
            for v in train_variables:
                norm_csv["mu"][v] = unnormed_df[v].mean()
                norm_csv["std"][v] = unnormed_df[v].std()
                if norm_csv["std"][v] == 0.:
                    print("std. deviation of variable {} is zero - please disable it for the future")
                    continue
                tmp_train_variables.append(v)
            train_variables = tmp_train_variables
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
        self.output_classes = train_variables
        self.input_samples = input_samples

        # init non trainable samples
        self.non_train_samples = None

    def get_non_train_samples(self):
        # get samples with flag 'isTrainSample == False' and load those
        samples = []
        for sample in self.input_samples.samples:
            sample.load_dataframe(self.event_category, self.lumi)
            sample.data[self.train_variables] = (sample.data[self.train_variables] - self.norm_csv["mu"])/(self.norm_csv["std"])
            samples.append(sample)
        self.non_train_samples = samples
        

    # train data -----------------------------------
    def get_train_data(self, as_matrix = True):
        if as_matrix: return self.df_train[ self.train_variables ].values
        else:         return self.df_train[ self.train_variables ]

    def get_train_weights(self):
        return self.df_train["train_weight"].values

    # test data ------------------------------------
    def get_test_data(self, as_matrix = True, normed = True):
        if not normed: return self.df_test_unnormed[ self.train_variables ]
        if as_matrix:  return self.df_test[ self.train_variables ].values
        else:          return self.df_test[ self.train_variables ]

    def get_test_weights(self):
        return self.df_test["total_weight"].values
    def get_lumi_weights(self):
        return self.df_test["lumi_weight"].values

    # full sample ----------------------------------
    def get_full_df(self):
        return self.unsplit_df[self.train_variables]
