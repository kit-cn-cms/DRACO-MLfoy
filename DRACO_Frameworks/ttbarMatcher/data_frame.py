import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.utils import shuffle

class InputFeatures:
    def __init__(self):
        self.objects = []
        self.objectVariables = {}
        self.variables = []

    def addObject(self, object, length = None, max = None):
        self.objects.append( {"name": object, "index": length, "max": max} )

    def addVariables(self, variables):
        if type(variables) == str:
            variables = [variables]
        self.variables += variables

    def generateVariableSet(self):
        variableSet = []
        self.nVariables = len(self.variables)
        self.indexVariables = []

        for obj in self.objects:
            objVariables = []
            # add variables if object has no index
            if not obj["index"]: 
                objVariables += ["{}_{}".format(obj["name"], v) for v in self.variables]

            else:
                # otherwise add indexing variable
                self.indexVariables.append(obj["index"])

                # loop over indices
                for idx in range(0,obj["max"]):
                    objVariables += ["{}_{}[{}]".format(obj["name"], v, idx) for v in self.variables]
            variableSet += objVariables
            self.objectVariables[obj["name"]] = objVariables

        print("-"*50)
        print("input features:")
        for v in variableSet: print(v)
        print("-"*50)

        self.variableSet = variableSet
        
class TargetFeatures:
    def __init__(self):
        self.objects = []
        self.variables = []
    
    def addObjects(self, objects):
        if type(objects) == str:
            objects = [objects]
        self.objects += objects

    def addVariables(self, variables):
        if type(variables) == str:
            variables = [variables]
        self.variables += variables

    def generateTargets(self):
        targetVariables = []
        self.nVariables = len(self.variables)
        for obj in self.objects:
            targetVariables += ["{}_{}".format(obj, v) for v in self.variables]

        print("-"*50)
        print("target variables:")
        for v in targetVariables: print(v)
        print("-"*50)

        self.nTargets = len(targetVariables)
        self.targetVariables = targetVariables


class Sample:
    def __init__(self, path, label, max_events = None):
        self.path = path
        self.label = label
        self.max_events = max_events

    def load_dataframe(self, variables = None):
        print("-"*50)
        print("loading sample file "+str(self.path))
        with pd.HDFStore( self.path, mode = "r" ) as store:
            df = store.select("data", stop = self.max_events, columns = variables)

        self.data = df
        print("-"*50)

class InputSamples:
    def __init__(self, input_path, max_events = None):
        self.input_path = input_path
        self.max_events = max_events
        self.samples = []

    def addSample(self, sample_path, label):
        path = self.input_path + "/" + sample_path
        self.samples.append( Sample(path, label, self.max_events) )
        

class DataFrame(object):
    def __init__(self, input_samples, input_features, target_features, feature_scaling = 1000., test_percentage = 0.2, val_percentage = 0.2):
        # list of variables
        self.variables = []
        self.variables += input_features.variableSet
        self.variables += input_features.indexVariables
        self.variables += target_features.targetVariables
        self.variables = list(set(self.variables))

        # loop over all input samples and load dataframe
        train_samples = []
        for sample in input_samples.samples:
            sample.load_dataframe(self.variables)
            train_samples.append(sample.data)
        
        # concatenating all dataframes
        df = pd.concat( train_samples )
        del train_samples

        # shuffle dataframe
        df = shuffle(df, random_state = 333)

        # apply some kind of data normalization
        df[input_features.variableSet] = df[input_features.variableSet]/feature_scaling
        df[target_features.targetVariables] = df[target_features.targetVariables]/feature_scaling

        self.unsplit_df = df.copy()

        # split test sample
        n_test_samples = int( df.shape[0]*test_percentage )
        n_val_samples = int( df.shape[0]*val_percentage )

        self.df_test = df.head(n_test_samples)
        df_val_and_train = df.tail(df.shape[0] - n_test_samples )

        self.df_val = df.head(n_val_samples)
        self.df_train = df.tail(df_val_and_train.shape[0] - n_val_samples)

        # print some counts
        print("total events after cuts:    "+str(df.shape[0]))
        print("events used for training:   "+str(self.df_train.shape[0]))
        print("events used for validation: "+str(self.df_val.shape[0]))
        print("events used for testing:    "+str(self.df_test.shape[0]))
        del df

        self.input_samples = input_samples
