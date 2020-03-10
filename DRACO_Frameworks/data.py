import pandas as pd
import os
import sys
import numpy as np
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.decomposition import PCA


# add samples 
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
        #self.addSampleSuffix = addSampleSuffix

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

        if self.addSampleSuffix in self.label:
            df["class_label"] = pd.Series([ c + self.addSampleSuffix for c in df["class_label"].values], index = df.index)

        # add lumi weight
        # adjust weights via 1/test_percentage such that yields in plots correspond to complete dataset

        df = df.assign(lumi_weight = lambda x: x.total_weight * lumi * self.train_weight * self.normalization_weight / self.test_percentage)
        print("sum of lumi weights: {}".format(sum(df["lumi_weight"].values)))
        self.data = df
        print("-"*50)

    '''
    relevant for saving model
    '''
    def getConfig(self):
        config = {}
        config["sampleLabel"] = self.label
        config["samplePath"] = self.path
        config["sampleWeight"] = self.normalization_weight
        config["sampleEvents"] = self.nevents
        config["min"] = self.min
        config["max"] = self.max
        return config


'''wrapper for samples'''
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
                category_label     = None,
                norm_variables = True,
                test_percentage = 0.2,
                lumi = 41.5,
                shuffleSeed = None,
                balanceSamples = True,
                evenSel = "",
                bkg_target = 0,
                signal = ["ttH"],
                combine = None,
                binary_classification = None,
                addSampleSuffix = ""):

    	self.event_category = event_category
        self.lumi = lumi
        self.evenSel = evenSel

        self.shuffleSeed = shuffleSeed
        self.balanceSamples = balanceSamples
        self.addSampleSuffix = addSampleSuffix

        self.binary_classification = binary_classification
        if self.binary_classification: 
            self.bkg_target = float(bkg_target)

        self.signal = signal
        self.combine = combine



        '''class translation for output nodes'''
        self.class_translation
        if self.binary_classification:
                self.class_translation["sig"] = 1
                self.class_translation["bkg"] = self.bkg_target
        else:
            index = 0

        '''loop over all input samples and load dataframe dependent on event category
        assign index number to sample class
        if combined sample class, renormalise new sample to 1'''
        train_samples = []
        for sample in input_samples.samples:
            sample.load_dataframe(self.event_category, self.lumi, self.evenSel)
            if self.binary_classification:
                if sample in self.signal:
                    sample.data["index_label"]=1
                    
                    sample.data["train_weight"] = sample.data["train_weight"]*df.shape[0]/len(self.signal)
                else:
                    sample.data["index_label"]=self.bkg_target
                    sample.data["train_weight"] = sample.data["train_weight"]*df.shape[0]/(len(self.signal)-len(input_samples.samples))
            else:
                if combine:
                    for combined_class in combine:
                        if sample in combine[combined_class]:
                            if not combined_class in self.class_translation:
                                self.class_translation[combined_class] = index
                                index += 1
                            sample.data["index_label"]  = self.class_translation[combined_class]
                            sample.data["train_weight"] = sample.data["train_weight"]*df.shape[0]/len(combine[combined_class])
                        else:
                            self.class_translation[sample.label] = index
                            sample.data["index_label"] = index
                            index += 1
                else:
                    self.class_translation[sample.label] = index
                    sample.data["index_label"] = index
                    index += 1
            train_samples.append(sample.data)

        # concatenating all dataframes
        df = pd.concat(train_samples, sort=True)
        del train_samples

        
        df["train_weight"] = df["train_weight"]*df.shape[0]/len(self.class_translation)


        # save some meta data about network
        self.n_input_neurons = len(train_variables)
        if self.binary_classification:
            self.n_output_neurons = 1
        else:
            self.n_output_neurons = len(self.classes)-input_samples.additional_samples


        # shuffle dataframe
        if not self.shuffleSeed:
           self.shuffleSeed = np.random.randint(low = 0, high = 2**16)

        print("using shuffle seed {} to shuffle input data".format(self.shuffleSeed))

        df = shuffle(df, random_state = self.shuffleSeed)
