import pandas as pd
from keras.utils import to_categorical
import numpy as np

class DataFrame( object ):
    def __init__(self, path_to_input_file, output_label = "class_label", 
                variables = [], n_particles = 10,
                normed_inputs = False, one_hot = True):
        print("-"*40)
        print("loading data from "+str(path_to_input_file))
        with pd.HDFStore( path_to_input_file, mode = "r") as store:
            df          = store.select("data")
            label_dict  = store.select("label_dict")

        columns = [var+"_"+str(i) for i in range(n_particles) for var in variables]
        self.X = df[columns]
        if normed_inputs:
            self.X = (self.X - self.X.mean())/self.X.std()

        self.input_shape        = (1,n_particles,len(variables))
        print("extracted input shape "+str(self.input_shape))
        self.input_size         = self.input_shape[0]*self.input_shape[1]*self.input_shape[2]
        print("size of input image: "+str(self.input_size))
        self.n_events           = df.shape[0]
        print("extracted number of events: "+str(self.n_events))

        # output labels
        print("chose '"+str(output_label)+"' as label for output classes")
        self.Y = df[output_label].values

        if output_label == "nJets":
            # TODO hardcoded restrictions to 12 jets not very elegant
            self.min_jets = 3
            self.max_jets = 12
            self.Y = [self.max_jets if j>=self.max_jets \
                    else self.min_jets if j<=self.min_jets \
                    else j for j in self.Y]

        if one_hot:
            # generating label vectors (one-hot-encoding)
            self.one_hot = to_categorical( self.Y )
            self.num_classes = self.one_hot.shape[1]
        else:
            self.num_classes = 1

        # loading label dictionary:
        label_dict = label_dict.to_dict('list')
        for key in label_dict:
            label_dict[key] = label_dict[key][0]
        
        if output_label == "class_label":
            # chose event type as classification goal
            self.inverted_label_dict = {val: key for key, val in label_dict.items()}
            self.label_dict = label_dict
        if output_label == "nJets":
            self.label_dict = {str(i)+" jets": i for i in range(self.num_classes)}
            self.inverted_label_dict = {val: key for key, val in self.label_dict.items()}

        # reshape as CNN inputs
        self.X = self.X.as_matrix().reshape(-1, *self.input_shape)
        print("data shape: {}".format( self.X.shape))
        print("-"*40)




