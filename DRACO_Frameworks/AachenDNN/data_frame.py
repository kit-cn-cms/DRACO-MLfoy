import pandas as pd
import numpy as np
from sklearn.utils import shuffle


class DataFrame(object):
    def __init__(self, inFile_path, classes, category, variables, intermediate_variables,
                test_percentage = 0.1, train_percentage = 0.5, norm_variables = False):

        dfs = []
        for cls in classes:
            class_file = inFile_path + "/" + cls + ".h5"
            print("loading class file "+str(class_file))
            with pd.HDFStore( class_file, mode = "r" ) as store:
                cls_df = store.select("data")
                print("number of events before selections: "+str(cls_df.shape[0]))

            # apply category
            cls_df.query(category, inplace = True)
            print("number of events after applying event category cut: "+str(cls_df.shape[0]))

            # add event weight
            weights = cls_df["Weight_XS"].values
            weight_sum = sum(weights)
            print("weight sum for "+str(cls)+ ": "+str(weight_sum))
            cls_df["train_weight"] = pd.Series( [1.*w/weight_sum for w in weights], index = cls_df.index )
            
            dfs.append( cls_df )

        # concatenating all dataframes
        df = pd.concat( dfs )



        # do whatever needed with the dataframes, eg. splitting into train/etc (TODO)
        df = shuffle(df)

        # generate dataframe with training variables
        self.X = df[ variables ]
        # generate dataframe with intermediate target variables
        self.intermediate_Y = df[ intermediate_variables ]
        # generate dataframe with train target
        self.Y = df["class_label"]
        # generate dataframe with training weights
        self.W = df["train_weight"]

        if norm_variables:
            self.X = (self.X - self.X.mean())/self.X.std()

        print(self.X.shape)
        print(self.Y.shape)
        print(self.intermediate_Y.shape)
        print(self.W.shape)


