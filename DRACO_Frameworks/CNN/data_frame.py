import pandas as pd
import os
import sys
import numpy as np
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import re
import base64

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
        self.shape=[1,1]
	self.hist_data = []

    def load_dataframe(self, event_category, lumi, evenSel = "", phi_padding=0):
        
	#read h5 file
        print("-"*50)
        print("loading sample file "+str(self.path))
        with pd.HDFStore( self.path, mode = "r" ) as store:
            df = store.select("data")#, stop=100)#stop=50000)#, stop=100) # for debbuging
            print("number of events before selections: "+str(df.shape[0]))
            mi = store.select("meta_info")
            self.shape=list(mi["input_shape"])
            print("picture size: "+str(self.shape))
        if phi_padding!=0:
            phi_padding=abs(phi_padding)
            print("utilizing phi-padding with factor "+str(phi_padding))
            npixel_topad=int(round(self.shape[1]*phi_padding))
            shape_padded=[self.shape[0],int(self.shape[1]+2*npixel_topad)]
            print("picture size after padding: "+str(shape_padded))

        #=====================================
        #transform base64 back and phi-padding
        #=====================================
        
	#find channels used for training
        columns_to_decode=[]
        for col in df.columns:
            m=re.match("(.*_Hist)", col)
            if m!=None:
                columns_to_decode.append(m.group(1)) 
              
	#decode and save maximum value for later 
        H_List_Dict={col:list() for col in columns_to_decode}
	self.hist_data = []
        
        for i in range(len(columns_to_decode)):
            counter = 0
	    column_name = columns_to_decode[i]
            empty_imgs_evtids=[]
            gr_0 = []
            print column_name
            for index, row in df.iterrows():
                counter += 1
                if counter%1000 == 0: 
                   print counter,"events decoded in channel", column_name

                r=base64.b64decode(row[column_name])
                u=np.frombuffer(r,dtype=np.float64)
           
                ###### old normalization!!!
                #maxjetinevt=np.max(u)
                #if(maxjetinevt!=0):
                #    u=u/maxjetinevt
                #else:
                #    empty_imgs_evtids.append(index[2])
                ######
           
                u=np.reshape(u,self.shape)
		
                for line in u:
                    for element in line:
                        if element > 0.:
                            gr_0.append(element)

                if phi_padding != 0:
                    u=np.concatenate((u[:,-npixel_topad:],u,u[:,:npixel_topad]), axis=1)
                H_List_Dict[column_name].append(u)
	    df[column_name]=H_List_Dict[column_name]
	    
            self.hist_data.append(gr_0)

            print("====> "+str(len(empty_imgs_evtids))+" empty images found in channel "+column_name)

        #event_list=np.array(H_List_Dict[columns_to_decode[0]])

        if phi_padding!=0:
            self.shape=shape_padded
        self.shape.append(len(columns_to_decode))
        #print(self.shape)

        '''
        # phipadding debugging plot test
        n=np.zeros(self.shape)
        for m in event_list:
            n+=np.reshape(m,self.shape)
        import matplotlib.pyplot as plt
        plt.imshow(np.transpose(n), aspect = 'equal', interpolation="none", origin="lower", cmap="Blues")
        plt.show()
        exit()
        '''

        #event_list=event_list.reshape(-1,*self.shape)
        #traindata=np.expand_dims(self.df_train[ self.train_variables[0] ].values,axis=3)
        #debugdata=df[columns_to_decode[0]].values
        #debugdata=np.expand_dims(np.stack(debugdata),axis=3)
        #print(debugdata)
        #print(debugdata.shape)
        #print(debugdata[2].shape)
        #exit()
        #print(event_list)



        
        #=====================

        # apply event category cut
        #query = event_category
        #if not evenSel == "":
        #    query+=" and "+evenSel
        #df.query(query, inplace = True)
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
        self.input_shape = None

    def addSample(self, sample_path, label, normalization_weight=1., train_weight=1., total_weight_expr='x.Weight_XS * x.Weight_CSV * x.Weight_GEN_nom'):
        if self.activate_samples and not label in self.activate_samples:
            print("skipping sample {}".format(label))
            return
        if not os.path.isabs(sample_path):
            sample_path = self.input_path + "/" + sample_path
        self.samples.append(Sample(sample_path, label, normalization_weight, train_weight, self.test_percentage, total_weight_expr=total_weight_expr))

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
                norm_variables = True,
                test_percentage = 0.2,
                lumi = 41.5,
                shuffleSeed = None,
                balanceSamples = True,
                evenSel = "",
                phi_padding = 0,
		normed_to = 1.,
                pseudoData = False):

        self.event_category = event_category
        self.lumi = lumi
        self.evenSel = evenSel
        self.phi_padding = phi_padding

        self.shuffleSeed = shuffleSeed
        self.balanceSamples = balanceSamples
	
	self.normed_to = normed_to		

        self.binary_classification = input_samples.binary_classification
        if self.binary_classification: self.bkg_target = input_samples.bkg_target

        # loop over all input samples and load dataframe
        train_samples = []
	entries = []
        self.input_shape = None
	
        for sample in input_samples.samples:
            sample.load_dataframe(self.event_category, self.lumi, self.evenSel, self.phi_padding)
            train_samples.append(sample.data)
	    entries.append(sample.hist_data)
            if not self.input_shape is None:
                if not self.input_shape == sample.shape:
                    sys.exit("input shapes do not match")
            self.input_shape = sample.shape 

        # concatenating all dataframes
        df = pd.concat(train_samples, sort=True)
        del train_samples

	#####################################################################################	
	
        #normalise data
	
	hist_data = []
	
	for i in range(len(train_variables)):
	        hist_data.append(entries[0][i]+entries[1][i])
                
        #print(hist_data)

        print('quantile set to ' + str(self.normed_to * 100) +'%')
	quantile = [np.quantile(data, normed_to) for data in hist_data]
		
	for i in range(len(train_variables)):
           counter = 0
           var = train_variables[i]     
	   normalisedData = []
	   for index, row in df.iterrows():
               counter += 1
               if counter%100000 == 0: 
                   print counter,"events normalised in channel", var

                # norm values according to quantile
               if not var == 'Jet_CSV[0-16]_Hist':
	   	   values = row[var]/quantile[i]
            
               # values in CSV channel are already normed
	       else:
                   values = row[var]/1.
                
               # set all values greater than 1 to 1 
	       for line in values:
	      	   for j in range(len(line)):
		      if line[j] > 1.: line[j] = 1.

                      # generate pseudo data: every entry is 1
                      if pseudoData and line[j] != 0.: 
                          line[j] = 1.
	     	   
	       normalisedData.append(values)
	   df[var] = normalisedData
        
        
        #####################################################################################         
        
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
            self.class_translation["bkg"] = 0

            self.classes = ["sig", "bkg"]
            self.index_classes = [self.class_translation[c] for c in self.classes]

            df["index_label"] = pd.Series( [1 if c.replace("ttHbb","ttH").replace("ttZbb","ttZ") in input_samples.signal_classes else 0 for c in df["class_label"].values], index = df.index)

            sig_df = df.query("index_label == 1")
            bkg_df = df.query("index_label == 0")

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

            df = pd.concat([sig_df, bkg_df])

            self.n_input_neurons = len(train_variables)
            self.n_output_neurons = 1

        # shuffle dataframe
        if not self.shuffleSeed:
           self.shuffleSeed = np.random.randint(low = 0, high = 2**16)

        print("using shuffle seed {} to shuffle input data".format(self.shuffleSeed))

        df = shuffle(df, random_state = self.shuffleSeed)
        #debug print of df after shuffle
        #print(df)


        self.unsplit_df = df.copy()

        # split test sample
        n_test_samples = int( df.shape[0]*test_percentage)
        self.df_test = df.head(n_test_samples)
        self.df_train = df.tail(df.shape[0] - n_test_samples)


        # save variable lists
        self.train_variables = train_variables
        print(self.train_variables)
        self.output_classes = self.classes
        self.input_samples = input_samples

    

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
            if self.binary_classification: class_label = 'sig' if class_label in self.input_samples.signal_classes else 'bkg'

            events = self.df_train.query("(class_label == '{}')".format(class_label))
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
        if as_matrix:
            if len(self.train_variables)==1:
                traindata=np.expand_dims(np.stack(self.df_train[ self.train_variables[0] ].values),axis=3)
            else:
                df_variables_tmp=[np.expand_dims(np.stack(self.df_train[ channel ].values), axis=3) for channel in self.train_variables]
                traindata=np.concatenate(df_variables_tmp,axis=3)
            


            return traindata

        else:     
            return self.df_train[ self.train_variables ]#not adjusted for cnn yet

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
        if not normed: 
            return self.df_test_unnormed[ self.train_variables ]#not adjusted for cnn yet
        if as_matrix: 
            if len(self.train_variables)==1:
                testdata=np.expand_dims(np.stack(self.df_test[ self.train_variables[0] ].values),axis=3)
            else:
                df_variables_tmp=[np.expand_dims(np.stack(self.df_test[ channel ].values), axis=3) for channel in self.train_variables]
                testdata=np.concatenate(df_variables_tmp,axis=3)
            return testdata
        else:          
            return self.df_test[ self.train_variables ]#not adjusted for cnn yet

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
