# global imports
import keras
import keras.models as models
import keras.layers as layer

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import os

# Limit gpu usage
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# local imports
import data_frame

class DNN():
    def __init__(self, in_path, save_path,
                event_classes, 
                event_category,
                train_variables, 
                prenet_targets,
                batch_size = 4000,
                train_epochs = 500,
                early_stopping = None,
                optimizer = "adam",
                loss_function = "categorical_crossentropy",
                test_percentage = 0.2,
                eval_metrics = None)

        # save some information
        
        # path to input files
        self.in_path = in_path
        # output directory for results
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs( self.save_path )
        # list of classes
        self.event_classes = event_classes
        # name of event category (usually nJet/nTag category)
        self.event_category = event_category
        # target variables for pre-net
        self.prenet_targets = prenet_targets

        # list of input features
        self.train_variables = train_variables

        # batch size for training
        self.batch_size = batch_size
        # number of maximum training epochs
        self.train_epochs = train_epochs
        # number of early stopping epochs
        self.early_stopping = early_stopping
        # percentage of events saved for testing
        self.test_percentage = test_percentage        

        # optimizer for training
        self.optimizer = optimizer
        # loss function for training
        self.loss_function = loss_function
        # additional metrics for evaluation of training process
        self.eval_metrics = eval_metrics


    def load_datasets(self):
        ''' load dataset '''
        self.data = data_frame.DataFrame(
            path_to_input_files = self.in_path,
            classes             = self.event_classes,
            event_category      = self.event_category,
            train_variables     = self.train_variables,
            prenet_targets      = self.prenet_targets,
            test_percentage     = self.test_percentage,
            norm_variables      = True)

    def build_default_model(self):
        ''' default Aachen-DNN model as used in the analysis '''

        # TODO
        return pre_net, main_net


    def build_model(self, pre_net = None, main_net = None):
        ''' build a DNN model
            if none is specified use default model '''

        if pre_net == None or main_net == None:
            print("loading default models")
            pre_net, main_net = self.build_default_model()
        
        # compile models
        pre_net.compile(
            loss = self.loss_function,
            optimizer = self.optimizer,
            metrics = self.eval_metrics,
            loss_weights = self.data.get_train_weights())
    
        main_net.compile(
            loss = self.loss_function,
            optimizer = self.optimizer,
            metrics = self.eval_metrics,
            loss_weights = self.data.get_train_weights())
            
        self.pre_net = pre_net
        self.main_net = main_net

        # model summaries
        self.pre_net.summary()
        self.main_net.summary()

        out_file = self.save_path+"/pre_net_summmary.yml"
        yml_pre_net = self.pre_net.to_yaml()
        with open(out_file, "w") as f:
            f.write(yml_pre_net)

        out_file = self.save_path+"/main_net_summmary.yml"
        yml_main_net = self.main_net.to_yaml()
        with open(out_file, "w") as f:
            f.write(yml_main_net)


    def train_models(self):
        ''' train prenet first then the main net '''

        if self.early_stopping:
            callbacks = [keras.callbacks.EarlyStopping(
                            monitor = "val_loss", 
                            patience = self.early_stopping)]
        else: callbacks = None

        self.trained_pre_net = self.pre_net.fit(
            x = self.data.get_train_data(as_matrix = True),
            y = self.data.get_prenet_train_labels(),
            batch_size = self.batch_size,
            epochs = self.train_epochs,
            shuffle = True,
            callbacks = callbacks
            # TODO implement cross validation
            )

        # save trained model
        out_file = self.save_path = "/trained_pre_net.h5py"
        self.pre_net.save(out_file)
        print("saved trained prenet model at "*str(out_file))

        prenet_config = self.pre_net.get_config()
        out_file = self.save_path +"/trained_pre_net_config"
        with open(out_file, "w") as f:
            f.write( str(prenet_config))
        print("saved prenet model config at "+str(out_file))

        out_file = self.save_path +"/trained_pre_net_weights.h5"
        self.pre_net.save_weights(out_file)
        print("wrote trained prenet weights to "+str(out_file))


        # train main net
        self.trained_main_net = self.main_net.fit(
            x = None, # TODO implement main net input
            y = self.data.get_train_labels(),
            batch_size = self.batch_size,
            epochs = self.train_epochs,
            shuffle = True,
            callbacks = callbacks
            # TODO implement cross validation
            )

        # save trained model
        out_file = self.save_path = "/trained_main_net.h5py"
        self.main_net.save(out_file)
        print("saved trained model at "*str(out_file))

        mainnet_config = self.main_net.get_config()
        out_file = self.save_path +"/trained_main_net_config"
        with open(out_file, "w") as f:
            f.write( str(mainnet_config))
        print("saved model config at "+str(out_file))

        out_file = self.save_path +"/trained_main_net_weights.h5"
        self.main_net.save_weights(out_file)
        print("wrote trained weights to "+str(out_file))


            





