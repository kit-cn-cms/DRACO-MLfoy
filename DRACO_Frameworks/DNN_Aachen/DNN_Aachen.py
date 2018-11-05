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
        
        number_of_input_neurons = self.data.n_input_neurons

        number_of_neurons_per_layer = [100,100]
        Dropout                     = [0.7,0.7]
        activation_function         = "relu"
        lw_regularization_beta      = 0.0001
    
        # prenet
        Inputs = layer.Input( shape = (self.data.n_input_neurons,) )

        X = Inputs
        layer_list = [X]
        for i, nNeurons in enumerate(number_of_neurons_per_layer):
            Dense = layer.Dense(nNeurons, activation = activation_function,
                                kernel_regularize = keras.regularizers.l2(l2_regularization_beta),
                                name = "Dense_"+str(i))(X)

            layer_lists.append( Dense )
            if dropout[i] != 1: 
                X = layer.Dropout( drouput[i] )(Dense)
        
        X = layer.Dense(self.data.n_prenet_output_neurons,
                activation = "sigmoid",
                kernel_regularizer = keras.regularizers.l2(l2_regularization_beta))(X)
        layers_list.append(X)

        pre_net = models.Model(inputs = [Inputs], outputs = [X])
        pre_net.summary()

        # compile and fit here?

        # Make Parameters of first model untrainable
        for layer in first_model.layers:
            layer.trainable = False

        # ---------------
        # main net
        number_of_neurons_per_layer = [100, 100]
        dropout                     = [0.7, 0.7]        

        # Create Input/conc layer for second NN
        conc_layer = layer.concatenate(layers_list, axis = -1)

        Y = conc_layer

        for i, nNeurons in enumerate(number_of_neurons_per_layer):
            Y = layer.Dense(nNeurons, activation = activation_function,
                            kernel_regularizer=keras.regularizers.l2(l2_regularization_beta),
                            name = "Dense_main_"+str(i))(Y)

            if dropout[i] != 1:
                Y = layer.Dropout(dropout[i])(Y)

        Y = layer.Dense(self.data.n_output_neurons,
                activation = "categorical_crossentropy",
                kernel_regularizer=keras.regularizers.l2(l2_regularization_beta))(Y)

        pre_net.trainable = False
        main_net = models.Model(inputs = [Inputs], outputs = [Y])
        main_net.summary()

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

        callbacks = None
        if self.early_stopping:
            callbacks = [keras.callbacks.EarlyStopping(
                            monitor = "val_loss", 
                            patience = self.early_stopping)]

        self.trained_pre_net = self.pre_net.fit(
            x = self.data.get_train_data(as_matrix = True),
            y = self.data.get_prenet_train_labels(),
            batch_size = self.batch_size,
            epochs = self.train_epochs,
            shuffle = True,
            callbacks = callbacks,
            validation_split=0.2,
            )

        # save trained model
        out_file = self.save_path = "/trained_pre_net.h5py"
        self.pre_net.save(out_file)
        print("saved trained prenet model at "+str(out_file))

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
            x = self.data.get_train_data(as_matrix = True),
            y = self.data.get_train_labels(),
            batch_size = self.batch_size,
            epochs = self.train_epochs,
            shuffle = True,
            callbacks = callbacks,
            validation_split=0.2,
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


    def eval_model(self):
        ''' evaluate trained model '''

        # prenet evaluation
        self.prenet_eval = self.pre_net.evaluate(
            self.data.get_test_data(as_matrix = True))
        print("prenet test loss: {}".format(self.prenet_eval[0]))
        for im, metric in enumerate(self.eval_metrics):
            print("prenet test {}: {}".format(metric, self.test_eval[im+1]))

        self.prenet_history = self.trained_pre_net.history

        self.prenet_predicted_vector = self.pre_net.predict( self.data.get_test_data(as_matrix = True) )


        # main net evaluation
        self.mainnet_eval = self.main_net.evaluate(
            x = self.data.get_test_data(as_matrix = True)
            )
        print("mainnet test loss: {}".format(self.mainnet_eval[0]))
        for im, metric in enumerate(self.eval_metrics):
            print("mainnet test {}: {}".format(metric, self. test_eval[im+1]))

        self.mainnet_history = self.trained_main_net.history

        self.mainnet_predicted_vector = self.main_net.predict() # TODO implement main net input

        self.predicted_classes = np.argmax( self.mainnet_predicted_vector, axis = 1)
    
        self.confusion_matrix = confusion_matrix(
            self.get_test_labels(), self.predicted_classes)

        

    # --------------------------------------------------------------------
    # result plotting functions
    # --------------------------------------------------------------------

    def plot_metrics(self):
        ''' plot history of loss function and evaluation metrics '''


        metrics = ["loss"]+self.eval_metrics

        for metric in metrics:
            # prenet plot
            plt.clf()
            train_history = self.prenet_history[metric]
            val_history = self.prenet_history["val_"+metric]

            n_epochs = len(train_history)
            epochs = np.arange(1,n_epochs+1,1)

            plt.plot(epochs, train_history, "b-", label = "train", lw = 2)
            plt.plot(epochs, val_history, "r-", label = "validation", lw = 2)
            plt.title("train and validation "+str(metric)+" of prenet")

            plt.grid()
            plt.xlabel("epoch")
            plt.ylabel(metric)

            plt.legend()

            out_path = self.save_path + "/prenet_history_"+str(metric)+".pdf"
            plt.savefig(out_path)
            print("saved plot of "+str(metric)+" at "+str(out_path))

            # main net
            plt.clf()
            train_history = self.mainnet_history[metric]
            val_history = self.mainnet_history["val_"+metric]

            n_epochs = len(train_history)
            epochs = np.arange(1,n_epochs+1,1)

            plt.plot(epochs, train_history, "b-", label = "train", lw = 2)
            plt.plot(epochs, val_history, "r-", label = "validation", lw = 2)
            plt.title("train and validation "+str(metric)+" of mainnet")

            plt.grid()
            plt.xlabel("epoch")
            plt.ylabel(metric)

            plt.legend()

            out_path = self.save_path + "/mainnet_history_"+str(metric)+".pdf"
            plt.savefig(out_path)
            print("saved plot of "+str(metric)+" at "+str(out_path))



















