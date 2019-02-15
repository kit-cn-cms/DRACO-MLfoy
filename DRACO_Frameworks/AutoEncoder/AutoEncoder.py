import os
import sys
import numpy as np
import json

# local imports
filedir  = os.path.dirname(os.path.realpath(__file__))
DRACOdir = os.path.dirname(filedir)
basedir  = os.path.dirname(DRACOdir)
sys.path.append(basedir)

# import with ROOT
from pyrootsOfTheCaribbean.evaluationScripts import plottingScripts

# imports with keras
import utils.generateJTcut as JTcut
import data_frame

import keras
import keras.models as models
import keras.layers as layer
from keras import backend as K
import matplotlib.pyplot as plt
import pandas as pd

# Limit gpu usage
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))




class EarlyStoppingByLossDiff(keras.callbacks.Callback):
    def __init__(self, monitor = "loss", value = 0.01, min_epochs = 20, patience = 10, verbose = 0):
        super(keras.callbacks.Callback, self).__init__()
        self.val_monitor = "val_"+monitor
        self.train_monitor = monitor
        self.patience = patience
        self.n_failed = 0

        self.min_epochs = min_epochs
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs = {}):
        current_val = logs.get(self.val_monitor)
        current_train = logs.get(self.train_monitor)

        if current_val is None or current_train is None:
            warnings.warn("Early stopping requires {} and {} available".format(
                self.val_monitor, self.train_monitor), RuntimeWarning)

        if abs(current_val-current_train)/(current_train) > self.value and epoch > self.min_epochs:
            if self.verbose > 0:
                print("Epoch {}: early stopping threshold reached".format(epoch))
            self.n_failed += 1
            if self.n_failed > self.patience:
                self.model.stop_training = True


class AutoEncoder():
    def __init__(self, 
            save_path,
            input_samples,
            event_category,
            train_variables,
            batch_size      = 5000,
            train_epochs    = 500,
            early_stopping  = 10,
            optimizer       = None,
            loss_function   = "categorical_crossentropy",
            test_percentage = 0.2,
            eval_metrics    = None,
            additional_cut  = None):

        # save some information
        # list of samples to load into dataframe
        self.input_samples = input_samples

        # output directory for results
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs( self.save_path )

        # name of event category (usually nJet/nTag category)
        self.JTstring       = event_category
        self.event_category = JTcut.getJTstring(event_category)
        self.categoryLabel  = JTcut.getJTlabel(event_category)

        # list of input variables
        self.train_variables = train_variables

        # batch size for training
        self.batch_size = batch_size
        # number of training epochs
        self.train_epochs = train_epochs
        # number of early stopping epochs
        self.early_stopping = early_stopping
        # percentage of events saved for testing
        self.test_percentage = test_percentage

        # loss function for training
        self.loss_function = loss_function
        # additional metrics for evaluation of the training process
        self.eval_metrics = eval_metrics

        # additional cuts to be applied after variable norm
        self.additional_cut = additional_cut

        # load data set
        self.data = self._load_datasets()

        # save variable norm
        self.cp_path = self.save_path+"/checkpoints/"
        if not os.path.exists(self.cp_path):
            os.makedirs(self.cp_path)
        out_file = self.cp_path + "/variable_norm.csv"
        self.data.norm_csv.to_csv(out_file)
        print("saved variabe norms at "+str(out_file))

        # make plotdir
        self.plot_path = self.save_path+"/plots/"
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)

        # dict with architectures for analysis
        self.inputName = "inputLayer"
        self.outputName = "outputLayer"

        # optimizer for training
        if not optimizer:
            self.optimizer = keras.optimizers.Adam(1e-4)
        else:
            self.optimizer = optimizer

        
    def _load_datasets(self):
        ''' load data set '''

        return data_frame.DataFrame(
            input_samples       = self.input_samples,
            event_category      = self.event_category,
            train_variables     = self.train_variables,
            test_percentage     = self.test_percentage,
            norm_variables      = True,
            additional_cut      = self.additional_cut)


    def build_default_model(self):
        ''' default straight forward DNN '''
        K.set_learning_phase(True)
        number_of_input_neurons     = self.data.n_input_neurons

        number_of_neurons_per_layer = [10,20,30,40]
        dropout                     = 1
        activation_function         = "elu"
        #l2_regularization_beta      = 1e-5

        Inputs = keras.layers.Input(
            shape = (number_of_input_neurons,),
            name  = self.inputName)

        X = Inputs
        self.layer_list = [X]

        # loop over dense layers
        for iLayer, nNeurons in enumerate(number_of_neurons_per_layer):
            X = keras.layers.Dense( nNeurons,
                activation = activation_function,
                #kernel_regularizer = keras.regularizers.l2(l2_regularization_beta),
                name = "DenseLayer_"+str(iLayer)
                )(X)

            if not dropout == 1:
                X = keras.layers.Dropout(dropout)(X)

        # generate output layer
        X = keras.layers.Dense( self.data.n_output_neurons,
            activation = "linear",
            #kernel_regularizer = keras.regularizers.l2(l2_regularization_beta),
            name = self.outputName
            )(X)
        
        # output activation
        #X = keras.layers.LeakyReLU(alpha = 0.1)(X)

        # define model
        model = models.Model(inputs = [Inputs], outputs = [X])
        model.summary()

        return model

    def build_model(self, model = None):
        ''' build a DNN model
            if none is epecified use default model '''

        if model == None:
            print("Loading default model")
            model = self.build_default_model()

        # compile the model
        model.compile(
            loss = self.loss_function,
            optimizer = self.optimizer,
            metrics = self.eval_metrics)

        # save the model
        self.model = model

        # save net information
        out_file = self.save_path+"/model_summary.yml"
        yml_model = self.model.to_yaml()
        with open(out_file, "w") as f:
            f.write(yml_model)


    def train_model(self):
        ''' train the model '''

        # add early stopping if activated
        callbacks = None
        if self.early_stopping:
            callbacks = [EarlyStoppingByLossDiff(
                monitor = "loss",
                value = self.early_stopping,
                min_epochs = 50,
                patience = 10,
                verbose = 1)]

        # train main net
        self.trained_model = self.model.fit(
            x = self.data.get_train_data(as_matrix = True),
            y = self.data.get_train_data(as_matrix = True),
            batch_size = self.batch_size,
            epochs = self.train_epochs,
            shuffle = True,
            callbacks = callbacks,
            validation_split = 0.25,
            sample_weight = self.data.get_train_weights())

        self.save_model()

    def save_model(self):
        # save trained model
        out_file = self.cp_path + "/trained_model.h5py"
        self.model.save(out_file)
        print("saved trained model at "+str(out_file))

        model_config = self.model.get_config()
        out_file = self.cp_path +"/trained_model_config"
        with open(out_file, "w") as f:
            f.write( str(model_config))
        print("saved model config at "+str(out_file))

        out_file = self.cp_path +"/trained_model_weights.h5"
        self.model.save_weights(out_file)
        print("wrote trained weights to "+str(out_file))

        # set model as non trainable
        for layer in self.model.layers:
            layer.trainable = False
        self.model.trainable = False

        K.set_learning_phase(False)

        out_file = self.cp_path + "/trained_model"
        sess = K.get_session()
        saver = tf.train.Saver()
        save_path = saver.save(sess, out_file)
        print("saved checkpoint files to "+str(out_file))

    def eval_model(self):
        ''' evaluate trained model '''

        # prenet evaluation
        self.model_eval = self.model.evaluate(
            self.data.get_test_data(as_matrix = True),
            self.data.get_test_data(as_matrix = True))

        # save history of eval metrics
        self.model_history = self.trained_model.history

        # save predicitons
        self.model_prediction_vector = self.model.predict(
            self.data.get_test_data(as_matrix = True) )

        if self.eval_metrics:
            print("model test loss: {}".format(self.model_eval[0]))
            for im, metric in enumerate(self.eval_metrics):
                print("model test {}: {}".format(metric, self.model_eval[im+1]))

        # evaluate non trainiable data
        print("loading non trainable data")
        self.data.get_non_train_samples()
        if len(self.data.non_train_samples) == 0:
            print("... no additional data found")

        for sample in self.data.non_train_samples:
            sample.addPrediction( self.model, self.train_variables )
            sample.calculateLossVector( self.loss_function )

        # add prediction to encoder sample
        self.data.train_sample.setInputValues( self.data.get_test_data(as_matrix = True) )
        self.data.train_sample.setPredictionVector(self.model_prediction_vector)
        self.data.train_sample.setLumiWeights(self.data.get_lumi_weights())
        self.data.train_sample.calculateLossVector(self.loss_function)
                
        

    def plot_metrics(self):
        ''' plot history of loss function and evaluation metrics '''
        metrics = ["loss"]
        if self.eval_metrics: metrics += self.eval_metrics

        for metric in metrics:
            plt.clf()
            train_history = self.model_history[metric]
            val_history = self.model_history["val_"+metric]

            n_epochs = len(train_history)
            epochs = np.arange(1,n_epochs+1,1)

            plt.plot(epochs, train_history, "b-", label = "train", lw = 2)
            plt.plot(epochs, val_history, "r-", label = "validation", lw = 2)
            plt.title(self.categoryLabel.replace("\\geq",">="), loc = "right")

            plt.grid()
            plt.xlabel("epoch")
            plt.ylabel(metric)

            plt.legend()

            out_path = self.save_path + "/model_history_"+str(metric)+".pdf"
            plt.savefig(out_path)
            print("saved plot of "+str(metric)+" at "+str(out_path))


    def plot_lossValues(self, log = False):
        nbins = 30
        bin_range = [0.,3.]
        
        plotLoss = plottingScripts.plotLoss(
            train_sample        = self.data.train_sample,
            other_samples       = self.data.non_train_samples,
            loss_function       = self.loss_function,
            variables           = self.train_variables,
            nbins               = nbins,
            bin_range           = bin_range,
            event_category      = self.categoryLabel,
            plotdir             = self.plot_path,
            logscale            = log)

        plotLoss.plot_mean()
        plotLoss.plot_nodes()
        
    def plot_reconstruction(self, log = False):
        nbins = 30
        bin_range = [-2.,2.]

        plotReco = plottingScripts.plotReconstruction(
            sample              = self.data.train_sample,
            variables           = self.train_variables,
            nbins               = nbins,
            bin_range           = bin_range,
            event_category      = self.categoryLabel,
            plotdir             = self.plot_path,
            logscale            = log)
        
        plotReco.plot()
