import os
import sys
import numpy as np
import json
import pickle
import math
from array import array
import ROOT
import tqdm
import matplotlib.pyplot as plt
from functools import partial, update_wrapper
import csv


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
import Derivatives
from Derivatives import Inputs, Outputs, Derivatives

import tensorflow.keras as keras
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.models as models
import tensorflow.keras.layers as layer
from tensorflow.keras import backend as K
import pandas as pd

# Limit gpu usage
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability.python.distributions as tfd

import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

class EarlyStopping(tf.keras.callbacks.Callback):
    ''' custom implementation of early stopping
        with options for
            - stopping when val/train loss difference exceeds a percentage threshold
            - stopping when val loss hasnt increased for a set number of epochs '''

    def __init__(self, monitor = "loss", value = None, min_epochs = 20, stopping_epochs = None, patience = 10, verbose = 0):
        super(keras.callbacks.Callback, self).__init__()
        self.val_monitor = "val_"+monitor
        self.train_monitor = monitor
        self.patience = patience
        self.n_failed = 0

        self.stopping_epochs = stopping_epochs
        self.best_epoch = 0
        self.best_validation = 999.
        self.min_epochs = min_epochs
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs = {}):
        current_val = logs.get(self.val_monitor)
        if epoch == 0:
            self.best_validation = current_val
        current_train = logs.get(self.train_monitor)

        if current_val is None or current_train is None:
            warnings.warn("Early stopping requires {} and {} available".format(
                self.val_monitor, self.train_monitor), RuntimeWarning)

        if current_val < self.best_validation:
            self.best_validation = current_val
            self.best_epoch = epoch

        # check loss by percentage difference
        if self.value:
            if (current_val-current_train)/(current_train) > self.value and epoch > self.min_epochs:
                if self.verbose > 0:
                    print("\nEpoch {}: early stopping threshold reached".format(epoch))
                self.n_failed += 1
                if self.n_failed > self.patience:
                    self.model.stop_training = True

        # check loss by validation performance increase
        if self.stopping_epochs:
            if self.best_epoch + self.stopping_epochs < epoch and epoch > self.min_epochs:
                if self.verbose > 0:
                    print("\nValidation loss has not decreased for {} epochs".format( epoch - self.best_epoch ))
                self.model.stop_training = True

class BNN_Flipout():
    def __init__(self,
            save_path,
            input_samples,
            category_name,
            train_variables,
            category_cutString        = None,
            category_label            = None,
            norm_variables            = True,
            qt_transformed_variables  = True,
            restore_fit_dir           = None,
            train_epochs              = 500,
            test_percentage           = 0.2,
            eval_metrics              = None,
            shuffle_seed              = None,
            balanceSamples            = False,
            evenSel                   = None,
            sys_variation             = False,
            gen_vars                  = False):

        # save some information
        # list of samples to load into dataframe
        self.input_samples = input_samples

        # output directory for results
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs( self.save_path )

        # name of event category (usually nJet/nTag category)
        self.category_name = category_name

        # string containing event selection requirements;
        # if not specified (default), deduced via JTcut
        self.category_cutString = (category_cutString if category_cutString is not None else JTcut.getJTstring(category_name))
        # category label (string);
        # if not specified (default), deduced via JTcut
        self.category_label = (category_label if category_label is not None else JTcut.getJTlabel (category_name))

        # selection
        self.evenSel = ""
        self.oddSel = "1."
        if not evenSel == None:
            if evenSel == True:
                self.evenSel = "(Evt_Odd==0)"
                self.oddSel  = "(Evt_Odd==1)"
            elif evenSel == False:
                self.evenSel = "(Evt_Odd==1)"
                self.oddSel  = "(Evt_Odd==0)"

        # list of input variables
        self.train_variables = train_variables

        # percentage of events saved for testing
        self.test_percentage = test_percentage

        # number of train epochs
        self.train_epochs = train_epochs

        # additional metrics for evaluation of the training process
        self.eval_metrics = eval_metrics

        # normalize variables in DataFrame
        self.norm_variables = norm_variables
        self.qt_transformed_variables = qt_transformed_variables
        self.restore_fit_dir = restore_fit_dir

        # load data set
        self.data = self._load_datasets(shuffle_seed, balanceSamples, sys_variation, gen_vars)
        self.event_classes = self.data.output_classes

        # save variable norm
        self.cp_path = self.save_path+"/checkpoints/"
        if not os.path.exists(self.cp_path):
            os.makedirs(self.cp_path)

        if self.norm_variables or self.qt_transformed_variables:
           out_file = self.cp_path + "/variable_norm.csv"
           self.data.norm_csv.to_csv(out_file)
           print("saved variable norms at "+str(out_file))

        # make plotdir
        self.plot_path = self.save_path+"/plots/"
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)

        # make eventdir
        self.event_path = self.save_path+"/events/"
        if not os.path.exists(self.event_path):
            os.makedirs(self.event_path)

        # layer names for in and output (needed for c++ implementation)
        self.inputName = "inputLayer"
        self.outputName = "outputLayer"
    
    def _load_datasets(self, shuffle_seed, balanceSamples, sys_variation, gen_vars):
        ''' load data set '''
        return data_frame.DataFrame(
            input_samples            = self.input_samples,
            event_category           = self.category_cutString,
            train_variables          = self.train_variables,
            test_percentage          = self.test_percentage,
            norm_variables           = self.norm_variables,
            qt_transformed_variables = self.qt_transformed_variables,
            restore_fit_dir          = self.restore_fit_dir,
            shuffleSeed              = shuffle_seed,
            balanceSamples           = balanceSamples,
            evenSel                  = self.evenSel,
            sys_variation            = sys_variation,
            gen_vars                 = gen_vars,
        )
    
    def _load_architecture(self, config):
        ''' load the architecture configs '''

        # define default network configuration
        self.architecture = {
          "layers":                   [20],
          "loss_function":            "neg_log_likelihood",
          # "Dropout":                  0.2,
          "batch_size":               2000,
          "optimizer":                optimizers.Adam(1e-3),
          "activation_function":      "relu",
          "output_activation":        "Sigmoid",
          "earlystopping_percentage": None,
          "earlystoppi# sampling output values from the intern tensorflow output distribution
    def bnn_calc_mean_std(self, n_samples=50):
        test_pred  = []
        print "Calculating the mean and std: "
        for i in tqdm.tqdm(range(n_samples)):
            test_pred_vector = self.model.predict(self.data.get_test_data(as_matrix = True))
            test_pred.append(test_pred_vector)
            test_preds = np.concatenate(test_pred, axis=1)
        return np.mean(test_preds, axis=1), np.std(test_preds, axis=1), test_predsng_epochs":     None,
        }

        for key in config:
            self.architecture[key] = config[key]


    # sampling output values from the intern tensorflow output distribution
    def bnn_calc_mean_std(self, n_samples=50):
        test_pred  = []
        print "Calculating the mean and std: "
        for i in tqdm.tqdm(range(n_samples)):
            test_pred_vector = self.model.predict(self.data.get_test_data(as_matrix = True))
            test_pred.append(test_pred_vector)
            test_preds = np.concatenate(test_pred, axis=1)
        return np.mean(test_preds, axis=1), np.std(test_preds, axis=1), test_preds
    

    def build_default_model(self):
        ''' build default straight forward BNN from architecture dictionary '''

        # infer number of input neurons from number of train variables
        number_of_input_neurons     = self.data.n_input_neurons

        # get all the architecture settings needed to build model
        number_of_neurons_per_layer = self.architecture["layers"]
        dropout                     = self.architecture["Dropout"]
        activation_function         = self.architecture["activation_function"]
        output_activation           = self.architecture["output_activation"]

        # Specify the posterior distributions for kernel and bias
        def posterior(kernel_size, bias_size=0, dtype=None):
            from tensorflow_probability import layers
            from tensorflow_probability import distributions as tfd
            import numpy as np
            import tensorflow as tf
            n = kernel_size + bias_size
            c = np.log(np.expm1(1.))
            return tf.keras.Sequential([
                layers.VariableLayer(2 * n, dtype=dtype),
                layers.DistributionLambda(lambda t: tfd.Independent(tfd.Normal(loc=t[..., :n], scale=1e-5 + tf.math.softplus(c + t[..., n:])), reinterpreted_batch_ndims=1)),
                ])

        # Specify the prior distributions for kernel and bias
        def prior(kernel_size, bias_size=0, dtype=None):
            from tensorflow_probability import layers
            from tensorflow_probability import distributions as tfd
            import numpy as np
            import tensorflow as tf
            n = kernel_size + bias_size
            c = np.log(np.expm1(1.))
            return tf.keras.Sequential([
                layers.VariableLayer(n, dtype=dtype),
                layers.DistributionLambda(lambda t: tfd.Independent(tfd.Normal(loc=t, scale=1.), reinterpreted_batch_ndims=1)), #[:n]#1e-5 + tf.math.softplus(c + t[n:])
                ])

        # define input layer
        Inputs = layer.Input(
            shape = (number_of_input_neurons,),
            name  = self.inputName)

        X = Inputs

        n_train_samples = 0.75 * self.data.get_train_data(as_matrix = True).shape[0] #1.0*self.architecture["batch_size"]

        # create i dense flipout layers with n neurons as specified in net_config
        for iLayer, nNeurons in enumerate(number_of_neurons_per_layer):
            X = tfp.layers.DenseFlipout(
            units                       =
            activation                  = activation_function, 
            activity_regularizer        = None, 
            trainable                   = True,
            kernel_posterior_fn         = tfp_layers_util.default_mean_field_normal_fn(),
            kernel_posterior_tensor_fn  = (lambda d: d.sample()),
            kernel_prior_fn             = tfp.layers.default_multivariate_normal_fn,
            kernel_divergence_fn        = (lambda q, p, ignore: kl_lib.kl_divergence(q, p)), 
            bias_posterior_fn           = tfp_layers_util.default_mean_field_normal_fn(is_singular=True),
            bias_posterior_tensor_fn    = (lambda d: d.sample()), 
            bias_prior_fn               = None,
            bias_divergence_fn          = (lambda q, p, ignore: kl_lib.kl_divergence(q, p)), 
            seed                        = None,
            name                        = "DenseFlipout_"+str(iLayer))
            
            tfp.layers.DenseVariational(
                units               = nNeurons,
                make_posterior_fn   = posterior,
                make_prior_fn       = prior,
                kl_weight           = 1. / n_train_samples,
                kl_use_exact        = False,
                use_bias            = self.use_bias,
                activation          = activation_function,
                name                = "DenseLayer_"+str(iLayer)
                )(X)

            # add dropout percentage to layer if activated
            if not dropout == 0:
                X = layer.Dropout(dropout, name = "DropoutLayer_"+str(iLayer))(X)

        # generate output layer
        X = DenseVariational(
            units               = self.data.n_output_neurons,
            make_posterior_fn   = posterior,
            make_prior_fn       = prior,
            kl_weight           = 1. / n_train_samples,
            kl_use_exact        = False,
            use_bias            = self.use_bias,
            activation          = output_activation.lower(),
            name                = self.outputName
            )(X)

        # define model
        model = models.Model(inputs = [Inputs], outputs = [X])
        model.summary()

        return model


model = tf.keras.Sequential([
    tfp.layers.DenseFlipout(512, activation=tf.nn.relu),
    tfp.layers.DenseFlipout(10),
])
logits = model(features)
neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
    labels=labels, logits=logits)
kl = sum(model.losses)
loss = neg_log_likelihood + kl
train_op = tf.train.AdamOptimizer().minimize(loss)