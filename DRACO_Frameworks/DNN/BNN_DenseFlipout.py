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
import datetime
import time


# local imports
filedir  = os.path.dirname(os.path.realpath(__file__))
DRACOdir = os.path.dirname(filedir)
basedir  = os.path.dirname(DRACOdir)
sys.path.append(basedir)

# import with ROOT
from pyrootsOfTheCaribbean.evaluationScripts import plottingScripts
from net_configs import config_dict

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
    
    def _load_architecture(self, config, restore_layer):
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
          "earlystopping_epochs":     None,
        } 

        for key in config:
            self.architecture[key] = config[key]
            if key == "layers" and (restore_layer is not None):
                self.architecture[key] = restore_layer


    def load_trained_model(self, inputDirectory, n_iterations=100):
        ''' load an already trained model '''
        #checkpoint_path = inputDirectory+"/checkpoints/trained_model.h5py"
        checkpoint_path = inputDirectory+"/checkpoints/trained_model_weights"
        netConfig_path = inputDirectory+"/checkpoints/net_config.json"


        # get the keras model
        #self.model = models.load_model(checkpoint_path, custom_objects={'tf':tf, 'tfp':tfp, 'tfd':tfd, 'DenseFlipout':DenseFlipout, 'neg_log_likelihood':self.neg_log_likelihood})
        with open(netConfig_path) as json_file:
            data = json.load(json_file)
            netConfigName = data["netConfig"]
            layer = data["layers"]

        config = config_dict[netConfigName]

        self.model = self.build_model(config = config, restore_layer=layer)
        load_status = self.model.load_weights(checkpoint_path)
        # Crosscheck whether weights are successfully loaded
        load_status.assert_existing_objects_matched()

        # evaluate test dataset with keras model
        start_eval = time.time()
        self.model_eval = self.model.evaluate(self.data.get_test_data(as_matrix = True), self.data.get_test_labels())
        end_eval = time.time()
        self.eval_duration = round(end_eval-start_eval)
        
        # save predictions
        start_pred = time.time()
        self.model_prediction_vector, self.model_prediction_vector_std, self.test_preds = self.bnn_calc_mean_std(n_samples=n_iterations)
        end_pred = time.time()
        self.pred_duration = round(end_pred-start_pred)
        #self.plot_event_output_distribution(save_dir=inputDirectory, preds=self.test_preds, n_events=len(self.test_preds), n_hist_bins=15)
        
        dict_eval_metrics = {}
        dict_eval_metrics["model_test_loss"] = self.model_eval[0]
        for im, metric in enumerate(self.eval_metrics):
            dict_eval_metrics["model_test_"+str(metric)] = self.model_eval[im+1]
        
        import collections
        dict_eval_metrics = collections.OrderedDict(sorted(dict_eval_metrics.items()))

        ''' save eval metrics to csv file'''
        filename = self.save_path.replace(self.save_path.split("/")[-1], "")+"eval_metrics.csv"
        file_exists = os.path.isfile(filename)
        with open(filename, "a+") as f:
            headers = np.concatenate((["project_name"], dict_eval_metrics.keys()))
            csv_writer = csv.DictWriter(f,delimiter=',', lineterminator='\n',fieldnames=headers)
            if not file_exists:
                csv_writer.writeheader()
            
            row = {"project_name": inputDirectory.split("workdir/")[-1]+"_loaded"}
            row.update(dict_eval_metrics)
            csv_writer.writerow(row)
            print("saved eval metrics to "+str(filename))

        ''' save eval duration loaded model to csv file'''
        filename = self.save_path.replace(self.save_path.split("/")[-1], "")+"eval_duration_loaded_model.csv"
        file_exists = os.path.isfile(filename)
        with open(filename, "a+") as f:
            headers = ["project_name", "eval_duration (hh:mm:ss)", "total_pred_duration (hh:mm:ss)", "mean_pred_duration (hh:mm:ss/npreds)"]
            csv_writer = csv.DictWriter(f,delimiter=',', lineterminator='\n',fieldnames=headers)
            if not file_exists:
                csv_writer.writeheader()
            csv_writer.writerow({"project_name": inputDirectory.split("workdir/")[-1]+"_loaded", "eval_duration (hh:mm:ss)": datetime.timedelta(seconds = self.eval_duration),
                                 "total_pred_duration (hh:mm:ss)": datetime.timedelta(seconds = self.pred_duration), "mean_pred_duration (hh:mm:ss/npreds)": datetime.timedelta(seconds = self.pred_duration/float(n_iterations))})
            print("saved eval duration loaded model to "+str(filename))

        # print evaluations with keras model
        from sklearn.metrics import roc_auc_score
        self.roc_auc_score = roc_auc_score(self.data.get_test_labels(), self.model_prediction_vector) #me
        print("\nROC-AUC score: {}".format(self.roc_auc_score))

        ''' save roc_auc_score to csv file'''
        filename = self.save_path.replace(self.save_path.split("/")[-1], "")+"roc_auc_score.csv"
        file_exists = os.path.isfile(filename)
        with open(filename, "a+") as f:
            headers = ["project_name", "roc_auc_score"]
            csv_writer = csv.DictWriter(f,delimiter=',', lineterminator='\n',fieldnames=headers)
            if not file_exists:
                csv_writer.writeheader()
            csv_writer.writerow({"project_name": inputDirectory.split("workdir/")[-1]+"_loaded", "roc_auc_score": self.roc_auc_score})
            print("saved roc_auc_score to "+str(filename))
            
        return self.model_prediction_vector, self.model_prediction_vector_std, self.data.get_test_labels()

    # sampling output values from the intern tensorflow output distribution
    def bnn_calc_mean_std(self, n_samples=100):
        test_pred  = []
        print "Calculating the mean and std: "
        for i in tqdm.tqdm(range(n_samples)):
            test_pred_vector = self.model.predict(self.data.get_test_data(as_matrix = True))
            test_pred.append(test_pred_vector)
            test_preds = np.concatenate(test_pred, axis=1)
        return np.mean(test_preds, axis=1), np.std(test_preds, axis=1), test_preds

    def build_default_model(self):
        ''' build default straight forward BNN from architecture dictionary '''
        '''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ATTENTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: 
            Do not forget to save the changes on the arguments for reloading the trained network at a later time
            since only the trained weights are saved in the end but NOT the architecture!!'''

        # infer number of input neurons from number of train variables
        number_of_input_neurons     = self.data.n_input_neurons

        n_train_samples = 0.75*self.data.get_train_data(as_matrix = True).shape[0] 

        # get all the architecture settings needed to build model
        number_of_neurons_per_layer = self.architecture["layers"]
        dropout                     = self.architecture["Dropout"]
        activation_function         = self.architecture["activation_function"]
        output_activation           = self.architecture["output_activation"]
        activity_regularizer        = self.architecture["activity_regularizer"]
        trainable                   = self.architecture["trainable"]
        kernel_posterior_fn         = self.architecture["kernel_posterior_fn"]
        kernel_prior_fn             = self.architecture["kernel_prior_fn"]
        bias_posterior_fn           = self.architecture["bias_posterior_fn"]
        bias_prior_fn               = self.architecture["bias_prior_fn"]
        seed                        = self.architecture["seed"]

        # define input layer
        Inputs = layer.Input(
            shape = (number_of_input_neurons,),
            name  = self.inputName)

        X = Inputs


        # create i dense flipout layers with n neurons as specified in net_config
        for iLayer, nNeurons in enumerate(number_of_neurons_per_layer):
            X = tfp.layers.DenseFlipout(
            units                       = nNeurons,
            activation                  = activation_function, 
            activity_regularizer        = activity_regularizer, 
            trainable                   = trainable,
            kernel_posterior_fn         = kernel_posterior_fn,
            kernel_posterior_tensor_fn  = lambda d: d.sample(),
            kernel_prior_fn             = kernel_prior_fn,
            kernel_divergence_fn        = lambda q, p, ignore: tfd.kl_divergence(q, p)/tf.to_float(n_train_samples),
            bias_posterior_fn           = bias_posterior_fn,
            bias_posterior_tensor_fn    = lambda d: d.sample(),
            bias_prior_fn               = bias_prior_fn,
            bias_divergence_fn          = lambda q, p, ignore: tfd.kl_divergence(q, p)/tf.to_float(n_train_samples),
            seed                        = seed, 
            name                        = "DenseFlipout_"+str(iLayer))(X)

            

            # add dropout percentage to layer if activated
            if not dropout == 0:
                X = layer.Dropout(dropout, name = "DropoutLayer_"+str(iLayer))(X)

        # generate output layer
        X = tfp.layers.DenseFlipout(
            units                       = self.data.n_output_neurons,
            activation                  = output_activation.lower(), 
            activity_regularizer        = activity_regularizer, 
            trainable                   = trainable,
            kernel_posterior_fn         = kernel_posterior_fn,
            kernel_posterior_tensor_fn  = lambda d: d.sample(),
            kernel_prior_fn             = kernel_prior_fn,
            kernel_divergence_fn        = lambda q, p, ignore: tfd.kl_divergence(q, p)/tf.to_float(n_train_samples),
            bias_posterior_fn           = bias_posterior_fn,
            bias_posterior_tensor_fn    = lambda d: d.sample(),
            bias_prior_fn               = bias_prior_fn,
            bias_divergence_fn          = lambda q, p, ignore: tfd.kl_divergence(q, p)/tf.to_float(n_train_samples),
            seed                        = seed, 
            name                        = self.outputName)(X)

        # define model
        model = models.Model(inputs = [Inputs], outputs = [X])
        model.summary()

        return model

    # custom loss definition
    def neg_log_likelihood(self, y_true, y_pred):
        sigma = 1.
        dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
        return -dist.log_prob(y_true) #tf.reduce_mean(dist.log_prob(y_true), axis=-1)

    def wrapped_partial(self, func, *args, **kwargs):
        partial_func = partial(func, *args, **kwargs)
        update_wrapper(partial_func, func)
        return partial_func

    def build_model(self, config = None, model = None, restore_layer = None):
        ''' build a BNN model
            use options defined in 'config' dictionary '''

        if config:
            self._load_architecture(config, restore_layer)
            print("loading non default net configs")

        if model == None:
            print("building model from config")
            model = self.build_default_model()

        # compile the model
        model.compile(
            loss        = self.wrapped_partial(self.neg_log_likelihood), 
            optimizer   = self.architecture["optimizer"],
            metrics     = self.eval_metrics+[self.wrapped_partial(self.neg_log_likelihood)]) 

        # save the model
        self.model = model

        # save net information
        out_file    = self.save_path+"/model_summary.yml"
        yml_model   = self.model.to_yaml()
        with open(out_file, "w") as f:
            f.write(yml_model)
        
        return self.model #me
    

    def train_model(self):
        ''' train the model '''

        # add early stopping if activated
        callbacks = None
        if self.architecture["earlystopping_percentage"] or self.architecture["earlystopping_epochs"]:
            callbacks = [EarlyStopping(
                monitor         = "loss",
                value           = self.architecture["earlystopping_percentage"],
                min_epochs      = 50,
                stopping_epochs = self.architecture["earlystopping_epochs"],
                verbose         = 1)]

        # train main net
        start_train = time.time()
        self.trained_model = self.model.fit(
            x = self.data.get_train_data(as_matrix = True),
            y = self.data.get_train_labels(),
            batch_size          = self.architecture["batch_size"],
            epochs              = self.train_epochs,
            shuffle             = True,
            callbacks           = callbacks,
            validation_split    = 0.25,
            sample_weight       = self.data.get_train_weights(),
            )
        end_train = time.time()
        self.train_duration = round(end_train - start_train)
    
    #debug
    def eval_model(self, iterations=5):
        # evaluate test dataset
        start_eval = time.time()
        self.model_eval = self.model.evaluate(
            self.data.get_test_data(as_matrix = True),
            self.data.get_test_labels())
        end_eval = time.time()
        self.eval_duration = round(end_eval - start_eval)

        # save history of eval metrics
        self.model_history = self.trained_model.history

        # save predicitons
        start_pred = time.time()
        self.model_prediction_vector, self.model_prediction_vector_std, self.test_preds = self.bnn_calc_mean_std(n_samples=iterations)
        end_pred = time.time()
        self.pred_duration = round(end_pred - start_pred)

        # print evaluations
        from sklearn.metrics import roc_auc_score
        self.roc_auc_score = roc_auc_score(self.data.get_test_labels(), self.model_prediction_vector) 
        
        ''' save duration to csv file'''
        filename = self.save_path.replace(self.save_path.split("/")[-1], "")+"duration.csv"
        file_exists = os.path.isfile(filename)
        with open(filename, "a+") as f:
            headers = ["project_name", "train_duration (hh:mm:ss)", "eval_duration (hh:mm:ss)", "total_pred_duration (hh:mm:ss)", "mean_pred_duration (hh:mm:ss/npreds)"]
            csv_writer = csv.DictWriter(f,delimiter=',', lineterminator='\n',fieldnames=headers)
            if not file_exists:
                csv_writer.writeheader()
            csv_writer.writerow({"project_name": self.save_path.split("/")[-1], "train_duration (hh:mm:ss)": datetime.timedelta(seconds = self.train_duration), "eval_duration (hh:mm:ss)": datetime.timedelta(seconds =self.eval_duration), 
                                 "total_pred_duration (hh:mm:ss)": datetime.timedelta(seconds = self.pred_duration), "mean_pred_duration (hh:mm:ss/npreds)": datetime.timedelta(seconds = self.pred_duration/float(iterations))})
            print("saved duration to "+str(filename))
    
        ''' save roc_auc_score to csv file'''
        filename = self.save_path.replace(self.save_path.split("/")[-1], "")+"roc_auc_score.csv"
        file_exists = os.path.isfile(filename)
        with open(filename, "a+") as f:
            headers = ["project_name", "roc_auc_score"]
            csv_writer = csv.DictWriter(f,delimiter=',', lineterminator='\n',fieldnames=headers)
            if not file_exists:
                csv_writer.writeheader()
            csv_writer.writerow({"project_name": self.save_path.split("/")[-1], "roc_auc_score": self.roc_auc_score})
            print("saved roc_auc_score to "+str(filename))

        print("\nROC-AUC score: {}".format(self.roc_auc_score))

        if self.eval_metrics:
            dict_eval_metrics = {}
            print("model test loss: {}".format(self.model_eval[0]))
            dict_eval_metrics["model_test_loss"] = self.model_eval[0]

            for im, metric in enumerate(self.eval_metrics):
                print("model test {}: {}".format(metric, self.model_eval[im+1]))
                dict_eval_metrics["model_test_"+str(metric)] = self.model_eval[im+1]
            
            import collections
            dict_eval_metrics = collections.OrderedDict(sorted(dict_eval_metrics.items()))

            ''' save eval metrics to csv file'''
            filename = self.save_path.replace(self.save_path.split("/")[-1], "")+"eval_metrics.csv"
            file_exists = os.path.isfile(filename)
            with open(filename, "a+") as f:
                headers = np.concatenate((["project_name"], dict_eval_metrics.keys()))
                csv_writer = csv.DictWriter(f,delimiter=',', lineterminator='\n',fieldnames=headers)
                if not file_exists:
                    csv_writer.writeheader()
                
                row = {"project_name": self.save_path.split("/")[-1]}
                row.update(dict_eval_metrics)
                csv_writer.writerow(row)
                print("saved eval metrics to "+str(filename))

        # return self.model_prediction_vector, self.model_prediction_vector_std
    
    def save_model(self, argv, execute_dir, netConfigName):
        ''' save the trained model '''

        # save executed command
        argv[0] = execute_dir+"/"+argv[0].split("/")[-1]
        execute_string = "python "+" ".join(argv)
        out_file = self.cp_path+"/command.sh"
        with open(out_file, "w") as f:
            f.write(execute_string)
        print("saved executed command to {}".format(out_file))

        # save model as h5py file
        out_file = self.cp_path + "/trained_model.h5py"
        self.model.save(out_file, save_format='h5')
        print("saved trained model at "+str(out_file))

        # save config of model
        model_config = self.model.get_config()
        out_file = self.cp_path +"/trained_model_config"
        with open(out_file, "w") as f:
            f.write( str(model_config))
        print("saved model config at "+str(out_file))

        # save weights of network
        #out_file = self.cp_path +"/trained_model_weights.h5"
        #self.model.save_weights(out_file)

        out_file = self.cp_path +"trained_model_weights" #me removed slash before trained
        self.model.save_weights(out_file)

        #Crosscheck whether save of weights was successfull
        load_status = self.model.load_weights(out_file)
        if load_status is None:
            raise TypeError("Weights not saved successfully.")

        print("wrote trained weights to "+str(out_file))

        # set model as non trainable
        for layer in self.model.layers:
            layer.trainable = False
        self.model.trainable = False

        self.netConfig = netConfigName

        # save checkpoint files (needed for c++ implementation)
        out_file = self.cp_path + "/trained_model"
        saver = tf.compat.v1.train.Saver()
        sess = tf.compat.v1.keras.backend.get_session()
        save_path = saver.save(sess, out_file)
        print("saved checkpoint files to "+str(out_file))

        # produce json file with configs
        configs = self.architecture
        configs["inputName"] = self.inputName
        configs["outputName"] = self.outputName+"/"+configs["output_activation"]
        configs = {key: configs[key] for key in configs if key not in ["optimizer", "kernel_posterior_fn", "kernel_prior_fn", "bias_posterior_fn","bias_prior_fn"]}

        # more information saving
        configs["inputData"] = self.input_samples.input_path
        configs["eventClasses"] = self.input_samples.getClassConfig()
        configs["JetTagCategory"] = self.category_name
        configs["categoryLabel"] = self.category_label
        configs["Selection"] = self.category_cutString
        configs["trainEpochs"] = self.train_epochs
        configs["trainVariables"] = self.train_variables
        configs["shuffleSeed"] = self.data.shuffleSeed
        configs["trainSelection"] = self.evenSel
        configs["evalSelection"] = self.oddSel
        configs["netConfig"] = self.netConfig
        configs["bestEpoch"] = len(self.model_history["acc"])
        configs["restoreFitDir"] = self.data.get_fit_dir()

        # save information for binary DNN
        if self.data.binary_classification:
            configs["binaryConfig"] = {
              "minValue": self.input_samples.bkg_target,
              "maxValue": 1.,
            }

        json_file = self.cp_path + "/net_config.json"
        with open(json_file, "w") as jf:
            json.dump(configs, jf, indent = 2, separators = (",", ": "))
        print("wrote net configs to "+str(json_file))

        '''  save configurations of variables for plotscript '''
        plot_file = self.cp_path+"/plot_config.csv"
        variable_configs = pd.read_csv(basedir+"/pyrootsOfTheCaribbean/plot_configs/variableConfig.csv").set_index("variablename", drop = True)
        variables = variable_configs.loc[self.train_variables]
        variables.to_csv(plot_file, sep = ",")
        print("wrote config of input variables to {}".format(plot_file))

        ''' save best epoch to csv file'''
        filename = self.save_path.replace(self.save_path.split("/")[-1], "")+"best_epoch.csv"
        file_exists = os.path.isfile(filename)
        with open(filename, "a+") as f:
            headers = ["project_name", "best_epoch"]
            csv_writer = csv.DictWriter(f,delimiter=',', lineterminator='\n',fieldnames=headers)
            if not file_exists:
                csv_writer.writeheader()
            csv_writer.writerow({"project_name": self.save_path.split("/")[-1], "best_epoch": len(self.model_history["acc"])})
            print("saved best_epoch to "+str(filename))


        # Serialize the test inputs for the analysis of the gradients
        ##pickle.dump(self.data.get_test_data(), open(self.cp_path+"/inputvariables.pickle", "wb")) #me as it needs much space
    
    def get_input_weights(self):
        ''' get the weights of the input layer and sort input variables by weight sum '''

        sess = tf.compat.v1.keras.backend.get_session()

        # get weights
        first_layer = self.model.layers[1]
        
        weights_mean_kernel_posterior = first_layer.kernel_posterior.mean().eval(session=sess)
        std_kernel_posterior = first_layer.kernel_posterior.stddev().eval(session=sess)
        
        #weights_mean_bias_posterior = first_layer.bias_posterior.mean().eval(session=sess)
        #std_bias_posterior = first_layer.bias_posterior.stddev().eval(session=sess) #is zero if in Denseflipout bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(is_singular=True) chosen
        
        #weights_mean_kernel_prior = first_layer.kernel_prior.mean().eval(session=sess)
        #std_kernel_prior = first_layer.kernel_prior.stddev().eval(session=sess)


        # #alternative
        # weights = first_layer.get_weights()
        # weights_mean_kernel_posterior = weights[0]
        # std_kernel_posterior = np.log(np.exp(weights[1])+1) #softplus transformation or tf.nn.softplus(weights[1]).eval(session=sess)
        # weights_mean_bias_posterior = weights[2]
        # std_bias_posterior =  np.log(np.exp(weights[3])+1) #softplus transformation or tf.nn.softplus(weights[3]).eval(session=see) #not available if in Denseflipout bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(is_singular=True) chosen because std is zero then

        #Range of first layer weights:
        # weights = self.model.layers[1].get_weights() 
        # print "Range of first layer weights:"
        # print "mean kernel posterior:" + str(np.amin(weights[0])) + " to "  + str(np.amax(weights[0]))
        # print "std kernel posterior:" + str(np.amin(weights[1])) + " to "  + str(np.amax(weights[1]))
        # print "mean bias posterior:" + str(np.amin(weights[2])) + " to "  + str(np.amax(weights[2]))

        # print "**************************************************************"
        # weights = self.model.layers[2].get_weights() 
        # print "Range of second layer weights:"
        # print "mean kernel posterior:" + str(np.amin(weights[0])) + " to "  + str(np.amax(weights[0]))
        # print "std kernel posterior:" + str(np.amin(weights[1])) + " to "  + str(np.amax(weights[1]))
        # print "mean bias posterior:" + str(np.amin(weights[2])) + " to "  + str(np.amax(weights[2]))

        weights_mean = np.split(weights_mean_kernel_posterior, len(self.train_variables))
        weights_std  = np.split(std_kernel_posterior, len(self.train_variables))


        self.weight_dict = {}
        for out_weights_mean, out_weights_std, variable in zip(weights_mean, weights_std, self.train_variables):
            w_mean_sum = np.sum(np.abs(out_weights_mean))
            w_std_sum = np.sqrt(np.sum(np.power(out_weights_std,2)))
            self.weight_dict[variable] = (w_mean_sum, w_std_sum)

        # sort weight dict
        rank_path = self.save_path + "/first_layer_weights.csv"
        with open(rank_path, "w") as f:
            f.write("variable,weight_mean_sum,weight_std_sum\n")
            for key, val in sorted(self.weight_dict.iteritems(), key = lambda (k,v): (v,k)):
                f.write("{},{},{}\n".format(key,val[0],val[1]))
        print("wrote weight ranking to "+str(rank_path))

    def get_model(self):
        return self.model

    # result plotting functions
    # --------------------------------------------------------------------
    def plot_binaryOutput(self, log = False, privateWork = False, printROC = False,
                        nbins = None, bin_range = [0.,1.], name = "binary_discriminator",
                        sigScale = -1):

        if not nbins:
            nbins = int(50*(1.-bin_range[0]))

        binaryOutput = plottingScripts.plotBinaryOutput(
            data                = self.data,
            test_predictions    = self.model_prediction_vector,
            train_predictions   = None,#self.model_train_prediction,
            nbins               = nbins,
            bin_range           = bin_range,
            event_category      = self.category_label,
            plotdir             = self.save_path,
            logscale            = log,
            sigScale            = sigScale)

        bkg_hist, sig_hist = binaryOutput.plot(ratio = False, printROC = printROC, privateWork = privateWork, name = name)

        binaryOutput_std = plottingScripts.plotBinaryOutput(
            data                = self.data,
            test_predictions    = self.model_prediction_vector_std,
            train_predictions   = None, # self.model_train_prediction_std,
            nbins               = 20, #me
            bin_range           = [0.,np.amax(self.model_prediction_vector_std)],
            event_category      = self.category_label,
            plotdir             = self.save_path,
            logscale            = log,
            sigScale            = sigScale,
            save_name           = "sigma_discriminator"
            )

        bkg_std_hist, sig_std_hist = binaryOutput_std.plot(ratio = False, printROC = printROC, privateWork = privateWork, name = "BNN Standardabweichung #sigma")


        self.plot_2D_hist_std_over_mean(privateWork = privateWork, bin_range=[50,50])
        self.plot_varied_histogram(privateWork = privateWork)

    def plot_2D_hist_std_over_mean(self, privateWork=False, bin_range=[40,40]):
        from matplotlib.colors import LogNorm
        plt.rc('xtick',labelsize=14)
        plt.rc('ytick',labelsize=14)
        plt.hist2d(self.model_prediction_vector, self.model_prediction_vector_std, bins=bin_range, cmin=1, norm=LogNorm())
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.xlabel("BNN Ausgabewert $\mu$", fontsize = 16)
        plt.ylabel("BNN Standardabweichung $\sigma$", fontsize = 16)
        if privateWork:
            plt.title(r"\textbf{CMS private work}", loc = "left", fontsize = 14)
        plt.tight_layout()
        plt.savefig(self.save_path+"/sigma_over_mu.png")
        print "sigma_over_mu.png was created"
        plt.savefig(self.save_path+"/sigma_over_mu.pdf")
        print "sigma_over_mu.pdf was created"
        plt.close()

    def plot_varied_histogram(self, privateWork=False):
        sig_preds, sig_preds_std, bkg_preds, bkg_preds_std = [], [], [], []
        for i in range(len(self.data.get_test_labels())):
            if self.data.get_test_labels()[i]==1:
                sig_preds.append(self.model_prediction_vector[i])
                sig_preds_std.append(self.model_prediction_vector_std[i])
            elif self.data.get_test_labels()[i]==0:
                bkg_preds.append(self.model_prediction_vector[i])
                bkg_preds_std.append(self.model_prediction_vector_std[i])
            else:
                print "--wrong event--"

        plt.rc('xtick',labelsize=14)
        plt.rc('ytick',labelsize=14)
        plt.hist(sig_preds, bins=20, range=(0,1), histtype='step', density=True, label="ttH", color="b") #me
        plt.hist(np.array(sig_preds)+1.*np.array(sig_preds_std), bins=20, range=(0,1), histtype='step', density=True, linestyle=('--'), color="b") #me
        plt.hist(np.array(sig_preds)-1.*np.array(sig_preds_std), bins=20, range=(0,1), histtype='step', density=True, linestyle=(':'), color="b") #me
        plt.hist(bkg_preds, bins=20, range=(0,1), histtype='step', density=True, label="bkg", color="r") #me
        plt.hist(np.array(bkg_preds)+1.*np.array(bkg_preds_std), bins=20, range=(0,1), histtype='step', density=True, linestyle=('--'), color="r") #me
        plt.hist(np.array(bkg_preds)-1.*np.array(bkg_preds_std), bins=20, range=(0,1), histtype='step', density=True, linestyle=(':'), color="r") #me
        plt.xlabel("BNN Ausgabewert $\mu$", fontsize = 16)
        plt.legend(fontsize=14)
        if privateWork:
            plt.title(r"\textbf{CMS private work}", loc = "left", fontsize = 14)
        plt.tight_layout()

        plt.savefig(self.save_path+"/varied_discr.png")
        print "varied_discr.png was created"
        plt.savefig(self.save_path+"/varied_discr.pdf")
        print "varied_discr.pdf was created"
        plt.close()

    def plot_metrics(self, privateWork = False):
        plt.rc('text', usetex=True)

        ''' plot history of loss function and evaluation metrics '''
        metrics = ["loss", "neg_log_likelihood"]
        if self.eval_metrics: metrics += self.eval_metrics

        # loop over metrics and generate matplotlib plot
        for metric in metrics:
            plt.clf()

            # get history of train and validation scores
            train_history = self.model_history[metric]
            val_history = self.model_history["val_"+metric]

            n_epochs = len(train_history)
            epochs = np.arange(1,n_epochs+1,1)

            # plot histories
            plt.plot(epochs, train_history, "b-", label = "Training", lw = 2)
            plt.plot(epochs, val_history, "r-", label = "Validierung", lw = 2)
            if privateWork:
                plt.title(r"\textbf{CMS private work}", loc = "left", fontsize = 14)
            else:
                plt.title("Beste Epoche: "+str(n_epochs), loc="left", fontsize = 16)



            # add title
            title = self.category_label
            title = title.replace("\\geq", "$\geq$")
            title = title.replace("\\leq", "$\leq$")
            plt.title(title, loc = "right", fontsize = 16)

            # make it nicer
            german = {"acc": "Genauigkeit", "loss": "Verlust", "neg log likelihood": "negative Log-Likelihood", "KLD": "KL"}
            #plt.grid()
            plt.xlabel("Epoche", fontsize = 16)
            plt.ylabel(german[metric.replace("_"," ")], fontsize = 16)
            #plt.ylim(ymin=0.)

            # add legend
            plt.legend(fontsize=14)
            plt.tight_layout()

            # save
            out_path = self.save_path + "/model_history_"+str(metric)+".pdf"
            plt.savefig(out_path)
            print("saved plot of "+str(metric)+" at "+str(out_path))
            plt.close()

        train_history_KLD = np.array(self.model_history["loss"]) - np.array(self.model_history["neg_log_likelihood"])
        val_history_KLD = np.array(self.model_history["val_loss"]) - np.array(self.model_history["val_neg_log_likelihood"])

        plt.clf()

        n_epochs = len(train_history_KLD)
        epochs = np.arange(1,n_epochs+1,1)

        # plot histories
        plt.rc('xtick',labelsize=16)
        plt.rc('ytick',labelsize=16)
        plt.plot(epochs, train_history_KLD, "b-", label = "Training", lw = 2)
        plt.plot(epochs, val_history_KLD, "r-", label = "Validierung", lw = 2)
        if privateWork:
            plt.title(r"\textbf{CMS private work}", loc = "left", fontsize = 14)

        # add title
        title = self.category_label
        title = title.replace("\\geq", "$\geq$")
        title = title.replace("\\leq", "$\leq$")
        plt.title(title, loc = "right", fontsize = 16)

        # make it nicer
        #plt.grid()
        plt.xlabel("Epoche", fontsize = 16)
        plt.ylabel("KL", fontsize = 16)
        #plt.ylim(ymin=0.)

        # add legend
        plt.legend(fontsize=14)
        plt.tight_layout()

        # save
        out_path = self.save_path + "/model_history_"+"KLD"+".pdf"
        plt.savefig(out_path)
        print("saved plot of "+"KLD"+" at "+str(out_path))
        plt.close()

    #copied from DNN.py in dev-bnn branch
    def plot_confusionMatrix(self, norm_matrix = True, privateWork = False, printROC = False):
        ''' plot confusion matrix '''
        plotCM = plottingScripts.plotConfusionMatrix(
            data                = self.data,
            prediction_vector   = self.model_prediction_vector,
            event_classes       = self.event_classes,
            event_category      = self.category_label,
            plotdir             = self.save_path)

        plotCM.plot(norm_matrix = norm_matrix, privateWork = privateWork, printROC = printROC)

    def plot_discriminators(self, log = False, printROC = False, privateWork = False,
                            signal_class = None, nbins = None, bin_range = None,
                            sigScale = -1):

        ''' plot all events classified as one category '''
        if not bin_range:
            bin_range = [round(1./self.data.n_output_neurons,2), 1.]
        if not nbins:
            nbins = int(25*(1.-bin_range[0]))

        plotDiscrs = plottingScripts.plotDiscriminators(
            data                = self.data,
            prediction_vector   = self.model_prediction_vector,
            event_classes       = self.event_classes,
            nbins               = nbins,
            bin_range           = bin_range,
            signal_class        = signal_class,
            event_category      = self.category_label,
            plotdir             = self.plot_path,
            logscale            = log,
            sigScale            = sigScale)

        bkg_hist, sig_hist = plotDiscrs.plot(ratio = False, printROC = printROC, privateWork = privateWork)
        #print("ASIMOV: mu=0: sigma (-+): ", self.binned_likelihood(bkg_hist, sig_hist, 0))
        #print("ASIMOV: mu=1: sigma (-+): ", self.binned_likelihood(bkg_hist, sig_hist, 1))

    def plot_outputNodes(self, log = False, printROC = False, signal_class = None,
                            privateWork = False, nbins = 30, bin_range = [0.,1.],
                            sigScale = -1):

        ''' plot distribution in outputNodes '''
        plotNodes = plottingScripts.plotOutputNodes(
            data                = self.data,
            prediction_vector   = self.model_prediction_vector,
            event_classes       = self.event_classes,
            nbins               = nbins,
            bin_range           = bin_range,
            signal_class        = signal_class,
            event_category      = self.category_label,
            plotdir             = self.plot_path,
            logscale            = log,
            sigScale            = sigScale)

        plotNodes.plot(ratio = False, printROC = printROC, privateWork = privateWork)

    def plot_eventYields(self, log = False, privateWork = False, signal_class = None, sigScale = -1):
        eventYields = plottingScripts.plotEventYields(
            data                = self.data,
            prediction_vector   = self.model_prediction_vector,
            event_classes       = self.event_classes,
            event_category      = self.category_label,
            signal_class        = signal_class,
            plotdir             = self.save_path,
            logscale            = log)

        eventYields.plot(privateWork = privateWork)

    def plot_closureTest(self, log = False, privateWork = False,
                            signal_class = None, nbins = None, bin_range = None):
        ''' plot comparison between train and test samples '''

        if not bin_range:
            bin_range = [round(1./self.data.n_output_neurons,2), 1.]
        if not nbins:
            nbins = int(20*(1.-bin_range[0]))

        closureTest = plottingScripts.plotClosureTest(
            data                = self.data,
            test_prediction     = self.model_prediction_vector,
            train_prediction    = self.model_train_prediction,
            event_classes       = self.event_classes,
            nbins               = nbins,
            bin_range           = bin_range,
            signal_class        = signal_class,
            event_category      = self.category_label,
            plotdir             = self.plot_path,
            logscale            = log)

        closureTest.plot(ratio = False, privateWork = privateWork)