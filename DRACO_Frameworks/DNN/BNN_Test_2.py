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
from BNN_layer import DenseVariational

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


class BNN():
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
          "earlystopping_epochs":     None,
        }

        for key in config:
            self.architecture[key] = config[key]
        
    def load_trained_model(self, inputDirectory, n_iterations=200):
        ''' load an already trained model '''
        checkpoint_path = inputDirectory+"/checkpoints/trained_model.h5py"

        # get the keras model
        self.model = models.load_model(checkpoint_path, custom_objects={'tf':tf, 'tfp':tfp, 'tfd':tfd, 'DenseVariational':DenseVariational, 'neg_log_likelihood':self.neg_log_likelihood})
        self.model.summary()

        # evaluate test dataset with keras model
        self.model_eval = self.model.evaluate(self.data.get_test_data(as_matrix = True), self.data.get_test_labels())

        # save predictions
        self.model_prediction_vector, self.model_prediction_vector_std, self.test_preds = self.bnn_calc_mean_std(n_samples=n_iterations)
        #self.plot_event_output_distribution(save_dir=inputDirectory, preds=self.test_preds, n_events=len(self.test_preds), n_hist_bins=15)
        #DEBUG
        print self.model_prediction_vector
        
        # print evaluations  with keras model
        from sklearn.metrics import roc_auc_score
        self.roc_auc_score = roc_auc_score(self.data.get_test_labels(), self.model_prediction_vector)
        print("\nROC-AUC score: {}".format(self.roc_auc_score))

        return self.model_prediction_vector, self.model_prediction_vector_std, self.data.get_test_labels()

    # make plots of the output distribution for one single event
    def plot_event_output_distribution(self, save_dir, preds, n_events=20, n_hist_bins=20):
        #if len(preds) < n_events: return "not enouth events to draw output distribution"
        for i in tqdm.tqdm(range(n_events)):
            if self.model_prediction_vector_std[i] > 0.15:
                n, bins, patches = plt.hist(preds[i], n_hist_bins, facecolor='g', alpha=0.75)
                plt.xlabel("$\mu$ of event", fontsize = 16)
                plt.ylabel("number of samples", fontsize = 16)
                plt.savefig(save_dir+"/events/event_{}_hist.pdf".format(i))
                plt.close()
                event_path = save_dir + "/events/event_{}_vars.csv".format(i)
                with open(event_path, "w") as f:
                    f.write("variable,value,normed value\n")
                    for k in range(len(self.train_variables)):
                        f.write("{},{},{}\n".format(self.train_variables[k], self.data.get_test_data(normed=False).iloc[i,k], self.data.get_test_data(as_matrix = False).iloc[i,k]))


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
            return tf.keras.Sequential([
                layers.VariableLayer(n, dtype=dtype), #trainable = False
                layers.DistributionLambda(lambda t: tfd.Independent(tfd.Normal(loc=t, scale=1.), reinterpreted_batch_ndims=1)), #[:n]#1e-5 + tf.math.softplus(c + t[n:]) #DEBUG
                ])

        # define input layer
        Inputs = layer.Input(
            shape = (number_of_input_neurons,),
            name  = self.inputName)

        X = Inputs
        self.layer_list = [X]

        n_train_samples = 0.75 * self.data.get_train_data(as_matrix = True).shape[0] #1.0*self.architecture["batch_size"]
        self.use_bias = True

        # loop over dense layers
        for iLayer, nNeurons in enumerate(number_of_neurons_per_layer):
            X = DenseVariational(
                units               = nNeurons,
                make_posterior_fn   = posterior,
                make_prior_fn       = prior,
                kl_weight           = 1. / n_train_samples,
                kl_use_exact        = False, #Debug
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
            kl_use_exact        = False, #Debug
            use_bias            = self.use_bias,
            activation          = output_activation.lower(),
            name                = self.outputName
            )(X)

        # define model
        model = models.Model(inputs = [Inputs], outputs = [X])
        model.summary()

        return model

    # custom loss definition
    def neg_log_likelihood(self, y_true, y_pred):
        sigma = 1.
        dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
        return -dist.log_prob(y_true)#tf.reduce_mean(dist.log_prob(y_true), axis=-1)

    def wrapped_partial(self, func, *args, **kwargs):
        partial_func = partial(func, *args, **kwargs)
        update_wrapper(partial_func, func)
        return partial_func

    def build_model(self, config = None, model = None):
        ''' build a BNN model
            use options defined in 'config' dictionary '''

        if config:
            self._load_architecture(config)
            print("loading non default net configs")

        if model == None:
            print("building model from config")
            model = self.build_default_model()

        # compile the model
        model.compile(
            loss        = self.neg_log_likelihood,
            optimizer   = self.architecture["optimizer"],
            metrics     = self.eval_metrics+[self.wrapped_partial(self.neg_log_likelihood)])

        # save the model
        self.model = model

        #DEBUG
        self.get_input_weights() #DEBUG


        # save net information
        out_file    = self.save_path+"/model_summary.yml"
        yml_model   = self.model.to_yaml()
        with open(out_file, "w") as f:
            f.write(yml_model)

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

        #DEBUG TODO
        first_layer = True
        for num_layer in range(len(self.architecture["layers"])):
            if first_layer:
                num_neurons_previous_layer = self.data.n_input_neurons
                first_layer = False
            else: 
                num_neurons_previous_layer = self.architecture["layers"][num_layer-1]

            if self.use_bias:
                num_param = (num_neurons_previous_layer + 1) * self.architecture["layers"][num_layer]
            else:
                num_param = num_neurons_previous_layer * self.architecture["layers"][num_layer]


            initialize_posterior_weights = np.random.uniform(low=-1., high=1., size=(1,num_param))
            #initialize_posterior_std = np.random.uniform(low=-4, high=-2, size=(1,num_param))
            #test not changing initialization of std posterior
            initialize_posterior_std = self.model.layers[num_layer+1].get_weights()[0][num_param:]
            initialize_combined = np.append(initialize_posterior_weights, initialize_posterior_std)
            #Since Prior is set to untrainable in V5 --> initial weights not changed --> reuse self.model.layers[num_layer+1].get_weights()[1]
            initialize = [initialize_combined, self.model.layers[num_layer+1].get_weights()[1]]  #num_layer+1  science in self.model.layers for inputlayer is counted as 0
            self.model.layers[num_layer+1].set_weights(initialize)  

        #Initialize Output Layer (OL)
        if self.use_bias:
            num_param = (self.architecture["layers"][num_layer]+1) * self.data.n_output_neurons
        else:
            num_param = self.architecture["layers"][num_layer] * self.data.n_output_neurons

        initialize_OL_posterior_weights = np.random.uniform(low=-0.2, high=0.2, size=(1,num_param))
        initialize_OL_posterior_std = np.random.uniform(low=-4, high=-2, size=(1,num_param))
        initialize_OL_combined = np.append(initialize_OL_posterior_weights, initialize_OL_posterior_std)
        initialize_OL = [initialize_OL_combined, self.model.layers[num_layer+2].get_weights()[1]]
        self.model.layers[num_layer+2].set_weights(initialize_OL) 

        # train main net
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
        
        self.get_input_weights() #DEBUG


    def eval_model(self):

        # evaluate test dataset
        self.model_eval = self.model.evaluate(
            self.data.get_test_data(as_matrix = True),
            self.data.get_test_labels())

        # save history of eval metrics
        self.model_history = self.trained_model.history

        # save predicitons
        self.model_prediction_vector, self.model_prediction_vector_std, self.test_preds = self.bnn_calc_mean_std(n_samples=50) #TODO DEBUG 2 isntead of 50

        # print evaluations
        from sklearn.metrics import roc_auc_score
        self.roc_auc_score = roc_auc_score(self.data.get_test_labels(), self.model_prediction_vector) 

        ''' save roc_auc_score to csv file'''
        filename = self.save_path.replace(self.save_path.split("/")[-1], "")+"/roc_auc_score.csv"
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
            print("model test loss: {}".format(self.model_eval[0]))
            for im, metric in enumerate(self.eval_metrics):
                print("model test {}: {}".format(metric, self.model_eval[im+1]))

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
        out_file = self.cp_path +"/trained_model_weights.h5"
        self.model.save_weights(out_file)
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
        configs = {key: configs[key] for key in configs if not "optimizer" in key}

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
        filename = self.save_path.replace(self.save_path.split("/")[-1], "")+"/best_epoch.csv"
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

         # get weights
        first_layer = self.model.layers[1]
        weights = first_layer.get_weights()[0]

        #if self.use_bias: #DEBUG
        if True:
            weights_mean = np.split(weights[:len(weights)/2], len(self.train_variables)+1)
            weights_std  = weights[len(weights)/2:]
            print "weight_std"
            print weights_std*10**20
            print "argument"
            print np.log(np.exp(np.log(np.expm1(1.))+weights_std)+1)

            weights_std  = np.split(np.log(np.exp(np.log(np.expm1(1.))+weights_std)+1), len(self.train_variables)+1)
            print weights_std #DEBUG
        else:
            weights_mean = np.split(weights[:len(weights)/2], len(self.train_variables))
            weights_std  = weights[len(weights)/2:]
            weights_std  = np.split(np.log(np.exp(np.log(np.expm1(1.))+weights_std)+1), len(self.train_variables))

        # print "layer 1 post: ",self.model.layers[1].get_weights()[0][:10]
        # print "layer 1 pri: ",self.model.layers[1].get_weights()[1][:10]
        # print self.model.layers[4].get_weights()

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

    # --------------------------------------------------------------------
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
            nbins               = 30,
            bin_range           = [0.,np.amax(self.model_prediction_vector_std)],
            event_category      = self.category_label,
            plotdir             = self.save_path,
            logscale            = log,
            sigScale            = sigScale,
            save_name           = "sigma_discriminator"
            )

        bkg_std_hist, sig_std_hist = binaryOutput_std.plot(ratio = False, printROC = printROC, privateWork = privateWork, name = "\sigma of the Discriminator")

        # #DEBUG
        # binaryOutput = plottingScripts.plotBinaryOutput(
        #     data                = self.data,
        #     test_predictions    = self.model_prediction_vector,
        #     train_predictions   = None,#self.model_train_prediction,
        #     nbins               = nbins,
        #     bin_range           = bin_range,
        #     event_category      = self.category_label,
        #     plotdir             = self.save_path,
        #     logscale            = log,
        #     sigScale            = sigScale,
        #     save_name           = "binary_discriminator_log" #me
        #     )

        # bkg_hist, sig_hist = binaryOutput.plot(ratio = False, printROC = printROC, privateWork = privateWork, name = name)

        # binaryOutput_std = plottingScripts.plotBinaryOutput(
        #     data                = self.data,
        #     test_predictions    = self.model_prediction_vector_std,
        #     train_predictions   = None, # self.model_train_prediction_std,
        #     nbins               = 30,
        #     bin_range           = [0.,np.amax(self.model_prediction_vector_std)],
        #     event_category      = self.category_label,
        #     plotdir             = self.save_path,
        #     logscale            = log,
        #     sigScale            = sigScale,
        #     save_name           = "sigma_discriminator_log"
        #     )

        # bkg_std_hist, sig_std_hist = binaryOutput_std.plot(ratio = False, printROC = printROC, privateWork = privateWork, name = "\sigma of the Discriminator")


        self.plot_2D_hist_std_over_mean(bin_range=[50,50])
        self.plot_varied_histogram()

    def plot_2D_hist_std_over_mean(self, bin_range=[40,40]):
        from matplotlib.colors import LogNorm
        plt.hist2d(self.model_prediction_vector, self.model_prediction_vector_std, bins=bin_range, cmin=1, norm=LogNorm())
        plt.colorbar()
        plt.xlabel("$\mu$", fontsize = 16)
        plt.ylabel("$\sigma$", fontsize = 16)
        plt.savefig(self.save_path+"/sigma_over_mu.png")
        print "sigma_over_mu.png was created"
        plt.savefig(self.save_path+"/sigma_over_mu.pdf")
        print "sigma_over_mu.pdf was created"
        plt.close()

    def plot_varied_histogram(self):
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
        plt.hist(sig_preds, bins=15, range=(0,1), histtype='step', density=True, label="ttH", color="b")
        plt.hist(np.array(sig_preds)+1.*np.array(sig_preds_std), bins=15, range=(0,1), histtype='step', density=True, linestyle=('--'), color="b")
        plt.hist(np.array(sig_preds)-1.*np.array(sig_preds_std), bins=15, range=(0,1), histtype='step', density=True, linestyle=(':'), color="b")
        plt.hist(bkg_preds, bins=15, range=(0,1), histtype='step', density=True, label="bkg", color="r")
        plt.hist(np.array(bkg_preds)+1.*np.array(bkg_preds_std), bins=15, range=(0,1), histtype='step', density=True, linestyle=('--'), color="r")
        plt.hist(np.array(bkg_preds)-1.*np.array(bkg_preds_std), bins=15, range=(0,1), histtype='step', density=True, linestyle=(':'), color="r")
        plt.xlabel("$\mu$", fontsize = 16)
        plt.legend()
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
            plt.plot(epochs, train_history, "b-", label = "train", lw = 2)
            plt.plot(epochs, val_history, "r-", label = "validation", lw = 2)
            if privateWork:
                plt.title("CMS private work", loc = "left", fontsize = 16)
                plt.title("best epoch: "+str(n_epochs), loc="center", fontsize = 16)
            else:
                plt.title("best epoch: "+str(n_epochs), loc="left", fontsize = 16)



            # add title
            title = self.category_label
            title = title.replace("\\geq", "$\geq$")
            title = title.replace("\\leq", "$\leq$")
            plt.title(title, loc = "right", fontsize = 16)

            # make it nicer
            plt.grid()
            plt.xlabel("epoch", fontsize = 16)
            plt.ylabel(metric.replace("_"," "), fontsize = 16)
            #plt.ylim(ymin=0.)

            # add legend
            plt.legend()

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
        plt.plot(epochs, train_history_KLD, "b-", label = "train", lw = 2)
        plt.plot(epochs, val_history_KLD, "r-", label = "validation", lw = 2)
        if privateWork:
            plt.title("CMS private work", loc = "left", fontsize = 16)

        # add title
        title = self.category_label
        title = title.replace("\\geq", "$\geq$")
        title = title.replace("\\leq", "$\leq$")
        plt.title(title, loc = "right", fontsize = 16)

        # make it nicer
        plt.grid()
        plt.xlabel("epoch", fontsize = 16)
        plt.ylabel("KLD", fontsize = 16)
        #plt.ylim(ymin=0.)

        # add legend
        plt.legend()

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