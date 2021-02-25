import os
import sys
import numpy as np
import json
import pickle
import math
from array import array
import ROOT
import pprint

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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam, SGD
# from tensorflow.keras.utils import np_utils
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation
# from keras.optimizers import Adam, SGD
from keras.utils import np_utils

# Limit gpu usage
import tensorflow as tf
import shap

import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

class EarlyStopping(keras.callbacks.Callback):
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


class DNN():
    def __init__(self,
            save_path,
            input_samples,
            category_name,
            train_variables,
            category_cutString = None,
            category_label     = None,
            norm_variables     = True,
            train_epochs       = 500,
            test_percentage    = 0.2,
            eval_metrics       = None,
            shuffle_seed       = None,
            balanceSamples     = False,
            evenSel            = None,
            addSampleSuffix    = ""):

        # save some information
        # list of samples to load into dataframe
        self.input_samples = input_samples

        # suffix of additional (ttbb) sample
        self.addSampleSuffix = addSampleSuffix

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

        # load data set
        self.data = self._load_datasets(shuffle_seed, balanceSamples)
        self.event_classes = self.data.output_classes

        # save variable norm
        self.cp_path = self.save_path+"/checkpoints/"
        if not os.path.exists(self.cp_path):
            os.makedirs(self.cp_path)

        if self.norm_variables:
           out_file = self.cp_path + "/variable_norm.csv"
           self.data.norm_csv.to_csv(out_file)
           print("saved variabe norms at "+str(out_file))

        # make plotdir
        self.plot_path = self.save_path+"/plots/"
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)

        # layer names for in and output (needed for c++ implementation)
        self.inputName = "inputLayer"
        self.outputName = "outputLayer"

    def logarithmic(self, array):
        indices = [i for i, x in enumerate(array) if x <= 0]
        for i in indices:
            print("# DEBUG: logarithmic, entries: ", array[i])
        return map(lambda x: math.log(x), array)

    def factorial(self, array):
        return map(lambda x: math.factorial(x), array)

    def _load_datasets(self, shuffle_seed, balanceSamples):
        ''' load data set '''
        return data_frame.DataFrame(
            input_samples    = self.input_samples,
            event_category   = self.category_cutString,
            train_variables  = self.train_variables,
            test_percentage  = self.test_percentage,
            norm_variables   = self.norm_variables,
            shuffleSeed      = shuffle_seed,
            balanceSamples   = balanceSamples,
            evenSel          = self.evenSel,
            addSampleSuffix  = self.addSampleSuffix
        )

    def _load_architecture(self, config):
        ''' load the architecture configs '''

        # define default network configuration
        self.architecture = {
          "layers":                   [200],
          "loss_function":            "categorical_crossentropy",
          "Dropout":                  0.2,
          "L1_Norm":                  0.,
          "L2_Norm":                  1e-5,
          "batch_size":               5000,
          "optimizer":                optimizers.Adagrad(decay=0.99),
          "activation_function":      "elu",
          "output_activation":        "Softmax",
          "earlystopping_percentage": None,
          "earlystopping_epochs":     None,
        }

        for key in config:
            self.architecture[key] = config[key]

    def load_trained_model(self, inputDirectory):
        ''' load an already trained model '''
        checkpoint_path = inputDirectory+"/checkpoints/trained_model.h5py"

        # get the keras model
        self.model = keras.models.load_model(checkpoint_path)
        self.model.summary()

        # evaluate test dataset with keras model
        self.model_eval = self.model.evaluate(self.data.get_test_data(as_matrix = True), self.data.get_test_labels())

        # save predictions  with keras model
        self.model_prediction_vector = self.model.predict(self.data.get_test_data (as_matrix = True) )
        self.model_train_prediction  = self.model.predict(self.data.get_train_data(as_matrix = True) )
        
        # save predicted classes with argmax  with keras model
        self.predicted_classes = np.argmax( self.model_prediction_vector, axis = 1)

        # save confusion matrix
        from sklearn.metrics import confusion_matrix
        self.confusion_matrix = confusion_matrix(self.data.get_test_labels(as_categorical = False), self.predicted_classes)

        # print evaluations  with keras model
        from sklearn.metrics import roc_auc_score
        self.roc_auc_score = roc_auc_score(self.data.get_test_labels(), self.model_prediction_vector)
        print("\nROC-AUC score: {}".format(self.roc_auc_score))

        #return self.model_prediction_vector

    def predict_event_query(self, query ):
        events = self.data.get_full_df().query( query )
        print(str(events.shape[0]) + " events matched the query '"+str(query)+"'.")

        for index, row in events.iterrows():
            print("========== DNN output ==========")
            print("Event: "+str(index))
            print("-------------------->")
            output = self.model.predict( np.array([list(row.values)]) )[0]
            print("output:" + str(output))

            for i, node in enumerate(self.event_classes):
                if i>=6: continue
                print(str(node)+" node: "+str(output[i]))
            print("-------------------->")


    def build_default_model(self):
        ''' build default straight forward DNN from architecture dictionary '''

        # infer number of input neurons from number of train variables
        number_of_input_neurons     = self.data.n_input_neurons

        # get all the architecture settings needed to build model
        number_of_neurons_per_layer = self.architecture["layers"]
        dropout                     = self.architecture["Dropout"]
        activation_function         = self.architecture["activation_function"]
        if activation_function == "leakyrelu":
            activation_function = "linear"
        l2_regularization_beta      = self.architecture["L2_Norm"]
        #l1_regularization_beta      = self.architecture["L1_Norm"]
        output_activation           = self.architecture["output_activation"]

        # define input layer
        Inputs = keras.layers.Input(
            shape = (number_of_input_neurons,),
            name  = self.inputName)
        X = Inputs
        self.layer_list = [X]

        # loop over dense layers
        for iLayer, nNeurons in enumerate(number_of_neurons_per_layer):
            X = keras.layers.Dense(
                units               = nNeurons,
                activation          = activation_function,
                kernel_regularizer  = keras.regularizers.l2(l2_regularization_beta),
                name                = "DenseLayer_"+str(iLayer)
                )(X)

            if self.architecture["activation_function"] == "leakyrelu":
                X = keras.layers.LeakyReLU(alpha=0.3)(X)

            # add dropout percentage to layer if activated
            if not dropout == 0:
                X = keras.layers.Dropout(dropout, name = "DropoutLayer_"+str(iLayer))(X)

        # generate output layer
        X = keras.layers.Dense(
            units               = self.data.n_output_neurons,
            activation          = output_activation.lower(),
            kernel_regularizer  = keras.regularizers.l2(l2_regularization_beta),
            name                = self.outputName
            )(X)

        # define model
        model = models.Model(inputs = [Inputs], outputs = [X])
        model.summary()

        return model

    def build_model(self, config = None, model = None, penalty = None):
        ''' build a DNN model
            use options defined in 'config' dictionary '''

        if config:
            self._load_architecture(config)
            print("loading non default net configs")

        if model == None:
            print("building model from config")
            model = self.build_default_model()

        # compile the model
        model.compile(
            loss        = self.architecture["loss_function"],
            optimizer   = self.architecture["optimizer"],
            metrics     = self.eval_metrics)

        # save the model
        self.model = model

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

        # train main net
        self.trained_model = self.model.fit(
            x = self.data.get_train_data(as_matrix = True),
            y = self.data.get_train_labels(),
            batch_size          = self.architecture["batch_size"],
            epochs              = self.train_epochs,
            shuffle             = True,
            callbacks           = callbacks,
            validation_split    = 0.25,
            sample_weight       = self.data.get_train_weights())

    def save_model(self, argv, execute_dir, netConfigName, get_gradients = True):
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
        configs["addSampleSuffix"] =self.addSampleSuffix
        configs["netConfig"] = self.netConfig
        configs["netConfig"] = self.test_percentage

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

        # Serialize the test inputs for the analysis of the gradients
        if get_gradients:
            pickle.dump(self.data.get_test_data(), open(self.cp_path+"/inputvariables.pickle", "wb"))


    def eval_model(self):
        ''' evaluate trained model '''

        # evaluate test dataset
        self.model_eval = self.model.evaluate(
            self.data.get_test_data(as_matrix = True),
            self.data.get_test_labels())

        # save history of eval metrics
        self.model_history = self.trained_model.history

        # save predicitons
        self.model_prediction_vector = self.model.predict(self.data.get_test_data (as_matrix = True))
        self.model_train_prediction  = self.model.predict(self.data.get_train_data(as_matrix = True))
        
        #figure out ranges
        self.get_ranges()

        # save predicted classes with argmax
        self.predicted_classes = np.argmax( self.model_prediction_vector, axis = 1)

        # save confusion matrix
        from sklearn.metrics import confusion_matrix
        self.confusion_matrix = confusion_matrix(self.data.get_test_labels(as_categorical = False), self.predicted_classes)

        # print evaluations
        from sklearn.metrics import roc_auc_score
        self.roc_auc_score = roc_auc_score(self.data.get_test_labels(), self.model_prediction_vector)
        print("\nROC-AUC score: {}".format(self.roc_auc_score))

        if self.eval_metrics:
            print("model test loss: {}".format(self.model_eval[0]))
            for im, metric in enumerate(self.eval_metrics):
                print("model test {}: {}".format(metric, self.model_eval[im+1]))

    def model_to_optimize(self, train_params):
        number_of_input_neurons =  self.data.get_train_data(as_matrix = True).shape[1]
        # print(number_of_input_neurons)
        print("-"*40)
        print("Testing following configuration:")
        pprint.pprint(train_params)
        #Model providing function:
        #Create Keras model with double curly brackets dropped-in as needed.
        model = Sequential()

        leaky_relu_activation = keras.layers.LeakyReLU(alpha=0.3)
        keras.layers.Dropout(train_params['dropout'], name = "Drop_out_layer1")

        model.add(Dense(units=int(train_params['layer_size_1']), activation=leaky_relu_activation, kernel_regularizer=keras.regularizers.l2(train_params['l2_regularizer']),
                        input_shape=(number_of_input_neurons,)))
        model.add(keras.layers.Dropout(train_params['dropout'], name = "Drop_out_layer1"))

        model.add(Dense(units=int(train_params['layer_size_2']), kernel_regularizer=keras.regularizers.l2(train_params['l2_regularizer']),
                        activation=leaky_relu_activation,))
        model.add(keras.layers.Dropout(train_params['dropout'], name = "Drop_out_layer2"))
    
        model.add(Dense(units=int(train_params['layer_size_3']), kernel_regularizer=keras.regularizers.l2(train_params['l2_regularizer']),
                        activation=leaky_relu_activation,))
        model.add(keras.layers.Dropout(train_params['dropout'], name = "Drop_out_layer3"))

        if train_params['four_layer']['include']:
            model.add(Dense(units=int(train_params['four_layer']['layer_size_4']), kernel_regularizer=keras.regularizers.l2(train_params['l2_regularizer']),
                            activation=leaky_relu_activation,))
            model.add(keras.layers.Dropout(train_params['dropout'], name = "Drop_out_layer4"))

        model.add(Dense(self.data.n_output_neurons, kernel_regularizer=keras.regularizers.l2(train_params['l2_regularizer'])))

        if self.data.binary_classification :
            if 0 in self.data.get_train_labels():
                loss = 'binary_crossentropy'
                model.add(Activation('sigmoid'))
            else:
                loss = 'squared_hinge'
                model.add(Activation('tanh'))
        else:
            model.add(Activation('softmax'))
            loss = 'categorical_crossentropy'

        if train_params['optimizer']['name'] == 'adam':
            opt = keras.optimizers.Adam(lr = train_params['optimizer']['learning_rate'])
        elif train_params['optimizer']['name'] == 'sgd':
            opt = keras.optimizers.SGD(lr = train_params['optimizer']['learning_rate'])
        elif train_params['optimizer']['name'] == 'RMSprop':
            opt = keras.optimizers.RMSprop(lr = train_params['optimizer']['learning_rate'])
        elif train_params['optimizer']['name'] == 'Adagrad':
            # opt = keras.optimizers.Adagrad(lr = train_params['optimizer']['learning_rate'])
            opt = keras.optimizers.Adagrad()
        elif train_params['optimizer']['name'] == 'Adadelta':
            opt = keras.optimizers.Adadelta(lr = train_params['optimizer']['learning_rate'])
        elif train_params['optimizer']['name'] == 'Adamax':
            opt = keras.optimizers.Adamax(lr = train_params['optimizer']['learning_rate'])
        else:
            opt = keras.optimizers.Nadam(lr = train_params['optimizer']['learning_rate'])

        # add early stopping if activated
        callbacks = None
        callbacks = [EarlyStopping(
            monitor         = "loss",
            value           = 0.05,
            min_epochs      = 50,
            stopping_epochs = 50,
            verbose         = 1)]
        # compile the model
        model.compile(
            loss        =   loss,
            metrics     =   ['accuracy'],
            optimizer   =   opt)
        
        print(model.summary())

        # train main net
        model.fit(
            x                   = self.data.get_train_data(as_matrix = True),
            y                   = self.data.get_train_labels(),
            batch_size          = int(train_params['batch_size']),
            epochs              = 100,
            shuffle             = True,
            validation_split    = 0.25,
            sample_weight       = self.data.get_train_weights())

        return model

    def roc_model_to_optimize(self, model):
        from sklearn.metrics import roc_auc_score
        model_prediction_vector = model.predict(self.data.get_test_data (as_matrix = True) )
        roc_auc_score = roc_auc_score(self.data.get_test_labels(), model_prediction_vector)
        return np.nanmin(roc_auc_score)

    def evaluate_model_to_optimize(self, model):
        test_accuracy_to_optimize = model.evaluate(self.data.get_test_data(as_matrix = True), self.data.get_test_labels())
        return np.nanmin(test_accuracy_to_optimize)

    def hyperopt_fcn(self,params):
          from hyperopt import  STATUS_OK
          model = self.model_to_optimize(params)
          test_acc = np.nanmin(self.evaluate_model_to_optimize(model))
          roc_auc_score = np.nanmin(self.roc_model_to_optimize(model))
          print("roc_auc_score")
          print(roc_auc_score)
          K.clear_session()

          return {'loss': -roc_auc_score,  'status': STATUS_OK}



    def get_ranges(self):
        if not self.data.binary_classification:
            max_ = [0.]*len(self.input_samples.samples)
            for ev in self.model_prediction_vector:
                for i,node in enumerate(ev):
                    if node>max_[i]:
                        max_[i]=node
            for i, sample in enumerate(self.input_samples.samples):
                sample.max=round(float(max_[i]),2)
                sample.min=round(float(1./len(self.input_samples.samples)),2)



    def get_input_weights(self):
        ''' get the weights of the input layer and sort input variables by weight sum '''

         # get weights
        first_layer = self.model.layers[1]
        weights = first_layer.get_weights()[0]

        self.weight_dict = {}
        for out_weights, variable in zip(weights, self.train_variables):
            w_sum = np.sum(np.abs(out_weights))
            self.weight_dict[variable] = w_sum

        # sort weight dict
        rank_path = self.save_path + "/first_layer_weight_sums.csv"
        with open(rank_path, "w") as f:
            f.write("variable,weight_sum\n")
            for key, val in sorted(self.weight_dict.iteritems(), key = lambda (k,v): (v,k)):
                #print("{:50s}: {}".format(key, val))
                f.write("{},{}\n".format(key,val))
        print("wrote weight ranking to "+str(rank_path))

    def get_weights(self):
        ''' get the weights of the all hidden layers and sort input variables by weight sum'''

        # get weights
        for i, layer in enumerate(self.model.layers):
            #odd layers correspond to dropout layers
            if ("Dropout" in layer.name or "leaky" in layer.name or "inputLayer" in layer.name):
                continue
            else:
                weights = layer.get_weights()[0]

                self.weight_dict = {}
                for out_weights, variable in zip(weights, self.train_variables):
                    w_sum = np.sum(np.abs(out_weights))
                    self.weight_dict[variable] = w_sum

                # sort weight dict
                rank_path = self.save_path + "/layer_"+str(i)+"_weight_sums.csv"
                with open(rank_path, "w") as f:
                    f.write("variable,weight_sum\n")
                    for key, val in sorted(self.weight_dict.iteritems(), key = lambda (k,v): (v,k)):
                        #print("{:50s}: {}".format(key, val))
                        f.write("{},{}\n".format(key,val))
                print("wrote weight ranking to "+str(rank_path))

    def get_propagated_weights(self):
        weight_layers = []
        for i, layer in enumerate(self.model.layers):
            if ("Dropout" in layer.name or "leaky" in layer.name or "inputLayer" in layer.name):
                continue

            weights = layer.get_weights()[0]

            print("="*30)
            print("layer {}".format(i))
            print(weights)
            print("="*30)
            weight_layers.append(weights)

        # iteratively generate sums
        print("propagating weights")
        propagated_weights = []
        for i in range(len(weight_layers)):
            index = (len(weight_layers)-i)-1
            print(index)
            if i == 0:
                propagated_weights.append(
                    np.array([np.sum(np.abs(out_weights)) for out_weights in weight_layers[index]])
                    )
            else:
                propagated_weights.append(
                    [np.sum(np.abs(out_weights)*propagated_weights[i-1]) for out_weights in weight_layers[index]]
                    # [propagated_weights[i-1][j]*weight_layers[index][j] for j in range(len(weight_layers[index]))]
                    )
            print(propagated_weights[i])

        weight_dict = {}
        for weight, variable in zip(propagated_weights[-1], self.train_variables):
            weight_dict[variable] = weight

        rank_path = self.save_path+"/propagated_weight_sums.csv"
        with open(rank_path, "w") as f:
            f.write("variable,weight_sum\n")
            for key, val in sorted(weight_dict.iteritems(), key = lambda (k,v): (v,k)):
                print("{:50s}: {}".format(key, val))
                f.write("{},{}\n".format(key, val))
        print("wrote propagated weight ranking to "+str(rank_path))


    def get_variations(self, is_binary):
        if not os.path.exists(self.save_path + "/variations/"):
            os.makedirs(self.save_path + "/variations/")

        print("making plots for input feature variations")
        for i, v in enumerate(self.train_variables):

            test_values = np.linspace(-2,2,500)
            testset = np.array([
                np.array([0 if not j==i else k for j in range(len(self.train_variables))])
                for k in test_values])

            predictions = self.model.predict(testset)

            yrange = [0., 2./len(self.event_classes)]

            plt.clf()
            plt.plot([-2,2],[1./len(self.event_classes),1./len(self.event_classes)], "-", color = "black")
            plt.plot([0.,0.],yrange, "-", color = "black")

            for n, node in enumerate(self.event_classes):
                plt.plot(test_values, predictions[:,n], "-", linewidth = 2, label = node+" node")
                if is_binary: break

            plt.grid()
            plt.legend()
            title = self.category_label
            title = title.replace("\\geq", "$\geq$")
            title = title.replace("\\leq", "$\leq$")
            plt.title(title, loc = "right", fontsize = 16)
            plt.xlabel(v, fontsize = 16)
            plt.ylabel("node output", fontsize = 16)
            plt.xlim([-2,2])
            plt.ylim(yrange)
            plt.tight_layout()
            outpath = self.save_path + "/variations/"+str(v)+".pdf"
            plt.savefig(outpath)
            plt.savefig(outpath.replace(".pdf",".png"))
            print("plot saved at {}".format(outpath))


    def get_gradients(self, is_binary):

        # Load keras model
        checkpoint_path = self.cp_path+"/trained_model.h5py"
        # get the keras model
        model_keras = keras.models.load_model(checkpoint_path, compile=False)

        # Get TensorFlow graph
        inputs = Inputs(self.train_variables)
        try:
            import net_configs_tensorflow
        except:
            print("Failed to import Tensorflow models.")
            quit()

        try:
            name_keras_model = self.netConfig
            model_tensorflow_impl = getattr(
                net_configs_tensorflow, self.netConfig + "_tensorflow")
        except:
            print(
                "Failed to load TensorFlow version of Keras model {}.".format(
                    name_keras_model))
            quit()

        #  Get weights as numpy arrays, load weights in tensorflow variables and build tensorflow graph with weights from keras model
        model_tensorflow = model_tensorflow_impl(inputs.placeholders, model_keras)

        # Load test data
        x_in = pickle.load(open(self.cp_path+"/inputvariables.pickle", "rb"))

        if is_binary:
            outputs = Outputs(model_tensorflow, ['sig'])
            event_classes = ['sig']
        else:
            outputs = Outputs(model_tensorflow, self.event_classes)
            event_classes = self.event_classes

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Get operations for first-order derivatives
        deriv_ops = {}
        derivatives = Derivatives(inputs, outputs)
        for class_ in event_classes:
            deriv_ops[class_] = []
            for variable in self.train_variables:
                deriv_ops[class_].append(derivatives.get(class_, [variable]))

        mean_abs_deriv = {}

        for class_ in event_classes:

            weight = array("f", [-999])
            deriv_class = np.zeros((len(self.data.get_test_data(as_matrix = True)),len(self.train_variables)))
            weights = np.zeros(len(self.data.get_test_data()))

            # Calculate first-order derivatives
            deriv_values = sess.run(
                deriv_ops[class_],
                feed_dict={
                    inputs.placeholders: x_in
                })
            deriv_values = np.squeeze(deriv_values)

            mean_abs_deriv[class_] = np.average(np.abs(deriv_values), axis=1)


        # Normalize rows
        matrix = np.vstack([mean_abs_deriv[class_] for class_ in event_classes])
        for i_class, class_ in enumerate(event_classes):
            matrix[i_class, :] = matrix[i_class, :] / np.sum(matrix[i_class, :])


        # Make plot
        variables = self.train_variables
        plt.figure(0, figsize=(len(variables), len(event_classes)))
        axis = plt.gca()

        print(matrix.shape[0])
        print(matrix.shape[1])

        csvtext = "variable,"+",".join(event_classes)
        for j in range(matrix.shape[1]):
            csvtext+="\n"+variables[j]
            for i in range(matrix.shape[0]):
                csvtext+=",{:.3f}".format(matrix[i,j])
                axis.text(
                    j + 0.5,
                    i + 0.5,
                    '{:.3f}'.format(matrix[i, j]),
                    ha='center',
                    va='center')

        q = plt.pcolormesh(matrix, cmap='Oranges')
        #cbar = plt.colorbar(q)
        #cbar.set_label("mean(abs(Taylor coefficients))", rotation=270, labelpad=20)
        variables_label = [v.replace("_","\_") for v in variables]
        event_classes_label = [v.replace("_","\_") for v in event_classes]
        plt.xticks(
            np.array(range(len(variables))) + 0.5, variables_label, rotation='vertical')
        plt.yticks(
            np.array(range(len(event_classes))) + 0.5, event_classes_label, rotation='horizontal')
        plt.xlim(0, len(variables))
        plt.ylim(0, len(event_classes_label))
        output_path = os.path.join(self.cp_path,
                                   "keras_taylor_1D")

        plt.savefig(output_path+".png", bbox_inches='tight')
        print("Save plot to {}.png".format(output_path))
        plt.savefig(output_path+".pdf", bbox_inches='tight')
        print("Save plot to {}.pdf".format(output_path))

        with open(output_path+".csv", "w") as f:
            f.write(csvtext)
        print("wrote coefficient information to {}.csv".format(output_path))


    # def __create_shap_dependance_plots(self, index, shap_values, test_data, train_variables, outname):
    #     plt.clf()
    #     fig, ax = plt.subplots(figsize=(20, 8))
    #     shap.summary_plot(ind = index, shap_values = shap_values,
    #                                     features = test_data, 
    #                                     feature_names= train_variables,
    #                                     # interaction_index = None
    #                                     )
    #     plt.savefig("{}.pdf".format(outname))
    #     plt.savefig("{}.png".format(outname))
    #     plt.close()

    def __create_shap_plots(self, shap_values, class_names, outname, \
                            test_data, train_variables, plot_dependance = False):
        max_display = test_data.shape[1]

        plt.clf()
        fig, ax = plt.subplots(figsize=(20, 8))
        shap.summary_plot(shap_values, test_data, plot_type = "bar",
            feature_names = train_variables,
            class_names = class_names,
            max_display = max_display, show = False, auto_size_plot=False)

        plt.savefig(self.save_path+"/{}.pdf".format(outname))
        plt.savefig(self.save_path+"/{}.png".format(outname))
        plt.close()
        if plot_dependance:
            # outpath = os.path.join(self.save_path, class_names[0])
            # if not os.path.exists(outpath):
            #     os.makedirs(outpath)
            # outname = os.path.join(outpath, "dependance_{}".format(train_variables[i]))
            plt.clf()
            fig, ax = plt.subplots(figsize=(20, 8))
            shap.summary_plot(shap_values, test_data, plot_type = "violin",
                                feature_names = train_variables,
                                class_names = class_names,
                                max_display = max_display, show = False, 
                                auto_size_plot=False)
            plt.savefig(self.save_path+"/{}_violin.pdf".format(outname))
            plt.savefig(self.save_path+"/{}_violin.png".format(outname))
            plt.close()

            plt.clf()
            fig, ax = plt.subplots(figsize=(20, 8))
            shap.summary_plot(shap_values, test_data, plot_type = "layered_violin",
                                feature_names = train_variables,
                                class_names = class_names,
                                max_display = max_display, show = False, 
                                auto_size_plot=False)
            plt.savefig(self.save_path+"/{}_layered_violin.pdf".format(outname))
            plt.savefig(self.save_path+"/{}_layered_violin.png".format(outname))
            plt.close()
                # self.__create_shap_dependance_plots(index=i,
                #                                     shap_values=shap_values,
                #                                     test_data=test_data,
                #                                     train_variables=train_variables,
                #                                     outname = outname)
                

    def get_shap_evaluation(self, samplesize = 10000):


        train_variables = self.train_variables
        train_data = self.data.get_train_data (as_matrix = True, sort_list = train_variables)[:samplesize]
        test_data = self.data.get_test_data (as_matrix = True, sort_list = train_variables)[:samplesize]
        y_predicted = self.model_prediction_vector
        class_names = self.event_classes
        print("classes: {}".format(", ".join(class_names)))
        
        # exit(0)
        # for explainer, name  in ((shap.DeepExplainer(model,inX_test[:1000]),"DeepExplainer"), (shap.GradientExplainer(model,inX_test[:1000]),"GradientExplainer")):
        # print(json.dumps(self.model.__dict__, indent=4))
        for explainer, name  in [(shap.GradientExplainer(self.model, 
                                    train_data, 
                                    session = tf.compat.v1.keras.backend.get_session()),\
                                        "GradientExplainer")]:
            shap.initjs()
            print("... {0}: explainer.shap_values(X)".format(name))
            shap_values = explainer.shap_values(test_data)
            print("... shap.summary_plot")
            print("... do overall plot")
            outname = "shap_summary_{}_{}".format(name, samplesize)
            self.__create_shap_plots(shap_values=shap_values, class_names= class_names,
                                        outname=outname, test_data= test_data,
                                        train_variables=train_variables)
            exclude_nodes = "tHq tHW".split()
            interesting_nodes = class_names
            sub_shap_values = shap_values

            for node in exclude_nodes:
                idx = interesting_nodes.index(node)
                interesting_nodes.pop(idx)
                sub_shap_values.pop(idx)

            print("... do summary plot for nodes {}".format(", ".join(interesting_nodes)))
            outname = "shap_summary_{}_main_nodes_{}".format(name, samplesize)
            self.__create_shap_plots(shap_values=sub_shap_values, class_names= interesting_nodes,
                                        outname=outname, test_data= test_data,
                                        train_variables=train_variables)

            # summary plot for each node separately
            for i, cname in enumerate(class_names):
                print("... do summary plot for node {}".format(cname))
                outname = "shap_summary_{}_{}_{}".format(name, cname, samplesize)
                self.__create_shap_plots(shap_values=shap_values[i], class_names= [cname],
                                        outname=outname, test_data= test_data,
                                        train_variables=train_variables,
                                        plot_dependance=True)
            


    # --------------------------------------------------------------------
    # result plotting functions
    # --------------------------------------------------------------------
    def plot_metrics(self, privateWork = False):
        plt.rc('text', usetex=True)

        ''' plot history of loss function and evaluation metrics '''
        metrics = ["loss"]
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

            # add title
            title = self.category_label
            title = title.replace("\\geq", "$\geq$")
            title = title.replace("\\leq", "$\leq$")
            plt.title(title, loc = "right", fontsize = 16)

            # make it nicer
            plt.grid()
            plt.xlabel("epoch", fontsize = 16)
            plt.ylabel(metric.replace("_"," "), fontsize = 16)
            # plt.ylim(ymin=0.)

            # add legend
            plt.legend()

            # save
            out_path = self.save_path + "/model_history_"+str(metric)+".pdf"
            plt.savefig(out_path)
            print("saved plot of "+str(metric)+" at "+str(out_path))

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

    def plot_confusionMatrix(self, norm_matrix = True, privateWork = False, printROC = False):
        ''' plot confusion matrix '''
        plotCM = plottingScripts.plotConfusionMatrix(
            data                = self.data,
            prediction_vector   = self.model_prediction_vector,
            event_classes       = self.event_classes,
            event_category      = self.category_label,
            plotdir             = self.save_path)

        plotCM.plot(norm_matrix = norm_matrix, privateWork = privateWork, printROC = printROC)

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

    def plot_binaryOutput(self, log = False, privateWork = False, printROC = False,
                        nbins = None, bin_range = [0.,1.], name = "binary_discriminator",
                        sigScale = -1):

        if not nbins:
            nbins = int(50*(1.-bin_range[0]))

        binaryOutput = plottingScripts.plotBinaryOutput(
            data                = self.data,
            test_predictions    = self.model_prediction_vector,
            train_predictions   = self.model_train_prediction,
            nbins               = nbins,
            bin_range           = bin_range,
            event_category      = self.category_label,
            plotdir             = self.save_path,
            logscale            = log,
            sigScale            = sigScale)

        bkg_hist, sig_hist = binaryOutput.plot(ratio = False, printROC = printROC, privateWork = privateWork, name = name)
        #print("ASIMOV: mu=0: sigma (-+): ", self.binned_likelihood(bkg_hist, sig_hist, 0))
        #print("ASIMOV: mu=1: sigma (-+): ", self.binned_likelihood(bkg_hist, sig_hist, 1))

    def calc_LL(self,n_obs, n_exp):
        if n_obs > 0 and n_exp >= 0:
            n_obs = int(round(n_obs))
            tmp = n_exp - n_obs*math.log(n_exp) + math.log(math.factorial(n_obs))
        else:
            tmp = 0
        return tmp


    def binned_likelihood(self, bkg_bins, sig_bins, mu):
        '''Calculates sigma1 and sigma2 for asimov data set and makes a plot'''

        save_path = self.save_path + "/plots/"

        obs_bins = bkg_bins + mu * sig_bins
        #remove bins with no bkg events -> they will couse problems due to log
        indices = [i for i, x in enumerate(bkg_bins) if x <= 0]
        obs_bins = np.delete(obs_bins, indices)
        bkg_bins = np.delete(bkg_bins, indices)
        sig_bins = np.delete(sig_bins, indices)

        mu_scan = np.linspace(0, 2, 100, endpoint = True)

        ll_scan = []
        for m in mu_scan:
            s_b_bins = bkg_bins + sig_bins*m
            # print("scanning {}".format(m))
            tmp = 0
            for obs, pred in zip(obs_bins,s_b_bins):
                ll = self.calc_LL(obs,pred)
                tmp+=ll
            ll_scan.append(2*tmp)
        ll_scan=ll_scan-min(ll_scan)

        indices = [i for i, x in enumerate(ll_scan) if x > 10]
        ll_scan = np.delete(ll_scan, indices)
        mu_scan = np.delete(mu_scan, indices)


        c=ROOT.TCanvas("c","c",800,600)
        c.cd()
        gr = ROOT.TGraph(len(ll_scan),mu_scan,ll_scan)
        gr.Draw("A*")

        line = ROOT.TLine(gr.GetXaxis().GetXmin(),1,gr.GetXaxis().GetXmax(),1)
        line.SetLineColor(ROOT.kRed)
        line.Draw("same")

        fit = gr.Fit("pol4","S")
        f = gr.GetFunction("pol4")
        sigmin = abs(f.GetX(1,0,f.GetMinimumX()) - f.GetMinimumX())
        sigmax = abs(f.GetX(1,f.GetMinimumX(),2) - f.GetMinimumX())
        gr.GetXaxis().SetTitle("#mu")
        gr.GetYaxis().SetTitle("2NLL")
        text = ROOT.TLatex(0.11,0.8,"{0} + {1} - {2}".format(f.GetMinimumX(),sigmax, sigmin))
        text.SetNDC(ROOT.kTRUE);
        text.Draw()
        c.Print(save_path + "/LL_mu" + str(mu) +".pdf")
        c.Print(save_path + "/LL_mu" + str(mu) +".png")
        return sigmin, sigmax


def loadDNN(inputDirectory, outputDirectory, binary = False, signal = None, \
            binary_target = None, \
            total_weight_expr = 'x.Weight_XS * x.Weight_CSV * x.Weight_GEN_nom', \
            category_cutString = None, category_label= None):

    # get net config json
    configFile = inputDirectory+"/checkpoints/net_config.json"
    if not os.path.exists(configFile):
        sys.exit("config needed to load trained DNN not found\n{}".format(configFile))

    with open(configFile) as f:
        config = f.read()
    config = json.loads(config)

    test_percentage = config.get("test_percentage", 0.2)
    # load samples
    input_samples = data_frame.InputSamples(config["inputData"], addSampleSuffix = config["addSampleSuffix"], 
                                            test_percentage = test_percentage)

    if binary:
        input_samples.addBinaryLabel(signal, binary_target)

    for sample in config["eventClasses"]:
        input_samples.addSample(sample["samplePath"], sample["sampleLabel"], normalization_weight = sample["sampleWeight"], total_weight_expr = total_weight_expr)

    print("shuffle seed: {}".format(config["shuffleSeed"]))
    # init DNN class
    dnn = DNN(
      save_path       = outputDirectory,
      input_samples   = input_samples,
      category_name   = config["JetTagCategory"],
      train_variables = config["trainVariables"],
      shuffle_seed    = config["shuffleSeed"],
      addSampleSuffix = config["addSampleSuffix"],
      test_percentage = test_percentage
    )



    # load the trained model
    dnn.load_trained_model(inputDirectory)
    dnn.netConfig = config.get("netConfig", None)

    # dnn.predict_event_query()

    return dnn
