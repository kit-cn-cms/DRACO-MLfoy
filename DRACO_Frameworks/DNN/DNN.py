# -*- encoding: utf-8 -*-

import os
import sys
import numpy as np
import math
# trining to solve some displaying problems (via ssh) here
# solution is to use ssh -X ... or the following two lines
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.rcParams['backend'] = 'Agg'
import json

#sklearn imports
from sklearn import metrics
#interpolation function to calculate sigma
from scipy import interpolate

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
import keras.optimizers as optimizers
import keras.models as models
import keras.layers as layer
from keras import backend as K
import pandas as pd

# Limit gpu usage
import tensorflow as tf

# Use latex style in plots
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))




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
            path,
            name,
            input_samples,
            event_category,
            train_variables,
            binary_bkg_target,
            train_epochs    = 500,
            test_percentage = 0.2,
            eval_metrics    = None,
            shuffle_seed    = None,
            balanceSamples  = False,
            evenSel         = None):

        # save some information
        #path and name to save model and plots
        self.path = path
        self.name = name

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

        # get cutting value to dedide if event is background or target
        self.cut_value = (1+int(binary_bkg_target))/2

        #get background target value
        self.binary_bkg_target = binary_bkg_target

        # percentage of events saved for testing
        self.test_percentage = test_percentage

        # number of train epochs
        self.train_epochs = train_epochs

        # additional metrics for evaluation of the training process
        self.eval_metrics = eval_metrics

        # load data set
        self.data = self._load_datasets(shuffle_seed, balanceSamples)
        self.event_classes = self.data.output_classes

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

        # layer names for in and output (needed for c++ implementation)
        self.inputName = "inputLayer"
        self.outputName = "outputLayer"



    def _load_datasets(self, shuffle_seed, balanceSamples):
        ''' load data set '''
        return data_frame.DataFrame(
            input_samples       = self.input_samples,
            event_category      = self.event_category,
            train_variables     = self.train_variables,
            test_percentage     = self.test_percentage,
            shuffleSeed         = shuffle_seed,
            balanceSamples      = balanceSamples,
            evenSel             = self.evenSel)


    def _load_architecture(self, config):
        ''' load the architecture configs '''
        # defnie default network configuration
        self.architecture = {
            "layers":                   [200],
            "loss_function":            "categorical_crossentropy",
            "Dropout":                  0.2,
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
        checkpoint_path = inputDirectory+"/trained_model.h5py"

        # get the model
        self.model = keras.models.load_model(checkpoint_path)
        self.model.summary()

        # evaluate test dataset
        self.model_eval = self.model.evaluate(
            self.data.get_test_data(as_matrix = True),
            self.data.get_test_labels())

        # save predicitons
        self.model_prediction_vector = self.model.predict(
            self.data.get_test_data(as_matrix = True) )
        self.model_train_prediction  = self.model.predict(
            self.data.get_train_data(as_matrix = True) )

        # save predicted classes with argmax
        self.predicted_classes = np.argmax( self.model_prediction_vector, axis = 1)

        # save confusion matrix
        # from sklearn.metrics import confusion_matrix
        # self.confusion_matrix = confusion_matrix(
        #     self.data.get_test_labels(as_categorical = False), self.predicted_classes)

        # print evaluations
        # from sklearn.metrics import roc_auc_score
        # self.roc_auc_score = roc_auc_score(self.data.get_test_labels(), self.model_prediction_vector)
        # print("\nROC-AUC score: {}".format(self.roc_auc_score))


    def predict_event_query(self, query ):
        events = self.data.get_full_df().query( query )
        print(str(events.shape[0]) + " events matched the query '"+str(query)+"'.")

        for index, row in events.iterrows():
            print("========== DNN output ==========")
            print("Event: "+str(index))
            for var in row.values:
                print(var)
            print("-------------------->")
            output = self.model.predict( np.array([list(row.values)]) )[0]
            for i, node in enumerate(self.event_classes):
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


    # def ada_eval_training(self):
    #     '''Calculate weighted error and return alpha_t after each training'''
    #     # model_prediction_vector = self.model.predict(self.data.get_test_data(as_matrix = True))
    #     model_train_prediction = self.model.predict(self.data.get_train_data(as_matrix = True))
    #     model_train_label = self.data.get_train_labels(as_categorical = False) #not sure if should be True
    #     model_train_weights = self.data.get_train_weights()
    #     #Calculate epsilon and alpha
    #     num = model_train_prediction.shape[0]
    #     # make_discret = lambda x: -1 if x<0 else 1
    #     model_train_prediction_discret = np.array([])
    #     for x in model_train_prediction:
    #         if x<0:
    #             model_train_prediction_discret = np.append(model_train_prediction_discret, -1)
    #         else:
    #             model_train_prediction_discret = np.append(model_train_prediction_discret, 1)
    #     weight_sum = np.sum(model_train_weights)
    #     weight_false = 0
    #     for i in np.arange(0,num):
    #         if model_train_prediction_discret[i] != model_train_label[i]:
    #             weight_false += model_train_weights[i]
    #     epsilon = weight_false/weight_sum
    #     alpha = 0.5*np.log((1-epsilon)/epsilon)
    #     #adjust weights
    #     self.data.ada_adjust_weights(model_train_prediction_discret, alpha)
    #     #check if epsilon < 0.5
    #     if epsilon > 0.5:
    #         print("# DEBUG: In ada_eval_training epsilon > 0.5")
    #     return alpha


    def build_model(self, config = None, model = None):
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

        # if self.use_adaboost:
        #     # train with adaboost algorithm
        #     self.weak_model_trainout = [] #does not contain the trained model
        #     self.weak_model_trained = [] #trained weak Classifier
        #     self.alpha_t = []
        #     for t in np.arange(0,self.adaboost_epochs):
        #         self.weak_model_trainout.append(self.model.fit(
        #             x = self.data.get_train_data(as_matrix = True),
        #             y = self.data.get_train_labels(),
        #             batch_size          = self.architecture["batch_size"],
        #             epochs              = self.train_epochs,
        #             shuffle             = True,
        #             callbacks           = callbacks,
        #             validation_split    = 0.25,
        #             sample_weight       = self.data.get_train_weights()))
        #         self.alpha_t.append(self.ada_eval_training())   #make dict alpha -> model
        #         self.weak_model_trained.append(self.model)

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


    def save_model(self, signals):
        ''' save the trained model '''

        save_path = self.path + "save_model/" + self.name + "/"
        # create new dir for the trained net
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        else:
            print("Dir already exists -> can not save model")
            sys.exit()

        # save executed command
        # argv[0] = execute_dir+"/"+argv[0].split("/")[-1]
        # execute_string = "python "+" ".join(argv)
        # out_file = self.cp_path+"/command.sh"
        # with open(out_file, "w") as f:
        #     f.write(execute_string)
        # print("saved executed command to {}".format(out_file))

        # save model as h5py file
        out_file = save_path + "trained_model.h5py"
        self.model.save(out_file)
        print("saved trained model at "+str(out_file))

        # save config of model
        model_config = self.model.get_config()
        out_file = save_path +"trained_model_config"
        with open(out_file, "w") as f:
            f.write( str(model_config))
        print("saved model config at "+str(out_file))

        # save final prediciton_vector
        np.save(save_path + "pred_vec", self.model_prediction_vector)

        # save roc-aux score
        np.save(save_path + "roc", np.array([self.roc_auc]))

        # save weights of network
        # out_file = self.cp_path +"/trained_model_weights.h5"
        # self.model.save_weights(out_file)
        # print("wrote trained weights to "+str(out_file))

        # set model as non trainable
        # for layer in self.model.layers:
        #     layer.trainable = False
        # self.model.trainable = False

        # save checkpoint files (needed for c++ implementation)
        # out_file = self.cp_path + "/trained_model"
        # saver = tf.train.Saver()
        # sess = K.get_session()
        # save_path = saver.save(sess, out_file)
        # print("saved checkpoint files to "+str(out_file))

        # produce json file with configs
        configs = self.architecture
        configs["inputName"] = self.inputName
        configs["outputName"] = self.outputName+"/"+configs["output_activation"]
        configs = {key: configs[key] for key in configs if not "optimizer" in key}

        # more information saving
        configs["inputData"] = self.input_samples.input_path
        configs["eventClasses"] = self.input_samples.getClassConfig()
        configs["JetTagCategory"] = self.JTstring
        configs["categoryLabel"] = self.categoryLabel
        configs["Selection"] = self.event_category
        configs["trainEpochs"] = self.train_epochs
        configs["trainVariables"] = self.train_variables
        configs["shuffleSeed"] = self.data.shuffleSeed
        configs["trainSelection"] = self.evenSel
        configs["evalSelection"] = self.oddSel

        # save information for binary DNN
        if self.data.binary_classification:
            configs["binaryConfig"] = {
                "minValue": self.input_samples.bkg_target,
                "maxValue": 1.,
                "signals":  signals}

        json_file = save_path + "net_config.json"
        with open(json_file, "w") as jf:
            json.dump(configs, jf, indent = 2, separators = (",", ": "))
        print("wrote net configs to "+str(json_file))

        # save configurations of variables for plotscript
        # plot_file = self.cp_path+"/plot_config.csv"
        # variable_configs = pd.read_csv(basedir+"/pyrootsOfTheCaribbean/plot_configs/variableConfig.csv").set_index("variablename", drop = True)
        # variables = variable_configs.loc[self.train_variables]
        # variables.to_csv(plot_file, sep = ",")
        # print("wrote config of input variables to {}".format(plot_file))

    #
    # def eval_adamodel(self):
    #     '''Evaluate a model trained with adaboost after each trainround t'''
    #     # print("# DEBUG: evalv_adamodel size of weak_model_trained: ", len(self.weak_model_trained))
    #     pass

    def eval_model(self):
        ''' evaluate trained model '''

        #get saving path
        save_path = self.path + "plot/"
        #get the labels
        self.train_label = self.data.get_train_labels(as_categorical = False)
        self.test_label = self.data.get_test_labels(as_categorical = False)

        # evaluate test dataset
        self.model_eval = self.model.evaluate(
            self.data.get_test_data(as_matrix = True),
            self.data.get_test_labels())

        # save history of eval metrics
        self.model_history = self.trained_model.history

        # save predicitons
        self.model_prediction_vector = self.model.predict(
            self.data.get_test_data(as_matrix = True) )
        self.model_train_prediction  = self.model.predict(
            self.data.get_train_data(as_matrix = True) )

        #figure out ranges
        self.get_ranges()

        # save predicted classes with argmax
        self.predicted_classes = np.argmax( self.model_prediction_vector, axis = 1)

        # save confusion matrix
        from sklearn.metrics import confusion_matrix
        self.confusion_matrix = confusion_matrix(
            self.data.get_test_labels(as_categorical = False), self.predicted_classes)

        # print evaluations
        from sklearn.metrics import roc_auc_score
        self.roc_auc_score = roc_auc_score(self.data.get_test_labels(), self.model_prediction_vector)
        print("\nROC-AUC score: {}".format(self.roc_auc_score))

        if self.eval_metrics:
            print("model test loss: {}".format(self.model_eval[0]))
            for im, metric in enumerate(self.eval_metrics):
                print("model test {}: {}".format(metric, self.model_eval[im+1]))

        #get roc
        fpr, tpr, thresholds = metrics.roc_curve(self.test_label, self.model_prediction_vector)
        self.roc_auc = metrics.auc(fpr, tpr)

        # plt.figure(1)
        # plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % self.roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        # plt.ylabel('True Positive Rate')
        plt.ylabel('Anteil richtig positiv', fontsize=15)
        # plt.xlabel('False Positive Rate')
        plt.xlabel('Anteil falsch positiv', fontsize=15)
        plt.title("CMS private work", loc="left", fontsize=15)
        plt.tick_params(labelsize=14.5)
        plt.savefig(save_path + self.name +"_roc.pdf")
        plt.clf()


    def get_ranges(self):
        if not self.data.binary_classification:
            max_ = [0.]*len(self.input_samples.samples)
            for ev in self.model_prediction_vector:
                for i,node in enumerate(ev):
                    if node>max_[i]:
                        max_[i]=node
            print("Max: ",max_)
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
        rank_path = self.save_path + "/absolute_weight_sum.csv"
        with open(rank_path, "w") as f:
            f.write("variable,weight_sum\n")
            for key, val in sorted(self.weight_dict.iteritems(), key = lambda (k,v): (v,k)):
                print("{:50s}: {}".format(key, val))
                f.write("{},{}\n".format(key,val))
        print("wrote weight ranking to "+str(rank_path))


    def factorial(self, array):
        return map(lambda x: math.factorial(x), array)


    def logarithmic(self, array):
        indices = [i for i, x in enumerate(array) if x <= 0]
        for i in indices:
            print("# DEBUG: logarithmic, entries: ", array[i])
        # print("# DEBUG: logarithmic, indices: ", indices)
        return map(lambda x: math.log(x), array)


    def binned_likelihood(self, bkg_binns, tg_binns, mu):
        '''Calculares sigma1 and sigma2 for asimov data set and makes a plot'''
        mu_range = 1.1
        save_path = self.path + "plot/"
        measured = bkg_binns + mu * tg_binns
        for i in measured:
            if i == 0:
                print("Bin with zero events")
        avoid_problems = bkg_binns - (mu_range-mu) * tg_binns
        #remove bins with no bkg events -> they will couse problems due to log
        indices = [i for i, x in enumerate(avoid_problems) if x <= 0]
        print("# DEBUG: binned_likelihood, indices: ", indices)
        measured = np.delete(measured, indices)
        bkg_binns = np.delete(bkg_binns, indices)
        tg_binns = np.delete(tg_binns, indices)
        # print("# DEBUG: binned_likelihood, indices of ==0: ", indices)
        # print("# DEBUG: binned_likelihood, measured.shape: ", measured.shape)
        # print("# DEBUG: bkg_binns.shape : ", bkg_binns.shape)
        # print("# DEBUG: factorial: ", self.factorial(measured))
        # print("# DEBUG: np.log(factorial): ", np.log(self.factorial(measured)))
        # print("# DEBUG: np.log(bkg_binns + mu*tg_binns): ", np.log(bkg_binns + mu*tg_binns))
        # print("# DEBUG: bevore sum: ", 2*(self.logarithmic(self.factorial(measured)) + bkg_binns + mu*tg_binns - bkg_binns*self.logarithmic(bkg_binns + mu*tg_binns)))
        minimum = np.sum(2*(self.logarithmic(self.factorial(measured)) + bkg_binns + mu*tg_binns - measured*self.logarithmic(measured)))
        # print("# DEBUG: binned_likelihood, minimum: ", minimum)
        nxvals = 51     #51/2 hard coded in interpolate
        mu_draw = np.linspace(mu-mu_range, mu+mu_range, nxvals, endpoint = True)
        loglike = np.array([])
        for i in range(0, mu_draw.shape[0]):        #better use while loglike < 2+y_min
            tmp = 2*(self.logarithmic(self.factorial(measured)) + bkg_binns + mu_draw[i]*tg_binns - measured*self.logarithmic(bkg_binns + mu_draw[i]*tg_binns))
            # if i < 4:
            #     print("+ ", self.logarithmic(self.factorial(measured)))
            #     print("+ ", bkg_binns + mu_draw[i]*tg_binns)
            #     print("- ", bkg_binns*self.logarithmic(bkg_binns + mu_draw[i]*tg_binns))
            loglike = np.append(loglike, np.sum(tmp)-minimum)
        #calculate 'sigma1' and 'sigma2'
        #binned likelihood function is invetable when seperated to left and right of its minimum
        # print("# DEBUG: binned_likelihood, loglike: ", loglike)
        # print("# DEBUG: binned_likelihood, maximum: ", np.amax(loglike))
        # print("# DEBUG: shapes in interpolate: ", loglike.shape, mu_draw.shape)
        intp_lefthand = interpolate.interp1d(loglike[:26], mu_draw[:26])
        intp_righthand = interpolate.interp1d(loglike[26:], mu_draw[26:])
        s1x = np.array([intp_lefthand(1), intp_righthand(1)])
        s2x = np.array([intp_lefthand(4), intp_righthand(4)])
        #real sigma1 and sigma2 values
        sigma1 = np.absolute(s1x-mu)
        sigma2 = np.absolute(s2x-mu)
        #sigma1 and sigma2 y value for plotting
        s1y = [1, 1]
        s2y = [4, 4]
        #plotting
        plt.xlabel(r'$\mu$', fontsize=15)
        plt.ylabel(r'$-2 \log L$', fontsize=15)
        plt.plot(mu_draw, loglike, 'k-')
        plt.xlim(left=mu_draw[0], right=mu_draw[-1])
        plt.axvline(x=mu, color='k', ls='--', ymin=0., ymax=np.amax(loglike))
        plt.plot(s1x, s1y, 'b-', label = r'$1\sigma=+{1}/-{0}$'.format(round(sigma1[0],3), round(sigma1[1],3)))
        plt.plot(s2x, s2y, 'r-', label = r'$2\sigma=+{1}/-{0}$'.format(round(sigma2[0],3), round(sigma2[1],3)))
        plt.legend(loc='best', fontsize = 'large')
        plt.title("CMS private work", loc="left", fontsize=15)
        plt.tick_params(labelsize=14.5)
        plt.savefig(save_path + self.name+ "_mu" + str(mu) +"_loglike.pdf")
        plt.clf()
        # print("# DEBUG: binned_likelihood, saved fig: ", save_path + self.name+ "_mu" + str(mu) +"_loglike.pdf")
        #to calculate real sigma1 and sigma2 and return it
        return sigma1, sigma2



    # --------------------------------------------------------------------
    # result plotting functions
    # --------------------------------------------------------------------
    def plot_metrics(self, privateWork = False):
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)

        #get saving path
        save_path = self.path + "plot/"

        ''' plot history of loss function and evaluation metrics '''
        metrics = ["acc"]
        if self.eval_metrics: metrics += self.eval_metrics

        # loop over metrics and generate matplotlib plot
        for metric in metrics:
            plt.clf()
            # get history of train and validation scores
            train_history = self.model_history[metric]
            val_history = self.model_history["val_"+metric]
            best_train = max(train_history)
            best_test = max(val_history)

            n_epochs = len(train_history)
            epochs = np.arange(1,n_epochs+1,1)
            print("Plotting frac")

            # plot histories
            plt.plot(epochs, train_history, "r-", label = "Trainingsdaten - Max: " + str(round(best_train, 3)))
            plt.plot(epochs, val_history, "b-", label = "Testdaten - Max: " + str(round(best_test, 3)))
            # if privateWork:
                # plt.title("CMS private work", loc = "left", fontsize = 16)

            # add title
            # title = self.categoryLabel
            # title = title.replace("\\geq", "$\geq$")
            # title = title.replace("\\leq", "$\leq$")
            # plt.title(title, loc = "right", fontsize = 16)

            # make it nicer
            # plt.grid()
            plt.xlabel("Epochen", fontsize=15)
            plt.ylabel("ARK Ereignisse", fontsize=15)
            plt.tick_params(labelsize=14.5)

            # add legend
            plt.legend(loc='lower right')
            # add CMS private work
            plt.title("CMS private work", loc="left", fontsize=15)

            # save
            # out_path = self.save_path + "/model_history_"+str(metric)+".pdf"
            plt.savefig(save_path + self.name + "_frac.pdf")
            plt.clf()
            # print("saved plot of "+str(metric)+" at "+str(out_path))




    def plot_outputNodes(self, log = False, printROC = False, signal_class = None,
                        privateWork = False,
                        nbins = 20, bin_range = [0.,1.]):

        ''' plot distribution in outputNodes '''
        plotNodes = plottingScripts.plotOutputNodes(
            data                = self.data,
            prediction_vector   = self.model_prediction_vector,
            event_classes       = self.event_classes,
            nbins               = nbins,
            bin_range           = bin_range,
            signal_class        = signal_class,
            event_category      = self.categoryLabel,
            plotdir             = self.plot_path,
            logscale            = log)

        plotNodes.plot(ratio = False, printROC = printROC, privateWork = privateWork)


    def plot_discriminators(self, log = False, printROC = False, privateWork = False,
                        signal_class = None, nbins = 18, bin_range = [0.1,1.]):

        ''' plot all events classified as one category '''
        plotDiscrs = plottingScripts.plotDiscriminators(
            data                = self.data,
            prediction_vector   = self.model_prediction_vector,
            event_classes       = self.event_classes,
            nbins               = nbins,
            bin_range           = bin_range,
            signal_class        = signal_class,
            event_category      = self.categoryLabel,
            plotdir             = self.plot_path,
            logscale            = log)

        plotDiscrs.plot(ratio = False, printROC = printROC, privateWork = privateWork)


    def plot_confusionMatrix(self, norm_matrix = True, privateWork = False, printROC = False):
        ''' plot confusion matrix '''
        plotCM = plottingScripts.plotConfusionMatrix(
            data                = self.data,
            prediction_vector   = self.model_prediction_vector,
            event_classes       = self.event_classes,
            event_category      = self.categoryLabel,
            plotdir             = self.save_path)

        plotCM.plot(norm_matrix = norm_matrix, privateWork = privateWork, printROC = printROC)

    def plot_closureTest(self, log = False, privateWork = False,
                        signal_class = None, nbins = 20, bin_range = [0.,1.]):
        ''' plot comparison between train and test samples '''

        bin_range = [1./self.data.n_output_neurons, 1.]
        closureTest = plottingScripts.plotClosureTest(
            data                = self.data,
            test_prediction     = self.model_prediction_vector,
            train_prediction    = self.model_train_prediction,
            event_classes       = self.event_classes,
            nbins               = nbins,
            bin_range           = bin_range,
            signal_class        = signal_class,
            event_category      = self.categoryLabel,
            plotdir             = self.plot_path,
            logscale            = log)

        closureTest.plot(ratio = False, privateWork = privateWork)

    def plot_eventYields(self, log = False, privateWork = False, signal_class = None):
        eventYields = plottingScripts.plotEventYields(
            data                = self.data,
            prediction_vector   = self.model_prediction_vector,
            event_classes       = self.event_classes,
            event_category      = self.categoryLabel,
            signal_class        = signal_class,
            plotdir             = self.save_path,
            logscale            = log)

        eventYields.plot(privateWork = privateWork)

    def plot_binaryOutput(self, log = False, privateWork = False, printROC = False,
                        nbins = 20, bin_range = [0.,1.], name = "Bin#ddot{a}rer Diskriminator"):

        binaryOutput = plottingScripts.plotBinaryOutput(
            data                = self.data,
            predictions         = self.model_prediction_vector,
            nbins               = nbins,
            bin_range           = bin_range,
            event_category      = self.categoryLabel,
            plotdir             = self.path + "plot/",
            pltname             = self.name,
            logscale            = log)

        bkg_hist, sig_hist = binaryOutput.plot(ratio = True, printROC = printROC, privateWork = privateWork, name = name)
        print("sigma: ", self.binned_likelihood(bkg_hist, sig_hist, 0))
        print("sigma: ", self.binned_likelihood(bkg_hist, sig_hist, 1))


def loadDNN(inputDirectory, outputDirectory):

    # get net config json
    configFile = inputDirectory+"/checkpoints/net_config.json"
    if not os.path.exists(configFile):
        sys.exit("config needed to load trained DNN not found\n{}".format(configFile))

    with open(configFile) as f:
        config = f.read()
    config = json.loads(config)

    # load samples
    input_samples = data_frame.InputSamples(config["inputData"])

    for sample in config["eventClasses"]:
        input_samples.addSample(sample["samplePath"], sample["sampleLabel"], normalization_weight = sample["sampleWeight"])

    if "binaryConfig" in config:
        input_samples.addBinaryLabel(signals = config["binaryConfig"]["signals"], bkg_target = config["binaryConfig"]["minValue"])
    print("shuffle seed: {}".format(config["shuffleSeed"]))
    # init DNN class
    dnn = DNN(
        save_path       = outputDirectory,
        input_samples   = input_samples,
        event_category  = config["JetTagCategory"],
        train_variables = config["trainVariables"],
        shuffle_seed    = config["shuffleSeed"]
        )

    # load the trained model
    dnn.load_trained_model(inputDirectory)


    return dnn
