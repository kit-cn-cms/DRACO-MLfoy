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
import keras.optimizers as optimizers
import keras.models as models
import keras.layers as layer
from keras import backend as K
import pandas as pd

# Limit gpu usage
import tensorflow as tf

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
            evenSel            = None):

        # save some information
        # list of samples to load into dataframe
        self.input_samples = input_samples

        # norm variables or not
        self.norm_variables = norm_variables

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
        )

    def _load_architecture(self, config):
        ''' load the architecture configs '''

        # define default network configuration
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
                X = keras.layers.LeakyReLU(alpha=0.1)(X)

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

    def save_model(self, argv, execute_dir):
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
        self.model.save(out_file)
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

        # save checkpoint files (needed for c++ implementation)
        out_file = self.cp_path + "/trained_model"
        saver = tf.train.Saver()
        sess = K.get_session()
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
#                    [propagated_weights[i-1][j]*weight_layers[index][j] for j in range(len(weight_layers[index]))]
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
        

    def get_variations(self):
        if not os.path.exists(self.save_path + "/variations/"):
            os.makedirs(self.save_path + "/variations/")
        import matplotlib.pyplot as plt

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

            plt.grid()
            plt.legend()
            title = self.category_label
            title = title.replace("\\geq", "$\geq$")
            title = title.replace("\\leq", "$\leq$")
            plt.title(title, loc = "right", fontsize = 16)
            plt.xlabel(v.replace("_","\_"), fontsize = 16)
            plt.ylabel("node output", fontsize = 16)
            plt.xlim([-2,2])
            plt.ylim(yrange)
            plt.tight_layout()
            outpath = self.save_path + "/variations/"+str(v)+".pdf"
            plt.savefig(outpath)
            plt.savefig(outpath.replace(".pdf",".png"))
            print("plot saved at {}".format(outpath))
            
    def get_binary_variations(self):
        ''' make sure to only use when binary classification is done '''
        if not os.path.exists(self.save_path + "/variations/"):
            os.makedirs(self.save_path + "/variations/")
        import matplotlib.pyplot as plt

        print("making plots for input feature variations")
        for i, v in enumerate(self.train_variables):

            test_values = np.linspace(-2,2,500)
            testset = np.array([
                np.array([0 if not j==i else k for j in range(len(self.train_variables))])
                for k in test_values])

            predictions = self.model.predict(testset)

            yrange = [0.3, 0.7]
            
            
            plt.clf()
            plt.plot([-2,2],[1./len(self.event_classes),1./len(self.event_classes)], "-", color = "black")
            plt.plot([0.,0.],yrange, "-", color = "black")

            #for node in enumerate(self.event_classes):
            node = self.event_classes
            plt.plot(test_values, predictions[:,0], "-", linewidth = 2, label = "output node")
            #plt.plot(test_values, 1-predictions[:,0], "-", linewidth = 2, label = node[1]+" node")
                
            plt.grid()
            plt.legend()
            title = self.category_label
            title = title.replace("\\geq", "$\geq$")
            title = title.replace("\\leq", "$\leq$")
            plt.title(title, loc = "right", fontsize = 16)
            plt.xlabel(v.replace("_","\_"), fontsize = 16)            #not yet the best solution for string processing
            plt.ylabel("node output", fontsize = 16)
            plt.xlim([-2,2])
            plt.ylim(yrange) 
            plt.tight_layout()
            outpath = self.save_path + "/variations/"+str(v)+".pdf"
            plt.savefig(outpath)
            plt.savefig(outpath.replace(".pdf",".png"))
            print("plot saved at {}".format(outpath))
            

            
    # --------------------------------------------------------------------
    # result plotting functions
    # --------------------------------------------------------------------
    def plot_metrics(self, privateWork = False):
        import matplotlib.pyplot as plt
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
#            plt.ylim(ymin=0.)

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

        plotDiscrs.plot(ratio = False, printROC = printROC, privateWork = privateWork)

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

        binaryOutput.plot(ratio = False, printROC = printROC, privateWork = privateWork, name = name)


def loadDNN(inputDirectory, outputDirectory, binary = False, signal = None, binary_target = None, total_weight_expr = 'x.Weight_XS * x.Weight_CSV * x.Weight_GEN_nom', category_cutString = None,
category_label= None):

    # get net config json
    configFile = inputDirectory+"/checkpoints/net_config.json"
    if not os.path.exists(configFile):
        sys.exit("config needed to load trained DNN not found\n{}".format(configFile))

    with open(configFile) as f:
        config = f.read()
    config = json.loads(config)


    # load samples
    input_samples = data_frame.InputSamples(config["inputData"])

    if binary:
        input_samples.addBinaryLabel(signal, binary_target)

    for sample in config["eventClasses"]:
        input_samples.addSample(sample["samplePath"], sample["sampleLabel"], normalization_weight = sample["sampleWeight"], total_weight_expr = total_weight_expr)

    print("shuffle seed: {}".format(config["shuffleSeed"]))
    # init DNN class
    dnn = DNN(
      save_path       = outputDirectory,
      input_samples   = input_samples,
      category_name  = config["JetTagCategory"],
      train_variables = config["trainVariables"],
      shuffle_seed    = config["shuffleSeed"],
    )



    # load the trained model
    dnn.load_trained_model(inputDirectory)
#    dnn.predict_event_query()

    return dnn
