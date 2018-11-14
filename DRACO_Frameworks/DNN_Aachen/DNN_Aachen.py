#global imports
import keras
import keras.models as models
import keras.layers as layer
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import os

# Limit gpu usage
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import roc_auc_score


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# local imports
import data_frame
from Network_architecture import architecture

class DNN():
    def __init__(self, in_path, save_path,
                event_classes, 
                event_category,
                train_variables, 
                prenet_targets,
                batch_size =500,
                train_epochs = 500,
                early_stopping = 5,
                optimizer = None,
                loss_function = "categorical_crossentropy",
                test_percentage = 0.2,
                eval_metrics = None):

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

        # loss function for training
        self.loss_function = loss_function
        # additional metrics for evaluation of training process
        self.eval_metrics = eval_metrics

        # load dataset
        self.data= self._load_datasets()

        # dict with aachen architectures for sl analysis
        architecture_1 = architecture()
        self.architecture_dic = architecture_1.get_architecture(self.event_category)

         # optimizer for training
        if not(optimizer):
            self.optimizer = self.architecture_dic["optimizer"]
        else:
            self.optimizer = optimizer

    def _get_color_dict(self):
        return {
            "ttHbb": "blue",
            "ttlf":  "salmon",
            "ttcc":  "tomato",
            "ttbb":  "brown",
            "tt2b":  "darkred",
            "ttb":   "red"
            }

    def _load_datasets(self):
        ''' load dataset '''
        return data_frame.DataFrame(
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


        number_of_neurons_per_layer = self.architecture_dic["prenet_layer"]
        dropout                     = self.architecture_dic["Dropout"]
        activation_function         = self.architecture_dic["activation_function"]
        l2_regularization_beta      = self.architecture_dic["L2_Norm"]
    
        # prenet
        Inputs = keras.layers.Input( shape = (self.data.n_input_neurons,) )

        X = Inputs
        self.layer_list = [X]
        for i, nNeurons in enumerate(number_of_neurons_per_layer):
            Dense = keras.layers.Dense(nNeurons, activation = activation_function,
                                kernel_regularizer = keras.regularizers.l2(l2_regularization_beta),
                                name = "Dense_"+str(i))(X)

            self.layer_list.append( Dense )
            if dropout != 1: 
                X = keras.layers.Dropout( dropout )(Dense)
            else:
                X= Dense 
        
        X = keras.layers.Dense(self.data.n_prenet_output_neurons,
                activation = "sigmoid",
                kernel_regularizer = keras.regularizers.l2(l2_regularization_beta))(X)
        self.layer_list.append(X)

        pre_net = models.Model(inputs = [Inputs], outputs = [X])
        pre_net.summary()

        # compile and fit here?

        # Make Parameters of first model untrainable
        for layer in pre_net.layers:
            layer.trainable = False

        # ---------------
        # main net
        number_of_neurons_per_layer = self.architecture_dic["prenet_layer"]      

        # Create Input/conc layer for second NN
        conc_layer = keras.layers.concatenate(self.layer_list, axis = -1)

        Y = conc_layer

        for i, nNeurons in enumerate(number_of_neurons_per_layer):
            Y = keras.layers.Dense(nNeurons, activation = activation_function,
                            kernel_regularizer=keras.regularizers.l2(l2_regularization_beta),
                            name = "Dense_main_"+str(i))(Y)

            if dropout != 1:
                Y = keras.layers.Dropout(dropout)(Y)

        Y = keras.layers.Dense(self.data.n_output_neurons,
                activation = "softmax",
                kernel_regularizer=keras.regularizers.l2(l2_regularization_beta))(Y)


        main_net = models.Model(inputs = [Inputs], outputs = [Y])
        main_net.summary()

        return pre_net, main_net


    def build_model(self, pre_net = None, main_net = None):
        ''' build a DNN model
            if none is specified use default model '''

        if pre_net == None or main_net == None:
            print("loading default models")
            pre_net, main_net = self.build_default_model()

        for layer in pre_net.layers:
            layer.trainable = True
        # compile models
        pre_net.compile(
            loss = self.loss_function,
            optimizer = self.optimizer,
            metrics = self.eval_metrics)

        for layer in pre_net.layers:
            layer.trainable = False
        main_net.compile(
            loss = self.loss_function,
            optimizer = self.optimizer,
            metrics = self.eval_metrics)
            
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
            sample_weight = self.data.get_train_weights()
            )

        for layer in self.pre_net.layers:
            layer.trainable = False
        # save trained model
        out_file = self.save_path + "/trained_pre_net.h5py"
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
            sample_weight = self.data.get_train_weights()
            )

        # save trained model
        out_file = self.save_path + "/trained_main_net.h5py"
        self.main_net.save(out_file)
        print("saved trained model at "+str(out_file))

        mainnet_config = self.main_net.get_config()
        out_file = self.save_path + "/trained_main_net_config"
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
            self.data.get_test_data(as_matrix = True),
            self.data.get_prenet_test_labels())

        self.prenet_history = self.trained_pre_net.history

        self.prenet_predicted_vector = self.pre_net.predict( self.data.get_test_data(as_matrix = True) )

        print("prenet test roc:  {}".format(
            roc_auc_score(self.data.get_prenet_test_labels(), self.prenet_predicted_vector)))
        if self.eval_metrics: 
            print("prenet test loss: {}".format(self.prenet_eval[0]))
            for im, metric in enumerate(self.eval_metrics):
                print("prenet test {}: {}".format(metric, self.prenet_eval[im+1]))



        # main net evaluation
        self.mainnet_eval = self.main_net.evaluate(
            self.data.get_test_data(as_matrix = True),
            self.data.get_test_labels())

        self.mainnet_history = self.trained_main_net.history

        self.mainnet_predicted_vector = self.main_net.predict( self.data.get_test_data(as_matrix = True) )

        self.predicted_classes = np.argmax( self.mainnet_predicted_vector, axis = 1)
    
        self.confusion_matrix = confusion_matrix(
            self.data.get_test_labels(as_categorical = False), self.predicted_classes)

        print("mainnet test roc:  {}".format(
            roc_auc_score(self.data.get_test_labels(), self.mainnet_predicted_vector)))
        if self.eval_metrics: 
            print("mainnet test loss: {}".format(self.mainnet_eval[0]))
            for im, metric in enumerate(self.eval_metrics):
                print("mainnet test {}: {}".format(metric, self.mainnet_eval[im+1]))


        

    # --------------------------------------------------------------------
    # result plotting functions
    # --------------------------------------------------------------------

    def plot_metrics(self):
        ''' plot history of loss function and evaluation metrics '''

        metrics = ["loss"]
        if self.eval_metrics: metrics += self.eval_metrics

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



    def plot_prenet_nodes(self, log = False):
        ''' plot prenet nodes '''

        for i, node_cls in enumerate(self.prenet_targets):       
            # get outputs of class node
            out_values = self.prenet_predicted_vector[:,i]

            prenet_labels = self.data.get_prenet_test_labels()[:,i]

            sig_values = [out_values[k] for k in range(len(out_values)) if prenet_labels[k] == 1]
            bkg_values = [out_values[k] for k in range(len(out_values)) if prenet_labels[k] == 0]

            sig_weights = [self.data.get_test_weights()[k] for k in range(len(out_values)) if prenet_labels[k] == 1]
            bkg_weights = [self.data.get_test_weights()[k] for k in range(len(out_values)) if prenet_labels[k] == 0]

            bkg_sig_ratio = 1.*sum(bkg_weights)/sum(sig_weights)
            sig_weights = [w*bkg_sig_ratio for w in sig_weights]

            sig_label = "True"
            bkg_label = "False"

            sig_label += "*{:.3f}".format(bkg_sig_ratio)

            # plot output
            plt.clf()
            plt.figure(figsize = [15,10])
            
            plt.hist( bkg_values, histtype = "stepfilled", bins = 15, range = [0.,1.], label = bkg_label, log = log, weights = bkg_weights)
            plt.hist( sig_values, histtype = "step", bins = 15, range = [0.,1.], label = sig_label, log = log, weights = sig_weights, lw = 2)

            plt.legend()
            plt.grid()
            plt.xlabel("output of prenet node")
            plt.title("output for {}".format(node_cls), loc = "right")
            plt.title("CMS private work", loc = "left")

            out_path = self.save_path + "/prenet_output_{}.pdf".format(node_cls)

            plt.savefig(out_path)
            print("plot for prenet output of {} saved at {}".format(node_cls, out_path))

            plt.clf()


    def plot_classification_nodes(self, log = False):
        ''' plot discriminators for output classes '''

        for i, node_cls in enumerate(self.event_classes):
            # get outputs of class node
            out_values = self.mainnet_predicted_vector[:,i]

            bkg_values = []
            bkg_labels = []
            bkg_weights = []
            bkg_colors = []
            n_bkg_evts = 0
            n_sig_evts = 0

            # fill lists according to class
            for j, truth_cls in enumerate(self.event_classes):
                class_index = self.data.class_translation[truth_cls]
                # filter values per event class
                filtered_values = [ out_values[k] for k in range(len(out_values)) if self.data.get_test_labels(as_categorical = False)[k] == class_index ]
                filtered_weights = [ self.data.get_test_weights()[k] for k in range(len(out_values)) if self.data.get_test_labels(as_categorical = False)[k] == class_index ]
                if i == j:
                    # signal in this node
                    sig_values = filtered_values 
                    sig_label = str(truth_cls)
                    sig_weights = filtered_weights
                    sig_color = self._get_color_dict()[truth_cls]
                    n_sig_evts += len(filtered_values)
                else:
                    # background in this node
                    bkg_values.append(filtered_values)
                    bkg_labels.append(str(truth_cls))
                    bkg_weights.append(filtered_weights)
                    bkg_colors.append( self._get_color_dict()[truth_cls] )
                    n_bkg_evts += len(filtered_values)
            
            if n_sig_evts == 0: continue
            
            # plot discriminator output
            plt.clf()
            plt.figure(figsize = [15,10])
        
            # stack backgrounds
            plt.hist( bkg_values, stacked = True, histtype = "stepfilled", bins = 15, range = [0.,1.], label = bkg_labels, log = log, weights = bkg_weights, color = bkg_colors)

            # get weights for signal
            bkg_sig_ratio = 1.* sum([sum(ws) for ws in bkg_weights])/sum(sig_weights)
            sig_weights = [w*bkg_sig_ratio for w in sig_weights]
            sig_label += "*{:.3f}".format(bkg_sig_ratio)

            # plot signal shape
            plt.hist( sig_values, histtype = "step", bins = 15, range = [0.,1.], label = sig_label, log = log, weights = sig_weights, lw = 2, color = sig_color)

            plt.legend()
            plt.grid()
            plt.xlabel("discriminator output")
            plt.title("discriminator for {}".format(node_cls), loc = "right")
            plt.title("CMS private work", loc = "left")
            

            out_path = self.save_path + "/discriminator_{}.pdf".format(node_cls)
        
            plt.savefig(out_path)
            print("plot for discriminator of {} saved at {}".format(node_cls, out_path))

            plt.clf()



    def plot_input_output_correlation(self):
        return

    def plot_output_output_correlation(self):
        return

    def plot_confusion_matrix(self, norm_matrix = True):
        ''' generate confusion matrix '''
        n_classes = self.confusion_matrix.shape[0]

        # norm confusion matrix if wanted
        if norm_matrix:
            cm = np.empty( (n_classes, n_classes), dtype = np.float64 )
            for yit in range(n_classes):
                evt_sum = float(sum(self.confusion_matrix[yit,:]))
                for xit in range(n_classes):
                    cm[yit,xit] = self.confusion_matrix[yit,xit]/evt_sum

            self.confusion_matrix = cm

        plt.clf()

        plt.figure( figsize = [10,10])

        minimum = np.min( self.confusion_matrix )/(np.pi**2.0 * np.exp(1.0)**2.0)
        maximum = np.max( self.confusion_matrix )*(np.pi**2.0 * np.exp(1.0)**2.0)

        x = np.arange(0, n_classes+1, 1)
        y = np.arange(0, n_classes+1, 1)

        xn, yn = np.meshgrid(x,y)

        plt.pcolormesh(xn, yn, self.confusion_matrix,
            norm = LogNorm( vmin = max(minimum, 1e-6), vmax = min(maximum,1.) ))
        plt.colorbar()

        plt.xlim(0, n_classes)
        plt.ylim(0, n_classes)

        plt.xlabel("Predicted")
        plt.ylabel("True")
    
        # add textlabel
        for yit in range(n_classes):
            for xit in range(n_classes):
                plt.text( 
                    xit+0.5, yit+0.5,
                    "{:.3f}".format(self.confusion_matrix[yit, xit]),
                    horizontalalignment = "center",
                    verticalalignment = "center")
        
        plt_axis = plt.gca()
        plt_axis.set_xticks(np.arange( (x.shape[0] -1)) + 0.5, minor = False )
        plt_axis.set_yticks(np.arange( (y.shape[0] -1)) + 0.5, minor = False )

        plt_axis.set_xticklabels(self.data.classes)
        plt_axis.set_yticklabels(self.data.classes)

        plt_axis.set_aspect("equal")
        plt.tight_layout()

        out_path = self.save_path + "/confusion_matrix.pdf"
        plt.savefig(out_path)
        print("saved confusion matrix at "+str(out_path))
        plt.clf()       
















