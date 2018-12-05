#global imports
import rootpy.plotting as rp
import keras
import keras.models as models
import keras.layers as layer
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

# Limit gpu usage
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

import numpy as np
import pandas

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

import os

# local imports
import data_frame
import plot_configs.variable_binning as binning
import plot_configs.plotting_styles as pltstyle
import DNN_Architecture as Architecture


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



class DNN():
    def __init__(self, in_path, save_path,
                event_classes,
                event_category,
                train_variables,
                prenet_targets,
                batch_size = 5000,
                train_epochs = 500,
                early_stopping = 10,
                optimizer = None,
                loss_function = "categorical_crossentropy",
                test_percentage = 0.2,
                eval_metrics = None,
                additional_cut = None):

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

        # additional cut to be applied after variable norm
        self.additional_cut = additional_cut

        # load dataset
        self.data = self._load_datasets()
        out_path = self.save_path+"/checkpoints/"
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_file = out_path+"/variable_norm.csv"
        self.data.norm_csv.to_csv(out_file)
        print("saved variable norms at "+str(out_file))

        # dict with aachen architectures for sl analysis
        arch_cls = Architecture.Architecture()
        self.architecture = arch_cls.get_architecture(self.event_category)

         # optimizer for training
        if not(optimizer):
            self.optimizer = self.architecture["optimizer"]
        else:
            self.optimizer = optimizer

    def _load_datasets(self):
        ''' load dataset '''
        return data_frame.DataFrame(
            path_to_input_files = self.in_path,
            classes             = self.event_classes,
            event_category      = self.event_category,
            train_variables     = self.train_variables,
            prenet_targets      = self.prenet_targets,
            test_percentage     = self.test_percentage,
            norm_variables      = True,
            additional_cut      = self.additional_cut)


    def load_trained_model(self):
        ''' load an already trained model '''
        checkpoint_path = self.save_path + "/checkpoints/trained_main_net.h5py"

        self.main_net = keras.models.load_model(checkpoint_path)

        self.mainnet_eval = self.main_net.evaluate(
            self.data.get_test_data(as_matrix = True),
            self.data.get_test_labels())

        self.mainnet_predicted_vector = self.main_net.predict(
            self.data.get_test_data(as_matrix = True))

        self.predicted_classes = np.argmax( self.mainnet_predicted_vector, axis = 1)

        # save confusion matrix
        self.confusion_matrix = confusion_matrix(
            self.data.get_test_labels(as_categorical = False), self.predicted_classes)

        # print evaluations
        self.main_roc_score = roc_auc_score(self.data.get_test_labels(), self.mainnet_predicted_vector)
        print("mainnet test roc: {}".format(self.main_roc_score))
        if self.eval_metrics:
            print("mainnet test loss: {}".format(self.mainnet_eval[0]))
            for im, metric in enumerate(self.eval_metrics):
                print("mainnet test {}: {}".format(metric, self.mainnet_eval[im+1]))

    def predict_event_query(self, query ):
        events = self.data.get_full_df().query( query )
        print(str(events.shape[0]) + " events matched the query '"+str(query)+"'.")

        for index, row in events.iterrows():
            print("Index: "+str(index))
            print("evaluation:")
            print( self.main_net.predict( np.array([list(row.values)]) ) )
            print("-"*50)



    def build_default_model(self):
        ''' default Aachen-DNN model as used in the analysis '''
        K.set_learning_phase(True)

        number_of_input_neurons = self.data.n_input_neurons

        number_of_neurons_per_layer = self.architecture["prenet_layer"]
        dropout                     = self.architecture["Dropout"]
        batchNorm                   = self.architecture["batchNorm"]
        activation_function         = self.architecture["activation_function"]
        l2_regularization_beta      = self.architecture["L2_Norm"]

        # build pre net ===========================================================================
        Inputs = keras.layers.Input(
            shape = (self.data.n_input_neurons,),
            name = "input")

        X = Inputs
        self.layer_list = [X]

        # loop over dense layers
        for i, nNeurons in enumerate(number_of_neurons_per_layer):
            Dense = keras.layers.Dense(
                nNeurons,
                activation = activation_function,
                kernel_regularizer = keras.regularizers.l2(l2_regularization_beta),
                name = "Dense_"+str(i)
                )(X)
            self.layer_list.append( Dense )

            if dropout != 1:
                X = keras.layers.Dropout(dropout)(Dense)
            else:
                X = Dense

            if batchNorm:
                X = keras.layers.BatchNormalization()(Dense)
            else:
                X = Dense

        # generate output layer
        X = keras.layers.Dense(
            self.data.n_prenet_output_neurons,
            activation = "sigmoid",
            kernel_regularizer = keras.regularizers.l2(l2_regularization_beta)
            )(X)
        self.layer_list.append(X)

        # define model
        pre_net = models.Model(inputs = [Inputs], outputs = [X])
        pre_net.summary()

        # Make Parameters of first model untrainable
        for layer in pre_net.layers:
            layer.trainable = False

        # build main net ==========================================================================
        number_of_neurons_per_layer = self.architecture["prenet_layer"]

        # Create Input/conc layer for second NN
        conc_layer = keras.layers.concatenate(self.layer_list, axis = -1)
        Y = conc_layer

        # loop over dense layers
        for i, nNeurons in enumerate(number_of_neurons_per_layer):
            Y = keras.layers.Dense(
                nNeurons,
                activation = activation_function,
                kernel_regularizer = keras.regularizers.l2(l2_regularization_beta),
                name = "Dense_main_"+str(i)
                )(Y)

            if dropout != 1:
                Y = keras.layers.Dropout(dropout)(Y)

            if batchNorm:
                Y = keras.layers.BatchNormalization()(Y)

        # generate output layer
        Y = keras.layers.Dense(
            self.data.n_output_neurons,
            activation = "softmax",
            kernel_regularizer = keras.regularizers.l2(l2_regularization_beta),
	        name = "output"
            )(Y)

        # define model
        main_net = models.Model(inputs = [Inputs], outputs = [Y])
        main_net.summary()

        return pre_net, main_net


    def build_model(self, pre_net = None, main_net = None):
        ''' build a DNN model
            if none is specified use default model '''

        if pre_net == None or main_net == None:
            print("loading default models")
            pre_net, main_net = self.build_default_model()

        # set pre net trainable
        for layer in pre_net.layers:
            layer.trainable = True

        # compile prenet
        pre_net.compile(
            loss = self.architecture["prenet_loss"],
            optimizer = self.optimizer,
            metrics = self.eval_metrics)

        # ste pre net untrainable
        for layer in pre_net.layers:
            layer.trainable = False

        # compile main net
        main_net.compile(
            loss = self.architecture["mainnet_loss"],
            optimizer = self.optimizer,
            metrics = self.eval_metrics)

        # save compiled nets
        self.pre_net = pre_net
        self.main_net = main_net

        # save net information
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

        # checkpoint files
        cp_path = self.save_path + "/checkpoints/"
        if not os.path.exists(cp_path):
            os.makedirs(cp_path)

        # add early stopping if activated
        callbacks = None
        if self.early_stopping:
            callbacks = [EarlyStoppingByLossDiff(
                monitor = "loss",
                value = self.architecture["earlystopping_percentage"],
                min_epochs = 100,
                patience = 10,
                verbose = 1)]

        self.trained_pre_net = self.pre_net.fit(
            x = self.data.get_train_data(as_matrix = True),
            y = self.data.get_prenet_train_labels(),
            batch_size = self.architecture["batch_size"],
            epochs = self.train_epochs,
            shuffle = True,
            callbacks = callbacks,
            validation_split = 0.25,
            sample_weight = self.data.get_train_weights()
            )

        # set pre net as untrainable
        for layer in self.pre_net.layers:
            layer.trainable = False

        # save trained prenet model
        out_file = cp_path + "/trained_pre_net.h5py"
        self.pre_net.save(out_file)
        print("saved trained prenet model at "+str(out_file))

        prenet_config = self.pre_net.get_config()
        out_file = cp_path +"/trained_pre_net_config"
        with open(out_file, "w") as f:
            f.write( str(prenet_config))
        print("saved prenet model config at "+str(out_file))

        out_file = cp_path +"/trained_pre_net_weights.h5"
        self.pre_net.save_weights(out_file)
        print("wrote trained prenet weights to "+str(out_file))

        # add early stopping if activated
        callbacks = None
        if self.early_stopping:
            callbacks = [EarlyStoppingByLossDiff(
                monitor = "loss",
                value = self.architecture["earlystopping_percentage"],
                min_epochs = 100,
                patience = 20,
                verbose = 1)]

        # train main net
        self.trained_main_net = self.main_net.fit(
            x = self.data.get_train_data(as_matrix = True),
            y = self.data.get_train_labels(),
            batch_size = self.architecture["batch_size"],
            epochs = self.train_epochs,
            shuffle = True,
            callbacks = callbacks,
            validation_split = 0.25,
            sample_weight = self.data.get_train_weights()
            )

        # save trained model
        out_file = cp_path + "/trained_main_net.h5py"
        self.main_net.save(out_file)
        print("saved trained model at "+str(out_file))

        mainnet_config = self.main_net.get_config()
        out_file = cp_path + "/trained_main_net_config"
        with open(out_file, "w") as f:
            f.write( str(mainnet_config))
        print("saved model config at "+str(out_file))

        out_file = cp_path +"/trained_main_net_weights.h5"
        self.main_net.save_weights(out_file)
        print("wrote trained weights to "+str(out_file))

        # set model as non trainable
        for layer in self.pre_net.layers:
            layer.trainable = False
        self.pre_net.trainable = False

        for layer in self.main_net.layers:
            layer.trainable = False
        self.main_net.trainable = False

        K.set_learning_phase(False)

        out_file = cp_path + "/trained_main_net"
        sess = keras.backend.get_session()
        saver = tf.train.Saver()
        save_path = saver.save(sess, out_file)
        print("saved checkpoint files to "+str(out_file))


    def eval_model(self):
        ''' evaluate trained model '''

        # prenet evaluation
        self.prenet_eval = self.pre_net.evaluate(
            self.data.get_test_data(as_matrix = True),
            self.data.get_prenet_test_labels())

        # save history of eval metrics
        self.prenet_history = self.trained_pre_net.history

        # save predicitons
        self.prenet_predicted_vector = self.pre_net.predict(
            self.data.get_test_data(as_matrix = True) )

        # print evaluations
        if self.eval_metrics:
            print("prenet test loss: {}".format(self.prenet_eval[0]))
            for im, metric in enumerate(self.eval_metrics):
                print("prenet test {}: {}".format(metric, self.prenet_eval[im+1]))


        # main net evaluation
        self.mainnet_eval = self.main_net.evaluate(
            self.data.get_test_data(as_matrix = True),
            self.data.get_test_labels())

        # save history of eval metrics
        self.mainnet_history = self.trained_main_net.history

        # save predictions
        self.mainnet_predicted_vector = self.main_net.predict(
            self.data.get_test_data(as_matrix = True) )

        # save predicted classes with argmax
        self.predicted_classes = np.argmax( self.mainnet_predicted_vector, axis = 1)

        # save confusion matrix
        self.confusion_matrix = confusion_matrix(
            self.data.get_test_labels(as_categorical = False), self.predicted_classes)

        # print evaluations
        self.main_roc_score = roc_auc_score(self.data.get_test_labels(), self.mainnet_predicted_vector)
        print("mainnet test roc: {}".format(self.main_roc_score))
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
        pltstyle.init_plot_style()
        n_bins = 20
        bin_range = [0.,1.]

        for i, node_cls in enumerate(self.prenet_targets):
            # get outputs of class node
            out_values = self.prenet_predicted_vector[:,i]

            prenet_labels = self.data.get_prenet_test_labels()[:,i]

            sig_values = [out_values[k] for k in range(len(out_values)) if prenet_labels[k] == 1]
            bkg_values = [out_values[k] for k in range(len(out_values)) if prenet_labels[k] == 0]

            sig_weights = [self.data.get_lumi_weights()[k] for k in range(len(out_values)) if prenet_labels[k] == 1]
            bkg_weights = [self.data.get_lumi_weights()[k] for k in range(len(out_values)) if prenet_labels[k] == 0]

            bkg_sig_ratio = 1.*sum(bkg_weights)/sum(sig_weights)
            sig_weights = [w*bkg_sig_ratio for w in sig_weights]

            sig_label = "True"
            bkg_label = "False"

            sig_title = sig_label +"*{:.3f}".format(bkg_sig_ratio)

            # plot output
            bkg_hist = rp.Hist(n_bins, *bin_range, title = bkg_label)
            pltstyle.set_bkg_hist_style( bkg_hist, bkg_label)
            bkg_hist.fill_array( bkg_values, bkg_weights )

            sig_hist = rp.Hist(n_bins, *bin_range, title = sig_title)
            pltstyle.set_sig_hist_style( sig_hist, sig_label )
            sig_hist.fill_array( sig_values, sig_weights )

            stack = rp.HistStack( [bkg_hist], stacked = True, drawstyle = "HIST E1 X0")
            stack.SetMinimum(1e-4)


            canvas = pltstyle.init_canvas()


            rp.utils.draw([stack,sig_hist],
                xtitle = "prenet node {}".format(node_cls), ytitle = "Events", pad = canvas)
            if log: canvas.cd().SetLogy()

            legend = pltstyle.init_legend([bkg_hist, sig_hist])
            pltstyle.add_lumi(canvas)
            pltstyle.add_category_label(canvas, self.event_category)

            out_path = self.save_path + "/prenet_output_{}.pdf".format(node_cls)

            pltstyle.save_canvas(canvas,out_path)


    def plot_discriminators(self, log = False, cut_on_variable = None):
        ''' plot discriminators for output classes '''
        pltstyle.init_plot_style()

        nbins = 50
        bin_range = [0., 1.]

        # get some ttH specific info for plotting
        ttH_index = self.data.class_translation["ttHbb"]
        ttH_true_labels = self.data.get_ttH_flag()

        # apply cut to output node value if wanted
        if cut_on_variable:
            cut_class = cut_on_variable["class"]
            cut_value = cut_on_variable["val"]

            cut_index = self.data.class_translation[cut_class]
            cut_prediction = self.mainnet_predicted_vector[:,cut_index]

        # loop over discriminator nodes
        for i, node_cls in enumerate(self.event_classes):
            # get outputs of node
            out_values = self.mainnet_predicted_vector[:,i]

            # calculate node specific ROC value
            node_ROC = roc_auc_score(ttH_true_labels, out_values)

            # fill lists according to class
            bkg_hists = []
            weight_integral = 0

            # loop over all classes to fill hist according to predicted class
            for j, truth_cls in enumerate(self.event_classes):
                class_index = self.data.class_translation[truth_cls]

                # filter values per event class
                if cut_on_variable:
                    filtered_values = [ out_values[k] for k in range(len(out_values)) \
                        if self.data.get_test_labels(as_categorical = False)[k] == class_index \
                        and cut_prediction[k] <= cut_value]
                    filtered_weights = [ self.data.get_lumi_weights()[k] for k in range(len(out_values)) \
                        if self.data.get_test_labels(as_categorical = False)[k] == class_index \
                        and cut_prediction[k] <= cut_value]
                else:
                    filtered_values = [ out_values[k] for k in range(len(out_values)) \
                        if self.data.get_test_labels(as_categorical = False)[k] == class_index ]
                    filtered_weights = [ self.data.get_lumi_weights()[k] for k in range(len(out_values)) \
                        if self.data.get_test_labels(as_categorical = False)[k] == class_index ]


                if j == ttH_index:
                    # ttH signal
                    sig_values = filtered_values
                    sig_label = str(truth_cls)
                    sig_weights = filtered_weights
                else:
                    # background in this node
                    weight_integral += sum( filtered_weights )
                    hist = rp.Hist( nbins, *bin_range, title = str(truth_cls))
                    pltstyle.set_bkg_hist_style(hist, truth_cls)
                    hist.fill_array( filtered_values, filtered_weights )
                    bkg_hists.append(hist)

            # stack backgrounds
            bkg_stack = rp.HistStack( bkg_hists, stacked = True, drawstyle = "HIST E1 X0")
            bkg_stack.SetMinimum(1e-4)
            max_val = bkg_stack.GetMaximum()*1.3
            bkg_stack.SetMaximum(max_val)

            # plot signal
            weight_sum = sum(sig_weights)
            scale_factor = 1.*weight_integral/weight_sum
            sig_weights = [w*scale_factor for w in sig_weights]

            sig_title = sig_label + "*{:.3f}".format(scale_factor)
            sig_hist = rp.Hist( nbins, *bin_range, title = sig_title)
            pltstyle.set_sig_hist_style(sig_hist, sig_label)
            sig_hist.fill_array( sig_values, sig_weights)

            # creating canvas
            canvas = pltstyle.init_canvas()

            # drawing histograms
            rp.utils.draw([bkg_stack, sig_hist],
                xtitle = node_cls+" Discriminator", ytitle = "Events", pad = canvas)
            if log: canvas.cd().SetLogy()

            # creating legend
            legend = pltstyle.init_legend( bkg_hists+[sig_hist] )
            pltstyle.add_lumi(canvas)
            pltstyle.add_category_label(canvas, self.event_category)

            # add ROC value to plot
            pltstyle.add_ROC_value(canvas, node_ROC)

            # save canvas
            out_path = self.save_path + "/discriminator_{}.pdf".format(node_cls)
            pltstyle.save_canvas(canvas, out_path)

    def plot_classification(self, log = False):
        ''' plot all events classified as one category '''

        pltstyle.init_plot_style()
        nbins = 20
        bin_range = [0., 1.]

        ttH_index = self.data.class_translation["ttHbb"]
        # loop over discriminator nodes
        for i, node_cls in enumerate(self.event_classes):
            node_index = self.data.class_translation[node_cls]

            # get outputs of node
            out_values = self.mainnet_predicted_vector[:,i]

            # fill lists according to class
            bkg_hists = []
            weight_integral = 0

            # loop over all classes to fill hist according to predicted class
            for j, truth_cls in enumerate(self.event_classes):
                class_index = self.data.class_translation[truth_cls]

                # filter values per event class
                filtered_values = [ out_values[k] for k in range(len(out_values)) \
                    if self.data.get_test_labels(as_categorical = False)[k] == class_index \
                        and self.predicted_classes[k] == node_index ]
                filtered_weights = [ self.data.get_lumi_weights()[k] for k in range(len(out_values)) \
                    if self.data.get_test_labels(as_categorical = False)[k] == class_index \
                        and self.predicted_classes[k] == node_index ]

                if j == ttH_index:
                    # signal in this node
                    sig_values = filtered_values
                    sig_label = str(truth_cls)
                    sig_weights = filtered_weights
                else:
                    # background in this node
                    weight_integral += sum(filtered_weights)
                    hist = rp.Hist(nbins, *bin_range, title = str(truth_cls))
                    pltstyle.set_bkg_hist_style(hist, truth_cls)
                    hist.fill_array( filtered_values, filtered_weights )
                    bkg_hists.append(hist)

            # stack backgrounds
            bkg_stack = rp.HistStack(bkg_hists, stacked = True, drawstyle = "HIST E1 X0")
            bkg_stack.SetMinimum(1e-4)
            max_val = bkg_stack.GetMaximum()*1.3
            bkg_stack.SetMaximum(max_val)

            # plot signal
            weight_sum = sum(sig_weights)
            scale_factor = 1.*weight_integral/weight_sum
            sig_weights = [w*scale_factor for w in sig_weights]

            sig_title = sig_label + "*{:.3f}".format(scale_factor)
            sig_hist = rp.Hist(nbins, *bin_range, title = sig_title)
            pltstyle.set_sig_hist_style(sig_hist, sig_label)
            sig_hist.fill_array(sig_values, sig_weights)

            # creatin canvas

            canvas = pltstyle.init_canvas()

            # drawing hists
            rp.utils.draw([bkg_stack, sig_hist],
                xtitle = "Events predicted as "+node_cls, ytitle = "Events", pad = canvas)
            if log: canvas.cd().SetLogy()

            # legend
            legend = pltstyle.init_legend( bkg_hists+[sig_hist] )
            pltstyle.add_lumi(canvas)
            pltstyle.add_category_label(canvas, self.event_category)
            print("S/B = {}".format(weight_sum/weight_integral))
            # save
            out_path = self.save_path + "/predictions_{}.pdf".format(node_cls)

            pltstyle.save_canvas(canvas, out_path)

    def plot_class_differences(self, log = False):

        pltstyle.init_plot_style()

        nbins = 20
        bin_range = [0.,1.]


        # loop over discriminator nodes
        for i, node_cls in enumerate(self.event_classes):
            node_index = self.data.class_translation[node_cls]

            # get outputs of node
            node_values = self.mainnet_predicted_vector[:,i]
            filtered_node_values = np.array([node_values[k] for k in range(len(node_values)) \
                if self.predicted_classes[k] == node_index])

            filtered_weights = [ self.data.get_lumi_weights()[k] for k in range(len(node_values)) \
                if self.predicted_classes[k] == node_index]

            histograms = []
            first = True
            max_val = 0
            # loop over other nodes and get those predictions
            for j, other_cls in enumerate(self.event_classes):
                if i == j: continue
                other_index = self.data.class_translation[other_cls]

                other_values = self.mainnet_predicted_vector[:,j]
                filtered_other_values = np.array([other_values[k] for k in range(len(other_values)) \
                    if self.predicted_classes[k] == node_index])

                # get difference of predicted node value and other value
                diff_values = (filtered_node_values - filtered_other_values)/filtered_node_values

                hist = rp.Hist(nbins, *bin_range, title = str(other_cls)+" node", drawstyle = "HIST E1 X0")
                pltstyle.set_sig_hist_style(hist, other_cls)
                hist.fill_array(diff_values, filtered_weights)
                if hist.GetMaximum() > max_val: max_val = hist.GetMaximum()

                if first:
                    stack = rp.HistStack([hist], stacked = True)
                    first_hist = hist
                    first = False
                else:
                    histograms.append(hist)

            # create canvas
            canvas = pltstyle.init_canvas()
            # drawing hists
            stack.SetMaximum(max_val*1.3)
            rp.utils.draw([stack]+histograms, pad = canvas,
                xtitle = "relative difference ("+str(node_cls)+" - X_node)/"+str(node_cls), ytitle = "Events")
            if log: canvas.cd().SetLogy()

            # legend
            legend = pltstyle.init_legend( [first_hist]+histograms )
            pltstyle.add_lumi(canvas)
            pltstyle.add_category_label(canvas, self.event_category)

            # save
            out_path = self.save_path + "/node_differences_{}.pdf".format(node_cls)
            pltstyle.save_canvas(canvas, out_path)


    def plot_input_output_correlation(self, plot = True):

        # get input variables from test set TODO get them unnormed
        input_data = self.data.get_test_data(as_matrix = False, normed = False)

        # initialize empty dataframe
        df = pandas.DataFrame()
        plt.figure(figsize = [10,10])

        # correlation plot path
        plt_path = self.save_path + "/correlations"
        if not os.path.exists(plt_path):
            os.makedirs(plt_path)

        # loop over classes
        for i_cls, cls in enumerate(self.event_classes):

            # get predictions for current output note
            pred_values = self.mainnet_predicted_vector[:,i_cls]

            # correlation values for class
            corr_values = {}

            # loop over input variables
            for i_var, var in enumerate(self.train_variables):
                # scatter plot:
                # x-axis: input variable value
                # y-axis: predicted discriminator output

                var_values = input_data[var].values

                assert( len(var_values) == len(pred_values) )


                # calculate correlation value
                correlation = np.corrcoef(var_values, pred_values)[0][1]
                print("correlation between {} and {}: {}".format(
                    cls, var, correlation))

                corr_values[var] = correlation

                if plot:
                    plt.hist2d(var_values, pred_values,
                        bins = [min(binning.binning[var]["nbins"],20), 20],
                        weights = self.data.get_lumi_weights(),
                        norm = LogNorm(),
                        cmap = "RdBu")
                    plt.colorbar()


                    # write correlation value on plot
                    plt.title( correlation, loc = "left")
                    plt.xlabel(var)
                    plt.ylabel(cls+"_predicted")

                    out_name = plt_path + "/correlation_{}_{}.pdf".format(cls,var)
                    plt.savefig(out_name.replace("[","_").replace("]",""))
                    plt.clf()


            # save correlation value to dataframe
            df[cls] = pandas.Series( corr_values )

        # save dataframe of correlations
        out_path = self.save_path + "/correlation_matrix.h5"
        df.to_hdf(out_path, "correlations")
        print("saved correlation matrix at "+str(out_path))

    def plot_output_output_correlation(self, plot = False):
        corr_path = self.save_path + "/output_correlations/"
        if not os.path.exists(corr_path):
            os.makedirs(corr_path)

        correlation_matrix = []
        for i_cls, xcls in enumerate(self.event_classes):
            correlations = []
            xvalues = self.mainnet_predicted_vector[:,i_cls]

            for j_cls, ycls in enumerate(self.event_classes):
                yvalues = self.mainnet_predicted_vector[:,j_cls]

                corr = np.corrcoef( xvalues, yvalues)[0][1]
                print("correlation between {} and {}: {}".format(xcls, ycls, corr))

                correlations.append(corr)

                if plot and i_cls < j_cls:
                    plt.clf()
                    plt.hist2d( xvalues, yvalues, bins = [20, 20],
                        weights = self.data.get_lumi_weights(),
                        norm = LogNorm(),
                        cmap = "RdBu")
                    plt.colorbar()

                    plt.title("corr = {}".format(corr), loc = "left")
                    plt.title(self.event_category, loc = "right")

                    plt.xlabel(xcls+" output node")
                    plt.ylabel(ycls+" output node")

                    out_name = corr_path + "/correlation_{}_{}.pdf".format(xcls, ycls)
                    plt.savefig(out_name)

            correlation_matrix.append(correlations)

        # plot correlation matrix
        n_classes = len(self.event_classes)

        x = np.arange(0, n_classes+1, 1)
        y = np.arange(0, n_classes+1, 1)

        xn, yn = np.meshgrid(x,y)

        plt.clf()
        plt.figure(figsize = [10,10])
        plt.pcolormesh(xn, yn, correlation_matrix, vmin = -1, vmax = 1)
        plt.colorbar()

        plt.xlim(0, n_classes)
        plt.ylim(0, n_classes)

        plt.xlabel("output nodes")
        plt.ylabel("output nodes")

        plt.title(self.event_category, loc = "right")

        # add textlabel
        for yit in range(n_classes):
            for xit in range(n_classes):
                plt.text(xit+0.5,yit+0.5,
                    "{:.3f}".format(correlation_matrix[yit][xit]),
                    horizontalalignment = "center",
                    verticalalignment = "center")

        ax = plt.gca()
        ax.set_xticks( np.arange((x.shape[0]-1))+0.5, minor = False)
        ax.set_yticks( np.arange((y.shape[0]-1))+0.5, minor = False)

        ax.set_xticklabels(self.event_classes)
        ax.set_yticklabels(self.event_classes)

        ax.set_aspect("equal")

        out_path = self.save_path + "/output_correlation.pdf"
        plt.savefig(out_path)
        print("saved output correlation at "+str(out_path))
        plt.clf()



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

        plt.title("ROC-AUC value: {:.4f}".format(self.main_roc_score), loc = "left")


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

        out_path = self.save_path + "/confusion_matrix.pdf"
        plt.savefig(out_path)
        print("saved confusion matrix at "+str(out_path))
        plt.clf()
