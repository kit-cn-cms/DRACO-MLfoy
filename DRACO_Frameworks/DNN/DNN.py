#global imports
import rootpy.plotting as rp
import keras
import keras.models as models
import keras.layers as layer
from keras import backend as K
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

import pandas

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


import data_frame
import plot_configs.variable_binning as binning
import plot_configs.plotting_styles as ps
import DNN_Architecture as Architecture


class DNN():

    def __init__(self, in_path, save_path,
                event_classes,
                event_category,
                train_variables,
                batch_size = 5000,
                train_epochs = 500,
                early_stopping = 10,
                optimizer = None,
                loss_function = "categorical_crossentropy",
                test_percentage = 0.2,
                eval_metrics= None):

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

        # load data set
        self.data = self._load_datasets()
        out_file = self.save_path + "/variable_norm.csv"
        print("saved variabe norms at " + str(out_file))

        # architecture of the aarchen net
        arch_cls = Architecture.Architecture()
        self.architecture = arch_cls.get_architecture(self.event_category)

        # optimizer for training
        if not(optimizer):
            self.optimizer = self.architecture["optimizer"]
        else:
            self.optimizer = optimizer

    def _load_datasets(self):
        ''' load data set '''

        return data_frame.DataFrame(
            path_to_input_files = self.in_path,
            classes             = self.event_classes,
            event_category      = self.event_category,
            train_variables     = self.train_variables,
            test_percentage     = self.test_percentage,
            norm_variables      = True)


    def build_default_model(self):

        dropout                     = self.architecture["Dropout"]
        batchNorm                   = self.architecture["batchNorm"]
        activation_function         = self.architecture["activation_function"]
        l2_regularization_beta      = self.architecture["L2_Norm"]
        number_of_input_neurons = self.data.n_input_neurons
        number_of_neurons_per_layer = self.architecture["prenet_layer"]

        # define model
        model = models.Sequential()
        # add input layer
        model.add(layer.Dense(
                    100,
                    input_dim = number_of_input_neurons,
                    activation = activation_function,
                    kernel_regularizer = keras.regularizers.l2(l2_regularization_beta)))

        # loop over all dens layers
        for i in number_of_neurons_per_layer:
            model.add(layer.Dense(
                    i,
                    activation = activation_function,
                    kernel_regularizer = keras.regularizers.l2(l2_regularization_beta)))


        # create output layer
        model.add(layer.Dense(self.data.n_output_neurons,
                    activation = "softmax",
                    kernel_regularizer = keras.regularizers.l2(l2_regularization_beta)))

        return model

    def build_model(self, model = None):
        ''' build a DNN model
            if none is epecified use default model '''

        if model == None:
            print("Loading default model")
            model = self.build_default_model()

        # compile the model
        model.compile(
            loss = self.architecture["mainnet_loss"],
            optimizer = self.optimizer,
            metrics = self.eval_metrics)

        # save the model
        self.model = model

        # save net information
        out_file = self.save_path+"/model_summmary.yml"
        yml_model = self.model.to_yaml()
        with open(out_file, "w") as f:
            f.write(yml_model)



    def train_model(self):
        ''' train the model '''

        # add early stopping if activated
        callbacks = None
        if self.early_stopping:
            callbacks = [keras.callbacks.EarlyStopping(
                monitor = "val_loss",
                patience = self.early_stopping)]


        self.trained_model = self.model.fit(
            x = self.data.get_train_data(as_matrix = True),
            y = self.data.get_train_labels(),
            batch_size = self.architecture["batch_size"],
            epochs = self.train_epochs,
            shuffle = True,
            callbacks = callbacks,
            validation_split = 0.2,
            sample_weight = self.data.get_train_weights())

        '''
        # save trained model
        out_file = cp_path + "/trained_model.h5py"
        self.model.save(out_file)
        print("saved trained model at "+str(out_file))

        model_config = self.model.get_config()
        out_file = cp_path +"/trained_model_config"
        with open(out_file, "w") as f:
            f.write( str(model_config))
        print("saved model config at "+str(out_file))
        '''

    def eval_model(self):
        ''' evaluate trained model '''

        # prenet evaluation
        self.model_eval = self.model.evaluate(
            self.data.get_test_data(as_matrix = True),
            self.data.get_test_labels())

        # save history of eval metrics
        self.model_history = self.trained_model.history

        # save predicitons
        self.model_prediction_vector = self.model.predict(
            self.data.get_test_data(as_matrix = True) )

        # save predicted classes with argmax
        self.predicted_classes = np.argmax( self.model_prediction_vector, axis = 1)

        # save confusion matrix
        self.confusion_matrix = confusion_matrix(
            self.data.get_test_labels(as_categorical = False), self.predicted_classes)

        # print evaluations
        #print("prenet test roc:  {}".format(
        #    roc_auc_score(self.data.get_prenet_test_labels(), self.prenet_predicted_vector)))
        if self.eval_metrics:
            print("model test loss: {}".format(self.model_eval[0]))
            for im, metric in enumerate(self.eval_metrics):
                print("model test {}: {}".format(metric, self.model_eval[im+1]))



    def plot_metrics(self):
        ''' plot history of loss function and evaluation metrics '''

        metrics = ["loss"]
        if self.eval_metrics: metrics += self.eval_metrics

        for metric in metrics:
            # prenet plot
            plt.clf()
            train_history = self.model_history[metric]
            val_history = self.model_history["val_"+metric]

            n_epochs = len(train_history)
            epochs = np.arange(1,n_epochs+1,1)

            plt.plot(epochs, train_history, "b-", label = "train", lw = 2)
            plt.plot(epochs, val_history, "r-", label = "validation", lw = 2)
            plt.title("train and validation "+str(metric)+" of model")

            plt.grid()
            plt.xlabel("epoch")
            plt.ylabel(metric)

            plt.legend()

            out_path = self.save_path + "/model_history_"+str(metric)+".pdf"
            plt.savefig(out_path)
            print("saved plot of "+str(metric)+" at "+str(out_path))

    def plot_input_output_correlation(self):

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
            pred_values = self.model_prediction_vector[:,i_cls]

            # correlation values for class
            corr_values = {}

            # loop over input variables
            for i_var, var in enumerate(self.train_variables):
                # scatter plot:
                # x-axis: input variable value
                # y-axis: predicted discriminator output

                var_values = input_data[var].values

                assert( len(var_values) == len(pred_values) )

                plt.hist2d(var_values, pred_values,
                    bins = [min(binning.binning[var]["nbins"],20), 20],
                    norm = LogNorm())
                plt.colorbar()

                # calculate correlation value
                correlation = np.corrcoef(var_values, pred_values)[0][1]
                print("correlation between {} and {}: {}".format(
                    cls, var, correlation))

                # write correlation value on plot
                plt.title( correlation, loc = "left")
                plt.xlabel(var)
                plt.ylabel(cls+"_predicted")

                out_name = plt_path + "/correlation_{}_{}.pdf".format(cls,var)
                plt.savefig(out_name.replace("[","_").replace("]",""))
                plt.clf()

                corr_values[var] = correlation

            # save correlation value to dataframe
            df[cls] = pandas.Series( corr_values )

        # save dataframe of correlations
        out_path = self.save_path + "/correlation_matrix.h5"
        df.to_hdf(out_path, "correlations")
        print("saved correlation matrix at "+str(out_path))

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

        out_path = self.save_path + "/confusion_matrix.pdf"
        plt.savefig(out_path)
        print("saved confusion matrix at "+str(out_path))
        plt.clf()
