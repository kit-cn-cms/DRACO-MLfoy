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
import architecture as arch
import data_frame

import keras
import keras.models as models
import keras.layers as layer
from keras import backend as K
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
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


class DNN():
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
            additional_cut  = None,
            use_pca         = False):

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

        # option for principle component analysis
        self.PCA = use_pca

        # load data set
        self.data = self._load_datasets()
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

        # dict with architectures for analysis
        self.architecture = arch.getArchitecture(self.JTstring)
        self.inputName = "inputLayer"
        self.outputName = "outputLayer"

        # optimizer for training
        if not(optimizer):
            self.optimizer = self.architecture["optimizer"]
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
            use_pca             = self.PCA,
            additional_cut      = self.additional_cut)


        

    def load_trained_model(self):
        ''' load an already trained model '''
        checkpoint_path = self.cp_path + "/trained_model.h5py"

        self.model = keras.models.load_model(checkpoint_path)

        self.model_eval = self.model.evaluate(
            self.data.get_test_data(as_matrix = True),
            self.data.get_test_labels())

        self.model_prediction_vector = self.model.predict(
            self.data.get_test_data(as_matrix = True))

        self.predicted_classes = np.argmax( self.model_prediction_vector, axis = 1)

        # save confusion matrix
        from sklearn.metrics import confusion_matrix
        self.confusion_matrix = confusion_matrix(
            self.data.get_test_labels(as_categorical = False), self.predicted_classes)

        # print evaluations
        from sklearn.metrics import roc_auc_score
        self.roc_auc_score = roc_auc_score(self.data.get_test_labels(), self.model_prediction_vector)
        print("ROC-AUC score: {}".format(self.roc_auc_score))

        if self.eval_metrics:
            print("model test loss: {}".format(self.model_eval[0]))
            for im, metric in enumerate(self.eval_metrics):
                print("model test {}: {}".format(metric, self.model_eval[im+1]))

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
        ''' default straight forward DNN '''
        K.set_learning_phase(True)
        number_of_input_neurons     = self.data.n_input_neurons

        number_of_neurons_per_layer = self.architecture["layers"]
        dropout                     = self.architecture["Dropout"]
        batchNorm                   = self.architecture["batchNorm"]
        activation_function         = self.architecture["activation_function"]
        l2_regularization_beta      = self.architecture["L2_Norm"]
        output_activation           = self.architecture["output_activation"]

        Inputs = keras.layers.Input(
            shape = (number_of_input_neurons,),
            name  = self.inputName)

        X = Inputs
        self.layer_list = [X]

        # loop over dense layers
        for iLayer, nNeurons in enumerate(number_of_neurons_per_layer):
            X = keras.layers.Dense( nNeurons,
                activation = activation_function,
                kernel_regularizer = keras.regularizers.l2(l2_regularization_beta),
                name = "DenseLayer_"+str(iLayer)
                )(X)

            if not dropout == 1:
                X = keras.layers.Dropout(dropout)(X)

            if batchNorm:
                X = keras.layers.BatchNormalization()(X)

        # generate output layer
        X = keras.layers.Dense( self.data.n_output_neurons,
            activation = output_activation.lower(),
            kernel_regularizer = keras.regularizers.l2(l2_regularization_beta),
            name = self.outputName
            )(X)

        # define model
        model = models.Model(inputs = [Inputs], outputs = [X])
        model.summary()

        return model

    def build_model(self, config = None, model = None):
        ''' build a DNN model
            if none is epecified use default model '''
        if config:
            self.architecture = config
            print("loading non default net config")

        if model == None:
            print("Loading default model")
            model = self.build_default_model()

        # compile the model
        model.compile(
            loss = self.architecture["loss_function"],
            optimizer = self.optimizer,
            metrics = self.eval_metrics)

        # save the model
        self.model = model

        # save net information
        out_file = self.save_path+"/model_summary.yml"
        yml_model = self.model.to_yaml()
        with open(out_file, "w") as f:
            f.write(yml_model)

        # save initialization of weights in first layer
        first_layer = self.model.layers[1]
        self.initial_weights = first_layer.get_weights()[0]


    def train_model(self):
        ''' train the model '''

        # add early stopping if activated
        callbacks = None
        if self.early_stopping:
            callbacks = [EarlyStoppingByLossDiff(
                monitor = "loss",
                value = self.architecture["earlystopping_percentage"],
                min_epochs = 50,
                patience = 10,
                verbose = 1)]

        # train main net
        self.trained_model = self.model.fit(
            x = self.data.get_train_data(as_matrix = True),
            y = self.data.get_train_labels(),
            batch_size = self.architecture["batch_size"],
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

        # produce json file with configs
        configs = self.architecture
        configs["inputName"] = self.inputName
        configs["outputName"] = self.outputName+"/"+configs["output_activation"]
        configs = {key: configs[key] for key in configs if not "optimizer" in key}

        json_file = self.cp_path + "/net_config.json"
        with open(json_file, "w") as jf:
            json.dump(configs, jf, indent = 2, separators = (",", ": "))
        print("wrote net configs to "+str(json_file))


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
        from sklearn.metrics import confusion_matrix
        self.confusion_matrix = confusion_matrix(
            self.data.get_test_labels(as_categorical = False), self.predicted_classes)

        # print evaluations
        from sklearn.metrics import roc_auc_score
        self.roc_auc_score = roc_auc_score(self.data.get_test_labels(), self.model_prediction_vector)
        print("ROC-AUC score: {}".format(self.roc_auc_score))

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
                
        

    def save_confusionMatrix(self, location, save_roc):
        ''' save confusion matrix as a line in output file '''
        flattened_matrix = self.confusion_matrix.flatten()
        labels = ["{}_in_{}_node".format(pred, true) for true in self.event_classes for pred in self.event_classes]
        data = {label: [float(flattened_matrix[i])] for i, label in enumerate(labels)}
        data["ROC"] = [float(self.roc_auc_score)]
        df = pd.DataFrame.from_dict(data)
        with pd.HDFStore(location, "a") as store:
            store.append("data", df, index = False)
        print("saved confusion matrix at "+str(location))

    # --------------------------------------------------------------------
    # result plotting functions
    # --------------------------------------------------------------------
    def get_input_weights(self):
        ''' get the weights of the input layer '''
        first_layer = self.model.layers[1]
        weights = first_layer.get_weights()[0]
        self.weight_dict = {}
        print("getting weights in first layer after training:")
        for out_weights, variable in zip( weights, self.train_variables ):
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
            
        
            

    def plot_metrics(self, privateWork = False):
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
            if privateWork:
                plt.title("CMS private work", loc = "left", fontsize = 16)

            title = self.categoryLabel
            title = title.replace("\\geq", "$\geq$")
            plt.title(title, loc = "right", fontsize = 16)

            plt.grid()
            plt.xlabel("epoch", fontsize = 16)
            plt.ylabel(metric, fontsize = 16)

            plt.legend()

            out_path = self.save_path + "/model_history_"+str(metric)+".pdf"
            plt.savefig(out_path)
            print("saved plot of "+str(metric)+" at "+str(out_path))



    def plot_outputNodes(self, log = False, cut_on_variable = None, plot_nonTrainData = False):
        ''' plot distribution in outputNodes '''
        nbins = 20
        bin_range = [0., 1.]

        plotNodes = plottingScripts.plotOutputNodes(
            data                = self.data,
            prediction_vector   = self.model_prediction_vector,
            event_classes       = self.event_classes,
            nbins               = nbins,
            bin_range           = bin_range,
            signal_class        = "ttHbb",
            event_category      = self.categoryLabel,
            plotdir             = self.plot_path,
            logscale            = log,
            plot_nonTrainData   = plot_nonTrainData)

        if cut_on_variable:
            plotNodes.set_cutVariable(
                cutClass = cut_on_variable["class"],
                cutValue = cut_on_variable["value"])

        plotNodes.set_printROCScore(True)
        plotNodes.plot(ratio = False)

    def plot_discriminators(self, log = False, plot_nonTrainData = False, signal_class = "ttHbb"):
        ''' plot all events classified as one category '''
        nbins = 18
        bin_range = [0.1, 1.]

        plotDiscrs = plottingScripts.plotDiscriminators(
            data                = self.data,
            prediction_vector   = self.model_prediction_vector,
            event_classes       = self.event_classes,
            nbins               = nbins,
            bin_range           = bin_range,
            signal_class        = signal_class,
            event_category      = self.categoryLabel,
            plotdir             = self.plot_path,
            logscale            = log,
            plot_nonTrainData   = plot_nonTrainData)

        plotDiscrs.set_printROCScore(True)
        plotDiscrs.plot(ratio = False)


    def plot_input_output_correlation(self, plot = False):

        # get input variables from test set TODO get them unnormed
        input_data = self.data.get_test_data(as_matrix = False, normed = False)

        # initialize empty dataframe
        df = pd.DataFrame()
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
            df[cls] = pd.Series( corr_values )

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
            xvalues = self.model_prediction_vector[:,i_cls]

            for j_cls, ycls in enumerate(self.event_classes):
                yvalues = self.model_prediction_vector[:,j_cls]

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
                    plt.title(self.categoryLabel, loc = "right")

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

        plt.title(self.categoryLabel, loc = "right")

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

    def plot_confusionMatrix(self, norm_matrix = True, privateWork = False, printROC = False):
        ''' plot confusion matrix '''
        plotCM = plottingScripts.plotConfusionMatrix(
            data                = self.data,
            prediction_vector   = self.model_prediction_vector,
            event_classes       = self.event_classes,
            event_category      = self.categoryLabel,
            plotdir             = self.save_path)

        plotCM.set_printROCScore(printROC)

        plotCM.plot(norm_matrix = norm_matrix, privateWork = privateWork)
