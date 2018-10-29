# global imports
import keras
import keras.models as models
import keras.layers as layer

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))
# local imports
import data_frame




class LSTM():
    def __init__(self, in_path, save_path,      
                class_label = "class_label", 
                input_variables = [],
                n_particles = 10,
                normed_inputs = False,
                batch_size = 128, 
                train_epochs = 20,
                early_stopping = None,
                optimizer = "adam", 
                loss_function = "mean_squared_error", 
                eval_metrics = None):

        # saving some information

        # path to input files
        self.in_path = in_path
        # output directory for result files
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs( self.save_path )
        # name of classification variable
        self.class_label = class_label

        # input variables in vector
        self.input_variables = input_variables
        # number of particles per event
        self.n_particles = n_particles
        # norm inputs to mean 0 and std 1
        self.normed_inputs = normed_inputs

        # batch size for training
        self.batch_size  = batch_size
        # number of training epochs
        self.train_epochs = train_epochs
        # early stopping
        self.early_stopping = early_stopping
        # optimizer
        self.optimizer = optimizer
        # loss_function
        self.loss_function = loss_function
        # eval_metrics
        self.eval_metrics = eval_metrics

    def load_datasets(self):
        ''' load train and validation dataset '''
        self.train_data = data_frame.DataFrame( 
            self.in_path+"_train.h5", output_label = self.class_label,
            variables = self.input_variables, n_particles = self.n_particles, 
            normed_inputs = self.normed_inputs)
        self.val_data = data_frame.DataFrame( 
            self.in_path+"_val.h5", output_label = self.class_label,
            variables = self.input_variables, n_particles = self.n_particles, 
            normed_inputs = self.normed_inputs)

        self.num_classes = self.train_data.num_classes

    def build_default_model(self):
        ''' default LSTM model for testing purposes '''
        model = models.Sequential()
        n_filter = 32

        #model.add(
        #    layer.Conv2D(n_filter, (1,1), activation = "relu", padding = "valid",
        #    input_shape = self.train_data.input_shape))
        # output shape = (1, n_particles, n_filter)       

        #model.add(
        #    layer.Reshape((-1,n_filter)))
        # output shape = (n_particles, n_filter)

        model.add(layer.Reshape((self.n_particles,-1),
            input_shape = self.train_data.input_shape))
        # input needs to have shape (time_steps, input_features)
        model.add(layer.LSTM(100))

        model.add(layer.Dense(100, activation = "relu"))
        model.add(layer.Dense(self.num_classes, activation = "softmax"))
        
        model.summary()
        return model


    def build_model(self, model = None):
        ''' build a CNN model
            if none is specified use default constructor '''
        if model == None:
            print("loading default model")
            model = self.build_default_model()
        
        # compile the model
        model.compile(
            loss        = self.loss_function,
            optimizer   = self.optimizer,
            metrics     = self.eval_metrics)
        
        self.model = model

        # model summary
        self.model.summary()
        out_file = self.save_path+"/model_summary.yml"

        yaml_model = self.model.to_yaml()
        with open(out_file, "w") as f:
            f.write(yaml_model)
        print("saved model summary at "+str(out_file))

    def train_model(self):
        self.trained_model = self.model.fit(
            x = self.train_data.X,
            y = self.train_data.one_hot,
            batch_size = self.batch_size,
            epochs = self.train_epochs,
            shuffle = True,
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor = "val_loss", patience = self.early_stopping, verbose = 1, mode = "auto")],
            validation_data = (self.val_data.X, self.val_data.one_hot ))

        # save trained model
        out_file = self.save_path +"/trained_model.h5py"
        self.model.save(out_file)
        print("saved trained model at "+str(out_file))

        model_config = self.model.get_config()
        out_file = self.save_path +"/trained_model_config"
        with open(out_file, "w") as f:
            f.write( str(model_config) )
        print("saved model config at "+str(out_file))

        out_file = self.save_path +"/trained_model_weights.h5"
        self.model.save_weights(out_file)
        print("wrote trained net weights to "+str(out_file))

    def eval_model(self):
        # loading test examples
        self.test_data = data_frame.DataFrame( 
            self.in_path+"_test.h5", output_label = self.class_label,
            variables = self.input_variables, n_particles = self.n_particles, 
            normed_inputs = self.normed_inputs)
        
        if self.class_label == "nJets":
            self.target_names = [self.test_data.inverted_label_dict[i] for i in range(
                self.test_data.min_jets, self.test_data.max_jets+1)]
        else:
            self.target_names = [self.test_data.inverted_label_dict[i] for i in range(
                self.num_classes)]

        self.test_eval = self.model.evaluate(
            self.test_data.X, self.test_data.one_hot)
        print("test loss:     {}".format( self.test_eval[0] ))
        for im, metric in enumerate(self.eval_metrics):
            print("test {}: {}".format( metric, self.test_eval[im+1] ))

        self.history = self.trained_model.history
        
        self.predicted_vector = self.model.predict( self.test_data.X )
        self.predicted_classes = np.argmax( self.predicted_vector, axis = 1)
        
        if self.class_label == "nJets":        
            self.predicted_classes = np.array([self.test_data.max_jets if j >= self.test_data.max_jets \
                                 else self.test_data.min_jets if j <= self.test_data.min_jets \
                                 else j for j in self.predicted_classes])

        self.confusion_matrix = confusion_matrix(
            self.test_data.Y, self.predicted_classes )

    # --------------------------------------------------------------------
    # result plotting functions
    # --------------------------------------------------------------------
    def print_classification_report(self):
        ''' print a classification report '''

        report = classification_report( 
            self.test_data.Y, self.predicted_classes,
            target_names = self.target_names )    

        print("classification report:")
        print(report)
        out_path = self.save_path + "/classification_report"
        with open(out_path, "w") as f:
            f.write(report)
        print("saved classification report to "+str(out_path))

    
    def plot_metrics(self):
        ''' plot history of loss function and metrics '''

        epochs = range(len(self.history["loss"]))
        metrics = ["loss"] + self.eval_metrics

        for metric in metrics:
            plt.clf()
            train_history = self.history[metric]
            val_history = self.history["val_"+metric]
            
            plt.plot(epochs, train_history, "b-", label = "train", lw = 2.5)
            plt.plot(epochs, val_history, "r-", label = "validation", lw = 2.5)
            plt.title("train and validation "+str(metric))

            plt.grid()
            plt.xlabel("epoch")
            plt.ylabel(metric)

            plt.legend()
            
            out_path = self.save_path + "/history_"+str(metric)+".pdf"
            plt.savefig(out_path)
            print("saved plot of "+str(metric)+" at "+str(out_path))

    def plot_discriminators(self, log = False):
        ''' plot discriminator for output classes '''

        for i in range(self.num_classes):
            values = self.predicted_vector[:,i]

            bkg_values = []
            bkg_labels = []
            n_bkg_evts = 0
            n_sig_evts = 0

            for j in range(self.num_classes):
                filtered_values = [values[k] for k in range(len(values)) if self.test_data.Y[k] == j]
                if i == j:
                    sig_values = filtered_values
                    sig_label = self.test_data.inverted_label_dict[j]
                    n_sig_evts += len(filtered_values)
                else:
                    bkg_values.append(filtered_values)
                    bkg_labels.append(self.test_data.inverted_label_dict[j])
                    n_bkg_evts += len(filtered_values)
        
            if n_sig_evts == 0: continue
            # plot the discriminator output
            plt.clf()
            plt.figure( figsize = [5,5] )
            # stack backgrounds
            plt.hist( bkg_values, stacked = True, histtype = "stepfilled", 
                        bins = 20, range = [0,1], label = bkg_labels, log = log)

            # get signal weights
            bkg_sig_ratio = 1.* n_bkg_evts / n_sig_evts     
            sig_weights = [bkg_sig_ratio]*len(sig_values)
            sig_label += "*{:.3f}".format(bkg_sig_ratio)

            # plot signal shape
            plt.hist( sig_values, histtype = "step", weights = sig_weights,
                        bins = 20, range = [0,1], label = sig_label, log = log)

            plt.legend(loc = "upper center")
            plt.xlabel("discriminator output")
            plt.title("discriminator for {}".format(
                self.test_data.inverted_label_dict[i]))
            
            out_path = self.save_path +"/discriminator_{}.pdf".format(
                self.test_data.inverted_label_dict[i].replace(" ","_"))
        
            plt.savefig(out_path)
            print("plot for discriminator of {} saved at {}".format(
                self.test_data.inverted_label_dict[i], out_path))

            plt.clf()



    def plot_confusion_matrix(self):
        ''' generate confusion matrix for classification '''
  
        plt.clf()
        plt.figure( figsize = (1.5*self.num_classes, 1.5*self.num_classes) )
        
        minimum = np.min( self.confusion_matrix ) /(np.pi**2.0 * np.exp(1.0)**2.0)
        maximum = np.max( self.confusion_matrix ) *(np.pi**2.0 * np.exp(1.0)**2.0)

        n_classes = self.confusion_matrix.shape[0]       
        x = np.arange(0,n_classes+1,1)
        y = np.arange(0,n_classes+1,1)

        xn, yn = np.meshgrid(x,y)

        plt.pcolormesh(xn, yn, self.confusion_matrix, 
            norm = LogNorm( vmin = max(minimum, 1e-6), vmax = maximum ))
        plt.colorbar()

        plt.xlim(0, n_classes)
        plt.ylim(0, n_classes)

        plt.xlabel("Predicted")
        plt.ylabel("True")

        for yit in range(n_classes):
            for xit in range(n_classes):
                plt.text( xit+0.5, yit+0.5,
                    "{:.1f}".format(self.confusion_matrix[yit, xit]),
                    horizontalalignment = "center",
                    verticalalignment = "center")

        plt_axis = plt.gca()
        plt_axis.set_xticks(np.arange( (x.shape[0] -1)) + 0.5, minor = False )
        plt_axis.set_yticks(np.arange( (y.shape[0] -1)) + 0.5, minor = False )
        plt_axis.set_xticklabels(self.target_names)
        plt_axis.set_yticklabels(self.target_names)

        plt_axis.set_aspect("equal")

        plt.tight_layout()

        out_path = self.save_path+"/confusion_matrix.pdf"
        plt.savefig(out_path)
        print("saved confusion matrix at "+str(out_path))
        plt.clf()
