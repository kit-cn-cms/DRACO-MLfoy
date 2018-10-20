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

# local imports
import data_frame




class CNN():
    def __init__(self, in_path, save_path,      
                class_label = "nJets", 
                batch_size = 128, 
                train_epochs = 20,
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
        
        # batch size for training
        self.batch_size  = batch_size
        # number of training epochs
        self.train_epochs = train_epochs
        # optimizer
        self.optimizer = optimizer
        # loss_function
        self.loss_function = loss_function
        # eval_metrics
        self.eval_metrics = eval_metrics

    def load_datasets(self):
        ''' load train and validation dataset '''
        self.train_data = data_frame.DataFrame( 
            self.in_path+"_train.h5", output_label = self.class_label, one_hot = False )
        self.val_data = data_frame.DataFrame( 
            self.in_path+"_val.h5", output_label = self.class_label, one_hot = False )

        self.num_classes = self.train_data.num_classes

    def build_default_model(self):
        ''' default CNN model for testing purposes
            has three conv layers with max pooling and one densly connected layer '''
        model = models.Sequential()
        
        # input layer
        model.add(
            layer.Conv2D( 64, kernel_size = (5,5), activation = "relu", padding = "same", 
            input_shape = self.train_data.input_shape ))
        model.add(
            layer.AveragePooling2D( pool_size = (3,3), padding = "same" ))
        model.add(
            layer.Dropout(0.1))

        # second layer
        model.add(
            layer.Conv2D( 128, kernel_size = (4,4), activation = "relu", padding = "same"))
        model.add(
            layer.AveragePooling2D( pool_size = (3,3), padding = "same" ))
        model.add(
            layer.Dropout(0.1))
    
        # third layer
        model.add(
            layer.Conv2D( 256, kernel_size = (3,3), activation = "relu", padding = "same"))
        model.add(
            layer.MaxPooling2D( pool_size = (3,3), padding = "same" ))
        model.add(
            layer.Dropout(0.1))

        # dense layer
        model.add(
            layer.Flatten())
        model.add(
            layer.Dense( 256, activation = "relu" ))
        model.add(
            layer.Dropout(0.1))

        # output
        model.add( 
            layer.Dense( self.num_classes, activation = "relu" ))
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
            y = self.train_data.Y,
            batch_size = self.batch_size,
            epochs = self.train_epochs,
            shuffle = True,
            validation_data = (self.val_data.X, self.val_data.Y ))

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
            self.in_path+"_test.h5", output_label = self.class_label, one_hot = False )


        self.test_eval = self.model.evaluate(
            self.test_data.X, self.test_data.Y)
        print("test loss: {}".format( self.test_eval[0] ))
        for im, metric in enumerate(self.eval_metrics):
            print("test {}: {}".format( metric, self.test_eval[im+1] ))

        self.history = self.trained_model.history
        
        self.predicted_vector = self.model.predict( self.test_data.X )
        
        # correct predictons (kind of arbitrary)
        # everything so far only implemented for a single output neuron with integer targets
        self.predicted_classes = [int(val+0.5) for val in self.predicted_vector[:,0]]

        self.confusion_matrix = confusion_matrix(
            self.test_data.Y, self.predicted_classes)

    # --------------------------------------------------------------------
    # result plotting functions
    # --------------------------------------------------------------------
    def print_classification_examples(self):
        ''' print some examples of classifications '''

        # plot correct examples
        plt.clf()
        plt.figure(figsize = [5,5])
        for i in range(4):
            plt.subplot(2,2,i+1)
            plt.imshow( 
                self.test_data.X[i].reshape(*self.test_data.input_shape[:2]),
                cmap = "Greens", norm = LogNorm())
        
            plt.title( "Predicted {}\nTrue {}".format(
                self.predicted_vector[i][0],
                self.test_data.Y[i] ))

        out_path = self.save_path + "/prediction_examples.pdf"
        plt.savefig( out_path )
        print("saved examples of predictions to "+str(out_path))
        plt.clf()
        
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

        epochs = range(self.train_epochs)
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

        predicted_values = self.predicted_vector
        true_values = self.test_data.Y
        
        min_val = np.min([np.min(self.predicted_classes), np.min(self.test_data.Y)])
        max_val = np.max([np.max(self.predicted_classes), np.max(self.test_data.Y)])

        n_bins = max_val - min_val + 1
        bin_range = [min_val - 0.5, max_val + 0.5]

        plt.clf()
        plt.hist(true_values, bins = n_bins, range = bin_range,
            histtype = "stepfilled", label = "truth")

        # prepare predicted histogram
        hist, _ = np.histogram(predicted_values, bins = n_bins, range = bin_range)
        x_values = np.arange(min_val, max_val+1, step = 1)
        plt.plot( x_values, hist, "o", color = "black", label = "prediction")

        plt.grid()
        plt.legend()
        plt.xlabel(str(self.class_label))
        
        out_file = self.save_path+"/discriminator_"+str(self.class_label)+".pdf"
        plt.savefig(out_file)
        print("saved discriminator plot for "+str(self.class_label)+" at "+str(out_file))



    def plot_confusion_matrix(self):
        ''' generate confusion matrix for classification '''
  
        plt.clf()
        plt.figure( figsize = (10,10) )
        
        minimum = np.min( self.confusion_matrix ) /(np.pi**2.0 * np.exp(1.0)**2.0)
        maximum = np.max( self.confusion_matrix ) *(np.pi**2.0 * np.exp(1.0)**2.0)
        
        min_val = np.min([np.min(self.predicted_classes), np.min(self.test_data.Y)])
        max_val = np.max([np.max(self.predicted_classes), np.max(self.test_data.Y)])


        x = np.arange(min_val, max_val+1, step = 1)
        y = np.arange(min_val, max_val+1, step = 1)
        xn, yn = np.meshgrid(x,y)

        plt.pcolormesh(xn, yn, self.confusion_matrix, 
            norm = LogNorm( vmin = max(minimum, 1e-6), vmax = maximum ))
        plt.colorbar()

        plt.xlim(min_val,max_val+1)
        plt.ylim(min_val,max_val+1)

        plt.xlabel("Predicted")
        plt.ylabel("True")

        for yit in range(self.confusion_matrix.shape[0]):
            for xit in range(self.confusion_matrix.shape[1]):
                plt.text( xit+min_val+0.5, yit+min_val+0.5,
                    "{:.1f}".format(self.confusion_matrix[yit, xit]),
                    horizontalalignment = "center",
                    verticalalignment = "center")

        plt_axis = plt.gca()
        plt_axis.set_xticks(x + 0.5, minor = False )
        plt_axis.set_yticks(y + 0.5, minor = False )

        plt_axis.set_xticklabels(x)
        plt_axis.set_yticklabels(y)

        plt_axis.set_aspect("equal")

        plt.tight_layout()

        out_path = self.save_path+"/confusion_matrix.pdf"
        plt.savefig(out_path)
        print("saved confusion matrix at "+str(out_path))
        plt.clf()
