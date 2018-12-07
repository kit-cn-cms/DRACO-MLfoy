import keras.layers as layers
import keras
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

# local imports
import data_frame

class AutoEncoder():
    def __init__(self, in_path, save_path,
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

        self.class_label = None

    def load_datasets(self):
        ''' load train and validation dataset '''
        self.train_data = data_frame.DataFrame(
            self.in_path+"_train.h5")
        self.val_data = data_frame.DataFrame(
            self.in_path+"_val.h5")

    def build_default_model(self):
        # convolutional default model
        # input_image = layers.Input( shape = (self.train_data.input_size, ) )
        input_image = layers.Input( shape = self.train_data.input_shape )
        #encoder = layers.Reshape(
        #    (*self.input_shape), input_shape = (self.train_data.input_size, ))(input_image)

        # encoder
        encoder = layers.Conv2D(
            1, (12,12), activation = "relu", padding = "same")(input_image)
        encoder = layers.AveragePooling2D(
            (2,2), padding = "same")(encoder)
        #encoder = layers.Conv2D(
        #    1, (4,4), activation = "relu", padding = "same")(encoder)
        #encoder = layers.AveragePooling2D(
        #    (2,2), padding = "same")(encoder)


        # decoder
        #decoder = layers.UpSampling2D(
        #    (2,2))(encoder)
        #decoder = layers.Conv2D(
        #    1, (4,4), activation = "relu", padding = "same")(decoder)
        decoder = layers.UpSampling2D(
            (2,2))(encoder)
        decoder = layers.Conv2D(
            1, (12,12), activation = "relu", padding = "same")(decoder)

        # definde autoencoder model
        model = keras.models.Model( input_image, decoder )

        # standalone encoder
        encoder_model = keras.models.Model( input_image, encoder )

        return model, encoder_model
    
    def build_model(self, model = None, encoder_model = None ):
        if model == None and encoder_model == None:
            print("loading default model")
            model, encoder_model = self.build_default_model()

        # save encoder 
        self.encoder = encoder_model
        self.encoder_shape = self.encoder.layers[-1].output_shape
        print("encoder output shape "+str(self.encoder_shape))

        # compile auto encoder
        model.compile(
            optimizer  = self.optimizer,
            loss        = self.loss_function,
            metrics     = self.eval_metrics)

        self.model = model
        self.model.summary()
        out_file = self.save_path+"/model_summary.yml"

        yaml_model = self.model.to_yaml()
        with open(out_file, "w") as f:
            f.write(yaml_model)
        print("saved model summary at "+str(out_file))


    def train_model(self):
        self.trained_model = self.model.fit(
            x = self.train_data.X,
            y = self.train_data.X,
            batch_size = self.batch_size,
            epochs = self.train_epochs,
            shuffle = True,
            validation_data = (self.val_data.X, self.val_data.X))

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
        self.test_data = data_frame.DataFrame(
            self.in_path+"_test.h5")

        self.test_eval = self.model.evaluate(
            self.test_data.X, self.test_data.X)
        print("test loss: {}".format( self.test_eval[0] ))
        for im, metric in enumerate(self.eval_metrics):
            print("test {}: {}".format( metric, self.test_eval[im+1] ))

        self.history = self.trained_model.history

        self.test_encoded_images = self.encoder.predict(
            self.test_data.X)
        self.test_decoded_images = self.model.predict(
            self.test_data.X)
    
        
    def print_classification_examples(self, n = 1):
        plt.clf()
        plt.figure(figsize = [15,5])
        print(self.encoder_shape)
        print(self.test_data.input_shape)
        for i in range(n):
            # plot original image
            ax = plt.subplot(1,3,1)
            img = self.test_data.X[i].reshape(*self.test_data.input_shape[:2])
            plt.imshow(img, cmap = "Greens", norm = LogNorm())
            plt.title("original image")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # plot encoded image
            ax = plt.subplot(1,3,2)
            img = self.test_encoded_images[i].reshape(*self.encoder_shape[1:3])
            plt.imshow(img, cmap = "Greens", norm = LogNorm())
            plt.title("encoded image")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            # plot decoded image
            ax = plt.subplot(1,3,3)
            img = self.test_decoded_images[i].reshape(*self.test_data.input_shape[:2])
            plt.imshow(img, cmap = "Greens", norm = LogNorm())
            plt.title("decoded image")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            out_path = self.save_path + "/encoder_example_"+str(i)+".pdf"
            plt.savefig(out_path)
            plt.clf()
            
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

