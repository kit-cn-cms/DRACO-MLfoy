from DNN import DNN, EarlyStopping

# import with ROOT
from pyrootsOfTheCaribbean.evaluationScripts import plottingScripts

# imports with keras
import Derivatives
from Derivatives import Inputs, Outputs, Derivatives

import keras
import keras.optimizers as optimizers
import keras.models as models
import keras.layers as layer
import pandas as pd

# Classifying Adversarial Network
class CAN(DNN):

    def _load_architecture(self, config):
        ''' load the architecture configs '''

        # define default network configuration
        self.architecture = {
          "layers":                   [100,100,100],
          "loss_function":            "categorical_crossentropy",
          "Dropout":                  0.50,
          "L1_Norm":                  0,
          "L2_Norm":                  1e-5,
          "batch_size":               4096,
          "optimizer":                optimizers.Adam(1e-4),
          "activation_function":      "elu",
          "output_activation":        "Softmax",
          "earlystopping_percentage": 0.02,
          "earlystopping_epochs":     100,
          "adversary_layers":         [100,100],
          "pretrain_class_epochs":    200,
          "pretrain_adv_epochs":      50,
          "adversary_epochs":         10,
          "adversary_iterations":     100,
        }

        for key in config:
            self.architecture[key] = config[key]

    def build_model(self, config = None, penalty = 10):
        ''' build default straight forward GAN from architecture dictionary '''

        if config:
            self._load_architecture(config)
            print("loading non default net configs")

        number_of_input_neurons     = self.data.n_input_neurons

        number_of_neurons_per_layer = self.architecture["layers"]
        dropout                     = self.architecture["Dropout"]
        activation_function         = self.architecture["activation_function"]
        if activation_function == "leakyrelu":
            activation_function = "linear"
        l2_regularization_beta      = self.architecture["L2_Norm"]
        output_activation           = self.architecture["output_activation"]
        number_of_neurons_per_adv_layer = self.architecture["adversary_layers"]

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
        self.class_model = models.Model(inputs = [Inputs], outputs = [X])

        #self.class_model.summary()
        adv_layers = self.class_model(Inputs)
        # loop over adversary dense layers
        for iLayer, nNeurons in enumerate(number_of_neurons_per_adv_layer):
            adv_layers = keras.layers.Dense(
                units               = nNeurons,
                activation          = activation_function,
                kernel_regularizer  = keras.regularizers.l2(l2_regularization_beta))(adv_layers)

        # generate adversary output layer
        adv_layers = keras.layers.Dense(
            units               = 1,
            activation          = 'sigmoid',
            kernel_regularizer  = keras.regularizers.l2(l2_regularization_beta))(adv_layers)
        self.adv_model = keras.models.Model(inputs=[Inputs], outputs = [adv_layers])

        def make_loss_Class(c):
            def loss_Class(y_true, y_pred):
                if self.data.binary_classification: return c * keras.losses.binary_crossentropy(y_true,y_pred)
                else: return c * keras.losses.categorical_crossentropy(y_true,y_pred)
            return loss_Class

        def make_loss_Adv(c):
            def loss_Adv(z_true, z_pred):
                return c * keras.losses.binary_crossentropy(z_true,z_pred)
            return loss_Adv

        self.class_model.compile(loss=self.architecture["loss_function"], optimizer=self.architecture["optimizer"], metrics=self.eval_metrics)
        self.class_model.summary()

        self.adv_model.compile(loss=[make_loss_Adv(c=1.0)], optimizer=self.architecture["optimizer"], metrics = self.eval_metrics)
        self.adv_model.summary()

        # Compile the model with mean squared error as the loss function
        self.adv_model.trainable = False
        self.class_model.trainable = True
        self.class_adv_model = keras.models.Model(inputs=[Inputs], outputs=[self.class_model(Inputs),self.adv_model(Inputs)])
        self.class_adv_model.compile(loss=[make_loss_Class(c=1.0),make_loss_Adv(c=-penalty)], optimizer=self.architecture["optimizer"], metrics=self.eval_metrics)

        self.class_adv_model.summary()
        self.adv_model.trainable = True
        self.class_model.trainable = False
        self.adv_class_model = keras.models.Model(inputs=[Inputs], outputs=[self.adv_model(Inputs)])
        self.adv_class_model.compile(loss =[make_loss_Adv(c=1.)], optimizer=self.architecture["optimizer"], metrics = self.eval_metrics)

        self.adv_class_model.summary()



        # save the model
        self.model = self.class_model

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
        if self.architecture["earlystopping_percentage"] or self.architecture["earlystopping_epochs"]:
            callbacks = [EarlyStopping(
                monitor         = "loss",
                value           = self.architecture["earlystopping_percentage"],
                min_epochs      = 50,
                stopping_epochs = self.architecture["earlystopping_epochs"],
                verbose         = 1)]

        # train main net
        self.adv_model.trainable = False
        self.class_model.trainable = True
        self.trained_model=self.class_model.fit(
            x = self.data.get_train_data(as_matrix = True),
            y = self.data.get_train_labels(),
            batch_size          = self.architecture["batch_size"],
            epochs              = self.architecture["pretrain_class_epochs"],
            shuffle             = True,
            callbacks           = callbacks,
            validation_split    = 0.25,
            sample_weight       = self.data.get_adversary_weights_classifier())

        self.adv_model.trainable = True
        self.class_model.trainable = False
        self.adv_class_model.fit(            
            x = self.data.get_train_data(as_matrix = True),
            y = self.data.get_adversary_labels(),
            batch_size          = self.architecture["batch_size"],
            epochs              = self.architecture["pretrain_adv_epochs"],
            shuffle             = True,
            callbacks           = callbacks,
            validation_split    = 0.25,
            sample_weight       = self.data.get_adversary_weights_adversary())
        # self.predict_event_query("Evt_ID==1163")
        
        for i in range(self.architecture["adversary_iterations"]):
            print("adversary iteration {}".format(i))
            self.adv_model.trainable = False
            self.class_model.trainable = True
            self.class_adv_model.fit(            
                x = self.data.get_train_data(as_matrix = True),
                y = [self.data.get_train_labels(),self.data.get_adversary_labels()],
                batch_size          = self.architecture["batch_size"],
                epochs              = self.architecture["adversary_epochs"],
                shuffle             = True,
                callbacks           = callbacks,
                validation_split    = 0.25,
                sample_weight       = [self.data.get_adversary_weights_classifier(),self.data.get_adversary_weights_adversary()])
                                

            self.adv_model.trainable = True
            self.class_model.trainable = False
            self.adv_class_model.fit(           
                x = self.data.get_train_data(as_matrix = True),
                y = self.data.get_adversary_labels(),
                batch_size          = self.architecture["batch_size"],
                epochs              = self.architecture["adversary_epochs"],
                shuffle             = True,
                callbacks           = callbacks,
                validation_split    = 0.25,
                sample_weight       = self.data.get_adversary_weights_adversary())

        # self.trained_model = self.class_model
        # self.predict_event_query("Evt_ID==1163")
        self.model = self.class_model        

    def plot_ttbbKS(self, log = False, privateWork = False,
                        signal_class = None, nbins = None, bin_range = None):
        ''' plot comparison between ttbb samples '''

        # evaluate trained model with different ttbb samples

        # save predicitons
        self.model_prediction_vector_nominal = self.model.predict(self.data.get_test_data_nominal (as_matrix = True))
        self.model_prediction_vector_additional = self.model.predict(self.data.get_test_data_additional (as_matrix = True))

        if not bin_range:
            bin_range = [round(1./self.data.n_output_neurons,2), 1.]
        if not nbins:
            nbins = int(40*(1.-bin_range[0]))

        ttbbKS = plottingScripts.plotttbbKS(
            data                        = self.data,
            test_prediction_nominal     = self.model_prediction_vector_nominal,
            test_prediction_additional  = self.model_prediction_vector_additional,
            event_classes               = self.event_classes,
            nbins                       = nbins,
            bin_range                   = bin_range,
            signal_class                = signal_class,
            event_category              = self.category_label,
            plotdir                     = self.plot_path,
            logscale                    = log,
            addSampleSuffix             = self.addSampleSuffix)

        ttbbKS.plot(ratio = False, privateWork = privateWork)

    def plot_ttbbKS_binary(self, log = False, privateWork = False,
                        signal_class = None, nbins = None, bin_range = None):
        ''' plot comparison between ttbb samples binary case '''

        # evaluate trained model with different ttbb samples

        # save predicitons
        self.model_prediction_vector_nominal = self.model.predict(self.data.get_test_data_nominal (as_matrix = True))
        self.model_prediction_vector_additional = self.model.predict(self.data.get_test_data_additional (as_matrix = True))

        if not bin_range:
            bin_range = [0., 1.]
        if not nbins:
            nbins = int(40*(1.-bin_range[0]))

        ttbbKS = plottingScripts.plotttbbKS_binary(
            data                        = self.data,
            test_prediction_nominal     = self.model_prediction_vector_nominal,
            test_prediction_additional  = self.model_prediction_vector_additional,
            event_classes               = self.event_classes,
            nbins                       = nbins,
            bin_range                   = bin_range,
            signal_class                = signal_class,
            event_category              = self.category_label,
            plotdir                     = self.plot_path,
            logscale                    = log,
            addSampleSuffix             = self.addSampleSuffix)

        ttbbKS.plot(ratio = False, privateWork = privateWork)