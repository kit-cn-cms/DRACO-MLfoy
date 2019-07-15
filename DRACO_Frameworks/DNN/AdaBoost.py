import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# plt.use('TkAgg')
# plt.rcParams['backend'] = 'Agg'
import json

#sklearn imports
from sklearn import metrics

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


class AdaBoost():
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
            adaboost_epochs = 100,
            evenSel         = None,
            m2              = False):

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

        #number of adaboost_epochs
        self.adaboost_epochs = adaboost_epochs

        #should Adaboost.M2 algorithm be Used
        self.m2 = m2

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
            "layers":                   [200,100],
            "loss_function":            "squared_hinge",
            "Dropout":                  0.3,
            "L2_Norm":                  0.,
            "batch_size":               4000,
            "optimizer":                optimizers.Adadelta(),
            "activation_function":      "selu",
            "output_activation":        "Tanh",
            "earlystopping_percentage": None,
            "earlystopping_epochs":     None,
            }

        for key in config:
            self.architecture[key] = config[key]


    def load_trained_model(self, inputDirectory):
        ''' load an already trained model '''
        load_path = inputDirectory + "save_model/" + self.name + "/"
        self.weak_model_trained = []
        for i in range(0, self.adaboost_epochs):
            load = load_path + "trained_model" + str(i) + ".h5py"
            model = keras.models.load_model(load)
            self.weak_model_trained.append(model)
        #load alpha
        self.alpha_t = np.load(load_path + "alpha.npy")
        #load epsilon
        self.epsilon = np.load(load_path + "epsilon.npy")
        #get predictions
        self.train_prediction_vector = []
        self.test_prediction_vector = []
        for i in range(0, self.alpha_t.shape[0]):
            self.train_prediction_vector.append(self.weak_model_trained[i].predict(self.data.get_train_data(as_matrix = True)))
            self.test_prediction_vector.append(self.weak_model_trained[i].predict(self.data.get_test_data(as_matrix = True)))


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


    def get_alpha_epsilon(self):
        '''Calculate weighted error and return alpha_t after each training'''
        # model_prediction_vector = self.model.predict(self.data.get_test_data(as_matrix = True))
        #hier kann man auch self... verwenden da prediciton vectoren schon in train_model erstellt
        model_train_prediction = self.train_prediction_vector[-1]
        model_train_label = self.data.get_train_labels(as_categorical = False) #not sure if should be True
        model_train_weights = self.data.get_train_weights()
        weight_sum = np.sum(model_train_weights)
        #Calculate epsilon and alpha
        num = model_train_weights.shape[0]

        #create a discret prediction vector
        model_train_prediction_discret = np.array([])
        for x in model_train_prediction:
            if x<self.cut_value:
                model_train_prediction_discret = np.append(model_train_prediction_discret, int(self.binary_bkg_target))
            else:
                model_train_prediction_discret = np.append(model_train_prediction_discret, int(1))
        #adaboost.m2 algorithm
        if self.m2:
            #normalize the weights
            model_train_weights = model_train_weights/weight_sum
            # print("# DEBUG: get_alpha_epsilon, normalized? ", np.sum(model_train_weights))
            epsilon = 0
            counter = 0
            model_train_prediction = (model_train_prediction + 1)/2     #shift output to [0,1]
            print(model_train_prediction[0:10])
            print(model_train_label[0:10])
            for i in np.arange(0, num):
                if model_train_prediction_discret[i] != model_train_label[i]:
                    counter += 1
                    if model_train_label[i] == 1:                                          #check if the real value is traget
                        # print("# DEBUG: model_train_weights: ", model_train_weights[i])
                        epsilon += model_train_weights[i]*(1-model_train_prediction[i])
                    else:
                        epsilon += model_train_weights[i]*(model_train_prediction[i])
            # print("# DEBUG: wong predicitons: ", counter)
            epsilon = epsilon/weight_sum
            alpha = epsilon/(1-epsilon)
            # print("# DEBUG: get_a epsilon, alpha")
        #normal adaboost
        else:
            # print(model_train_prediction[0:10])
            # print(model_train_label[0:10])
            weight_false = 0
            for i in np.arange(0,num):
                if model_train_prediction_discret[i] != model_train_label[i]:
                    # print("# DEBUG: pred, label", model_train_prediction_discret[i], model_train_label[i])
                    weight_false += model_train_weights[i]
            # print("# DEBUG: get_alpha_epsilon, weight_false, weight_sum ", weight_false, weight_sum)
            epsilon = weight_false/weight_sum
            alpha = 0.5*np.log((1-epsilon)/epsilon)
            # print("# DEBUG: get_alpha_epsilon, alpha: ", alpha)
            # print("# DEBUG: get_alpha_epsilon, epsilon: ", epsilon)
        #adjust weights
        self.data.ada_adjust_weights(model_train_prediction, model_train_prediction_discret, alpha, self.m2)
        #check if epsilon < 0.5
        if epsilon > 0.5:
            print("# DEBUG: In ada_eval_training epsilon > 0.5")
        print("# DEBUG: get_alpha_epsilon, alpha, epsilon", alpha, epsilon)
        return alpha, epsilon


    def build_model(self, config):
        ''' build a DNN model (weak Classifier)
            use options defined in 'config' dictionary '''

        self._load_architecture(config)
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
        '''train the model'''

        self.weak_model_trainout = [] #does not contain the trained model
        self.weak_model_trained = [] #trained weak Classifier
        self.epsilon = []
        self.alpha_t = []
        self.train_prediction_vector = []   #collect prediction vector for each weak classifier
        self.test_prediction_vector = []
        # self.train_label = []   #labels need to be collected because order of df changes after evaluation
        # self.test_label = []
        # print("# DEBUG: Watch weights: ", self.data.get_train_weights()[0:20])
        for t in np.arange(0,self.adaboost_epochs):
            print("adaboost_epoch: ", t)
            # df = self.data.get_full_df_train()
            # print("# DEBUG: Watch df_train: ", df.train_weight[200:300])
            self.weak_model_trainout.append(self.model.fit(
                x = self.data.get_train_data(as_matrix = True),
                y = self.data.get_train_labels(),
                batch_size          = self.architecture["batch_size"],
                epochs              = self.train_epochs,
                shuffle             = True,
                # callbacks           = callbacks,
                validation_split    = 0.25,
                sample_weight       = self.data.get_train_weights()))
            #get prediction vector for training and test
            self.train_prediction_vector.append(self.model.predict(self.data.get_train_data(as_matrix = True)))
            self.test_prediction_vector.append(self.model.predict(self.data.get_test_data(as_matrix = True)))
            #get alpha, epsilon and adjust weights
            alpha , epsilon = self.get_alpha_epsilon()
            self.epsilon.append(float(epsilon))
            self.alpha_t.append(float(alpha))   #make dict alpha -> model
            # print("# DEBUG: mean pred: ", np.mean(train_prediction_vector))
            # print("# DEBUG: Watch alpha", self.alpha_t)
            # print("# DEBUG: Watch epsilon", self.epsilon)
            #collect weak classifier
            self.weak_model_trained.append(self.model)


    def save_model(self, signals):
        ''' save the trained model'''

        # get the path
        # self.cp_path = "Just to here to avoid error"
        save_path = self.path + "save_model/" + self.name + "/"
        # create new dir for the trained net
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        else:
            print("Dir already exists -> can not save model")
            sys.exit()

        # save model as h5py file
        for i in range(0, self.adaboost_epochs):
            out_file = save_path + "trained_model" + str(i) + ".h5py"
            self.weak_model_trained[i].save(out_file)
        print("saved trained model")

        # save config of model
        model_config = self.weak_model_trained[0].get_config()
        out_file = save_path + "trained_model_config"
        with open(out_file, "w") as f:
            f.write( str(model_config))
        print("saved model config at")

        #save alpha
        alpha = np.asarray(self.alpha_t)
        np.save(save_path + "alpha", alpha)

        #save epsilon
        epsilon = np.asarray(self.epsilon)
        np.save(save_path + "epsilon", epsilon)

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
        configs["adaboost_epochs"] = self.adaboost_epochs

        # save information for binary DNN
        if self.data.binary_classification:
            configs["binaryConfig"] = {
                "minValue": self.input_samples.bkg_target,
                "maxValue": 1.,
                "signals":  signals}

        json_file = save_path + "/net_config.json"
        with open(json_file, "w") as jf:
            json.dump(configs, jf, indent = 2, separators = (",", ": "))
        print("wrote net configs to "+str(json_file))

        # save configurations of variables for plotscript
        # plot_file = self.cp_path+"/plot_config.csv"
        # variable_configs = pd.read_csv(basedir+"/pyrootsOfTheCaribbean/plot_configs/variableConfig.csv").set_index("variablename", drop = True)
        # variables = variable_configs.loc[self.train_variables]
        # variables.to_csv(plot_file, sep = ",")
        # print("wrote config of input variables to {}".format(plot_file))



    def weight_prediction(self, pred, alpha):
        pred = np.asarray(pred)
        alpha = np.asarray(alpha)
        # print("# DEBUG: weight_prediction, pred.shape: ", pred.shape)
        # print("# DEBUG: weight_prediction, alpha.shape: ", alpha.shape)
        print("# DEBUG: alpha: ", alpha)
        sum = 0
        # print("# DEBUG: len(alpha): ", len(alpha))
        print("# DEBUG: weight_prediction, initial pred: ", pred[:len(alpha),0:5])
        if self.m2:
            for i in range(0,len(alpha)):
                factor = np.log(1/alpha[i])
                pred[i] = pred[i]*factor
                sum +=  factor
        else:
            for i in range(0,len(alpha)):
                pred[i] = pred[i]*alpha[i]
                sum += alpha[i]
        pred = pred/sum
        # print("# DEBUG: pred: ", pred[0:len(alpha)][0:5])
        # final and initial prediction is ok so far
        print("# DEBUG: weight_prediction, final pred: ", pred[:len(alpha),0:5])
        return pred[0:len(alpha)]


    def strong_classification(self, pred, alpha):
        '''builds prediciton vector for strong classifier'''
        pred_array = self.weight_prediction(pred, alpha)
        prediction_vector = np.sum(pred_array, axis = 0)
        final_prediction_vector = np.array([])
        for x in prediction_vector:
            if x<self.cut_value:
                prediction_vector_disc = np.append(final_prediction_vector, int(self.binary_bkg_target))
            else:
                prediction_vector_disc = np.append(final_prediction_vector, int(1))
        return prediction_vector, prediction_vector_disc


    def eval_model(self):
        '''evalute trained model'''
        '''Should contain:  -Plot prediciton_fraction
                            -Get roc'''
        #get saving path
        save_path = self.path + "plot/"
        #get the labels
        self.train_label = self.data.get_train_labels(as_categorical = False)
        self.test_label = self.data.get_test_labels(as_categorical = False)
        #get prediction fraction after each adaboost_epoch
        train_fraction = np.array([])
        test_fraction = np.array([])
        for i in np.arange(0, len(self.train_prediction_vector)):
            train_prediction, train_prediction_disc = self.strong_classification(self.train_prediction_vector, self.alpha_t[0:i+1])
            test_prediction, test_prediction_disc = self.strong_classification(self.test_prediction_vector, self.alpha_t[0:i+1])
            # print("# DEBUG: count_nonzero: ", np.count_nonzero(test_prediction_final==self.test_label))
            train_fraction = np.append(train_fraction,
                        np.count_nonzero(train_prediction_disc==self.train_label)/float(self.train_label.shape[0]))
            test_fraction = np.append(test_fraction,
                        np.count_nonzero(test_prediction_disc==self.test_label)/float(self.test_label.shape[0]))
        #get roc
        fpr, tpr, thresholds = metrics.roc_curve(self.test_label, test_prediction)
        roc_auc = metrics.auc(fpr, tpr)
        #plot
        epoches = np.arange(1, len(self.train_prediction_vector)+1)
        print("# DEBUG: fraction: ", train_fraction, test_fraction)
        plt.figure(1)
        plt.plot(epoches, train_fraction, 'r-', label = "Trainingsdaten")
        plt.plot(epoches, test_fraction, 'g-', label = "Testdaten")
        plt.title("Anteil richtig Bestimmt - AdaBoost_binary_discret")
        plt.legend(loc='lower right')
        plt.savefig(save_path + self.name + "_frac.pdf")

        plt.figure(2)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(save_path + self.name +"_roc.pdf")

        plt.figure(3)
        plt.title('Epsilon')
        plt.plot(epoches, self.epsilon, '-')
        plt.savefig(save_path + self.name +"_eps.pdf")

        # plt.show()
