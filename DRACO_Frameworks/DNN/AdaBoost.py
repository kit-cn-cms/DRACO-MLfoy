import os
import sys
import numpy as np
import math
# trining to solve some displaying problems (via ssh) here
# solution is to use ssh -X ... or the following two lines
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.rcParams['backend'] = 'Agg'
import json

#sklearn imports
from sklearn import metrics
#interpolation function to calculate sigma
from scipy import interpolate

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

# Use latex style in plots
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

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
        #get final prediction
        #not loaded due to no use of eval_model
        # self.model_prediction_vector = np.load(load_path + "pred_vec.npy")

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
            # print(model_train_prediction[0:10])
            # print(model_train_label[0:10])
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
            # print("# DEBUG: train_model, train_prediction_vector: ", self.train_prediction_vector[t])
            #get alpha, epsilon and adjust weights
            alpha , epsilon = self.get_alpha_epsilon()
            self.epsilon.append(float(epsilon))
            self.alpha_t.append(float(alpha))   #make dict alpha -> model
            # print("# DEBUG: mean pred: ", np.mean(train_prediction_vector))
            # print("# DEBUG: Watch alpha", self.alpha_t)
            # print("# DEBUG: Watch epsilon", self.epsilon)
            #collect weak classifier
            self.weak_model_trained.append(self.model)


    def factorial(self, array):
        return map(lambda x: math.factorial(x), array)


    def logarithmic(self, array):
        indices = [i for i, x in enumerate(array) if x <= 0]
        for i in indices:
            print("# DEBUG: logarithmic, entries: ", array[i])
        # print("# DEBUG: logarithmic, indices: ", indices)
        return map(lambda x: math.log(x), array)


    def binned_likelihood(self, bkg_binns, tg_binns, mu):
        '''Calculares sigma1 and sigma2 for asimov data set and makes a plot'''
        mu_range = 1.3
        save_path = self.path + "plot/"
        measured = bkg_binns + mu * tg_binns
        for i in measured:
            if i == 0:
                print("Bin with zero events")
        avoid_problems = bkg_binns - (mu_range-mu) * tg_binns
        #remove bins with no bkg events -> they will couse problems due to log
        indices = [i for i, x in enumerate(avoid_problems) if x <= 0]
        print("# DEBUG: binned_likelihood, indices: ", indices)
        measured = np.delete(measured, indices)
        bkg_binns = np.delete(bkg_binns, indices)
        tg_binns = np.delete(tg_binns, indices)
        # print("# DEBUG: binned_likelihood, indices of ==0: ", indices)
        # print("# DEBUG: binned_likelihood, measured.shape: ", measured.shape)
        # print("# DEBUG: bkg_binns.shape : ", bkg_binns.shape)
        # print("# DEBUG: factorial: ", self.factorial(measured))
        # print("# DEBUG: np.log(factorial): ", np.log(self.factorial(measured)))
        # print("# DEBUG: np.log(bkg_binns + mu*tg_binns): ", np.log(bkg_binns + mu*tg_binns))
        # print("# DEBUG: bevore sum: ", 2*(self.logarithmic(self.factorial(measured)) + bkg_binns + mu*tg_binns - bkg_binns*self.logarithmic(bkg_binns + mu*tg_binns)))
        minimum = np.sum(2*(self.logarithmic(self.factorial(measured)) + bkg_binns + mu*tg_binns - measured*self.logarithmic(measured)))
        # print("# DEBUG: binned_likelihood, minimum: ", minimum)
        nxvals = 51     #51/2 hard coded in interpolate
        mu_draw = np.linspace(mu-mu_range, mu+mu_range, nxvals, endpoint = True)
        loglike = np.array([])
        for i in range(0, mu_draw.shape[0]):        #better use while loglike < 2+y_min
            tmp = 2*(self.logarithmic(self.factorial(measured)) + bkg_binns + mu_draw[i]*tg_binns - measured*self.logarithmic(bkg_binns + mu_draw[i]*tg_binns))
            # if i < 4:
            #     print("+ ", self.logarithmic(self.factorial(measured)))
            #     print("+ ", bkg_binns + mu_draw[i]*tg_binns)
            #     print("- ", bkg_binns*self.logarithmic(bkg_binns + mu_draw[i]*tg_binns))
            loglike = np.append(loglike, np.sum(tmp)-minimum)
        #calculate 'sigma1' and 'sigma2'
        #binned likelihood function is invetable when seperated to left and right of its minimum
        # print("# DEBUG: binned_likelihood, loglike: ", loglike)
        # print("# DEBUG: binned_likelihood, maximum: ", np.amax(loglike))
        # print("# DEBUG: shapes in interpolate: ", loglike.shape, mu_draw.shape)
        intp_lefthand = interpolate.interp1d(loglike[:26], mu_draw[:26])
        intp_righthand = interpolate.interp1d(loglike[26:], mu_draw[26:])
        s1x = np.array([intp_lefthand(1), intp_righthand(1)])
        s2x = np.array([intp_lefthand(4), intp_righthand(4)])
        #real sigma1 and sigma2 values
        sigma1 = np.absolute(s1x-mu)
        sigma2 = np.absolute(s2x-mu)
        #sigma1 and sigma2 y value for plotting
        s1y = [1, 1]
        s2y = [4, 4]
        #plotting
        plt.xlabel(r'$\mu$')
        plt.ylabel(r'$- \log L$')
        plt.plot(mu_draw, loglike, 'k-')
        plt.xlim(left=mu_draw[0], right=mu_draw[-1])
        plt.axvline(x=mu, color='k', ls='--', ymin=0., ymax=np.amax(loglike))
        plt.plot(s1x, s1y, 'b-', label = r'$\sigma_1=+{1}-{0}$'.format(round(sigma1[0],3), round(sigma1[1],3)))
        plt.plot(s2x, s2y, 'g-', label = r'$\sigma_2=+{1}-{0}$'.format(round(sigma2[0],3), round(sigma2[1],3)))
        plt.legend(loc='best')
        plt.savefig(save_path + self.name+ "_mu" + str(mu) +"_loglike.pdf")
        plt.clf()
        # print("# DEBUG: binned_likelihood, saved fig: ", save_path + self.name+ "_mu" + str(mu) +"_loglike.pdf")
        #to calculate real sigma1 and sigma2 and return it
        return sigma1, sigma2


    def weight_prediction(self, pred, alpha):
        pred = np.asarray(pred)
        alpha = np.asarray(alpha)
        # print("# DEBUG: weight_prediction, pred.shape: ", pred.shape)
        # print("# DEBUG: weight_prediction, alpha.shape: ", alpha.shape)
        # print("# DEBUG: alpha: ", alpha)
        sum = 0
        # print("# DEBUG: len(alpha): ", len(alpha))
        # print("# DEBUG: weight_prediction, initial pred: ", pred[:len(alpha),0:5])
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
        # print("# DEBUG: weight_prediction, final pred: ", pred[:len(alpha),0:5])
        return pred[0:len(alpha)]


    def strong_classification(self, pred, alpha):
        '''builds prediciton vector for strong classifier'''
        pred_array = self.weight_prediction(pred, alpha)
        # print("# DEBUG: pred_array: ", pred_array[len(alpha)-1,0:5])
        prediction_vector = np.sum(pred_array, axis = 0)
        # print("# DEBUG: prediction_vector: ", prediction_vector[0:5])
        # print("# DEBUG: labels: ", self.train_label[0:5])
        prediction_vector_disc = np.array([])
        for x in prediction_vector:
            if x<self.cut_value:
                prediction_vector_disc = np.append(prediction_vector_disc, int(self.binary_bkg_target))
            else:
                prediction_vector_disc = np.append(prediction_vector_disc, int(1))
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
            # print("# DEBUG: eval_model, train_prediction_vector: ", self.train_prediction_vector[i][0:5])
            train_prediction, train_prediction_disc = self.strong_classification(self.train_prediction_vector, self.alpha_t[0:i+1])
            test_prediction, test_prediction_disc = self.strong_classification(self.test_prediction_vector, self.alpha_t[0:i+1])
            # print("# DEBUG: train_prediction: ", train_prediction[0:5])
            # print("# DEBUG: train_prediction_disc: ", train_prediction_disc[0:5])
            # print("# DEBUG: count_nonzero: ", np.count_nonzero(test_prediction_final==self.test_label))
            train_fraction = np.append(train_fraction,
                        np.count_nonzero(train_prediction_disc==self.train_label)/float(self.train_label.shape[0]))
            test_fraction = np.append(test_fraction,
                        np.count_nonzero(test_prediction_disc==self.test_label)/float(self.test_label.shape[0]))
        #needed for discriminator plot
        self.model_prediction_vector = test_prediction
        #get roc
        fpr, tpr, thresholds = metrics.roc_curve(self.test_label, test_prediction)
        self.roc_auc = metrics.auc(fpr, tpr)
        #plot
        epoches = np.arange(1, len(self.train_prediction_vector)+1)
        print("# DEBUG: fraction: ", train_fraction, test_fraction)
        best_train = np.amax(train_fraction)
        best_test = np.amax(test_fraction)

        plt.figure(1)
        plt.plot(epoches, train_fraction, 'r-', label = "Trainingsdaten - Max: " + str(round(best_train, 3)))
        plt.plot(epoches, test_fraction, 'b-', label = "Testdaten - Max: " + str(round(best_test, 3)))
        # plt.title("Anteil richtig Bestimmt - AdaBoost_binary_discret")
        plt.xlabel("Ada-Epochen")
        plt.ylabel("Anteil Richtig Bestimmt")
        plt.legend(loc='best')
        plt.savefig(save_path + self.name + "_frac.pdf")

        plt.figure(2)
        # plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % self.roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(save_path + self.name +"_roc.pdf")

        plt.figure(3)
        # plt.title('Epsilon')
        plt.xlabel("Epochen")
        plt.ylabel("Epsilon")
        plt.plot(epoches, self.epsilon, '-')
        plt.savefig(save_path + self.name +"_eps.pdf")
        plt.clf()

        # self.binned_likelihood()
        # plt.show()


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

        # save final prediciton_vector
        np.save(save_path + "pred_vec", self.model_prediction_vector)

        # save roc-aux score
        np.save(save_path + "roc", np.array([self.roc_auc]))

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


    # --------------------------------------------------------------------
    # result plotting functions
    # --------------------------------------------------------------------
    def plot_binaryOutput(self, log = False, privateWork = False, printROC = False,
                        nbins = 20, bin_range = [-1.,1.], name = "binary discriminator"):

        binaryOutput = plottingScripts.plotBinaryOutput(
            data                = self.data,
            predictions         = self.model_prediction_vector,
            nbins               = nbins,
            bin_range           = bin_range,
            event_category      = self.categoryLabel,
            plotdir             = self.path + "plot/",
            pltname             = self.name,
            logscale            = log)

        bkg_hist, sig_hist = binaryOutput.plot(ratio = True, printROC = printROC, privateWork = privateWork, name = name)
        # print("# DEBUG: plot_binaryOutput, histos: ", bkg_hist, sig_hist)
        print("sigma: ", self.binned_likelihood(bkg_hist, sig_hist, 0))
        print("sigma: ", self.binned_likelihood(bkg_hist, sig_hist, 1))




    # def plot_discriminators(self, log = False, printROC = False, privateWork = False,
    #                     signal_class = None, nbins = 20, bin_range = [-1.,1.]):
    #
    #     ''' plot all events classified as one category '''
    #     plotDiscrs = plottingScripts.plotDiscriminators(
    #         data                = self.data,
    #         prediction_vector   = self.model_prediction_vector,
    #         event_classes       = self.event_classes,
    #         nbins               = nbins,
    #         bin_range           = bin_range,
    #         signal_class        = signal_class,
    #         event_category      = self.categoryLabel,
    #         plotdir             = self.path + "plot/",
    #         logscale            = log)
    #
    #     plotDiscrs.plot(ratio = False, printROC = printROC, privateWork = privateWork)
