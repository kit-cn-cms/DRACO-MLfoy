import os
import sys
import numpy as np

# local imports
filedir  = os.path.dirname(os.path.realpath(__file__))
DRACOdir = os.path.dirname(filedir)
basedir  = os.path.dirname(DRACOdir)
sys.path.append(basedir)

from pyrootsOfTheCaribbean.evaluationScripts import plottingScripts

# imports with keras
import data_frame

import keras
import keras.models as models
import keras.layers as layer
from keras import backend as K
import pandas as pd
from sklearn.utils import shuffle

# Limit gpu usage
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))

def get_indices(variable_list, particles):
    indices = []
    for p in particles:
        eIdx = variable_list.index(p+"_E")
        xIdx = variable_list.index(p+"_Px")
        yIdx = variable_list.index(p+"_Py")
        zIdx = variable_list.index(p+"_Pz")
        indices.append([eIdx, xIdx, yIdx, zIdx])
    print(indices)
    return indices


# implementation of customary mass fitting loss
def mass_reconstruction(y_true, y_pred, indices, feature_scaling):
    # idices is a list of lists containing indices of four vectors in y vectors
    f_true = K.flatten(y_true*feature_scaling)
    f_pred = K.flatten(y_pred*feature_scaling)
    loss_value = 0.     
    for particle in indices:
        tru_e  = f_true[particle[0]]
        tru_e2 = K.square(tru_e)
        tru_x  = f_true[particle[1]]
        tru_x2 = K.square(tru_x)
        tru_y  = f_true[particle[2]]
        tru_y2 = K.square(tru_y)
        tru_z  = f_true[particle[3]]
        tru_z2 = K.square(tru_z)

        tru_m = K.sqrt(tru_e2 - tru_x2 - tru_y2 - tru_z2)

        pre_e  = f_pred[particle[0]]
        pre_e2 = K.square(pre_e)
        pre_x  = f_pred[particle[1]]
        pre_x2 = K.square(pre_x)
        pre_y  = f_pred[particle[2]]
        pre_y2 = K.square(pre_y)
        pre_z  = f_pred[particle[3]]
        pre_z2 = K.square(pre_z)

        pre_m = K.sqrt(pre_e2 - pre_x2 - pre_y2 - pre_z2)

        loss_value += K.square(tru_m - pre_m) + K.square(tru_e - pre_e) + K.square(tru_x - pre_x) + K.square(tru_y - pre_y) + K.square(tru_z - pre_z)
    return loss_value 



def mschrode_kit_cool_loss(indices, feature_scaling = 1., mode = 1):
    if mode == 1:
        def loss(y_true, y_pred):
            return mass_reconstruction(y_true, y_pred, indices, feature_scaling)
    else:
        exit("no valid mode for massive loss")
    return loss



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

class BigChungusGenerator(object):
    def __init__(self, data, inputs, targets, shuffle = True, save_targets = False):
        self.data = data
        self.max_entries = self.data.shape[0]
        self.shuffle = shuffle        

        self.inputs = inputs
        self.targets = targets

        self.current_idx = 0

        self.generateSequenceLength()
    
        self.save_targets = save_targets
        if self.save_targets:
            self.targetValues = []

    def generateSequenceLength(self):
        # generate sequence length
        # loop over all objects, add min(indexVariable, maxValues) 
        self.data = self.data.assign(sequence_length = lambda row: sum([
            row[obj["index"]].apply(lambda x: x if x<obj["max"] else obj["max"]) if obj["index"] else 1
            for obj in self.inputs.objects]) )
    
    def getTargets(self):
        if not self.save_targets: return None
    
        return np.array(self.targetValues).reshape(-1,self.targets.nTargets)
        
    def generate(self):
        while True:
            if self.current_idx >= self.max_entries:
                self.current_idx = 0
                self.data = shuffle(self.data)
                if self.save_targets: self.targetValues = []

            evt = self.data.iloc[self.current_idx]
            self.current_idx += 1

            variables = []
            for obj in self.inputs.objects:
                if obj["index"]:
                    variables += self.inputs.objectVariables[obj["name"]][:int(self.inputs.nVariables*evt[obj["index"]])]
                else:
                    variables += self.inputs.objectVariables[obj["name"]]

            train_x = evt[variables].values.reshape((-1,int(evt["sequence_length"]),self.inputs.nVariables))
            if self.shuffle:
                train_x = shuffle(train_x)

            train_y = evt[self.targets.targetVariables].values.reshape((-1,self.targets.nTargets))
            if self.save_targets: self.targetValues.append(train_y)

            #print(train_x)
            #print(train_y)
            #raw_input()
        
            yield train_x, train_y

            
            


class ttbarMatcher():
    def __init__(self, save_path, input_samples, input_features, target_features, loss_function = "mean_squared_error", shuffle_inputs = True, feature_scaling = 1000., val_percentage = 0.2, test_percentage = 0.2, n_epochs = 50):
        # save some information
        # list of samples to load into dataframe
        self.input_samples = input_samples
        self.test_percentage = test_percentage
        self.val_percentage = val_percentage

        # output directory for results
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs( self.save_path )

        # set variables
        self.inputs = input_features
        self.targets = target_features
        self.feature_scaling = feature_scaling

        self.shuffle_inputs = shuffle_inputs

        self.loss_function = loss_function

        # load data set
        self.data = self._load_datasets()

        self.train_chungus = BigChungusGenerator(
            self.data.df_train,
            self.inputs,
            self.targets,
            shuffle = self.shuffle_inputs)

        self.val_chungus = BigChungusGenerator(
            self.data.df_val,
            self.inputs,
            self.targets,
            shuffle = self.shuffle_inputs)

        self.test_chungus = BigChungusGenerator(
            self.data.df_test,
            self.inputs,
            self.targets,
            save_targets = True,
            shuffle = self.shuffle_inputs)

        # hyper parameters
        self.n_epochs = n_epochs

    def _load_datasets(self):
        ''' load data set '''

        return data_frame.DataFrame(
            input_samples       = self.input_samples,
            input_features      = self.inputs,
            target_features     = self.targets,
            feature_scaling     = self.feature_scaling,
            test_percentage     = self.test_percentage,
            val_percentage      = self.val_percentage)


    def build_default_model(self):
        ''' build some example model '''
        model = models.Sequential()

        neurons = 100
        model.add( layer.LSTM(
            units = 200,
            input_shape = (None, self.inputs.nVariables),
            #activation = "linear",
            return_sequences = True))
        
        model.add( layer.LSTM(
            units = 200,
            return_sequences = True))

        model.add( layer.LSTM(
            units = self.targets.nTargets,
            #activation = "linear",
            return_sequences = False))

        return model
        



    def build_model(self, model = None):
        if model == None:
            print("Loading default model")
            model = self.build_default_model()

        # loss function
        if "mschrode_kit_cool_loss" in self.loss_function:
            four_vector_indices = get_indices(
                variable_list = self.targets.targetVariables,
                particles     = self.targets.objects                
                )

            if self.loss_function == "mschrode_kit_cool_loss":
                mode = 1
            else:
                mode = int(self.loss_function.split("#")[-1])

            self.loss_function = mschrode_kit_cool_loss(four_vector_indices, self.feature_scaling, mode = mode)

        # compile the model
        model.compile(
            loss = self.loss_function,
            optimizer = "ADAM")

        # save the model
        self.model = model

        # print model summary
        self.model.summary(100)

        # save net information
        out_file = self.save_path+"/model_summary.yml"
        yml_model = self.model.to_yaml()
        with open(out_file, "w") as f:
            f.write(yml_model)

    def train_model(self):
        ''' train the model '''
        self.model.fit_generator( 
            self.train_chungus.generate(), 
            steps_per_epoch = self.data.df_train.shape[0], 
            epochs = self.n_epochs,
            validation_data = self.val_chungus.generate(),
            validation_steps = self.data.df_val.shape[0], 
            verbose = 1)

    def eval_model(self):
        predictions = self.model.predict_generator(
            self.test_chungus.generate(),
            steps = self.data.df_test.shape[0],
            verbose = 1)

        print(predictions)
        truths = np.array(self.test_chungus.getTargets())

        for y, yhat in zip(predictions, truths):
            self.dump_evaluation(y, yhat)
        
    def dump_evaluation(self, pred, tru):
        pred*= self.feature_scaling
        tru *= self.feature_scaling

        print("\n"+"="*50)
        for i, obj in enumerate(self.targets.objects):
            print("reconstruction of {}:".format(obj))
            print("{} | {} | {} | {}".format("variable", "prediction", "truth", "absolute distance"))
            for j, var in enumerate(self.targets.variables):
                pred_val = pred[i*self.targets.nVariables+j]
                tru_val = tru[i*self.targets.nVariables+j]
                dist = np.abs(pred_val - tru_val)
                print("{} | {} | {} | {}".format(var, pred_val, tru_val, dist))

            # reconstruct mass
            e_idx = self.targets.targetVariables.index(obj+"_E")
            x_idx = self.targets.targetVariables.index(obj+"_Px")
            y_idx = self.targets.targetVariables.index(obj+"_Py")
            z_idx = self.targets.targetVariables.index(obj+"_Pz")

            reco_mass = get_mass( pred, e_idx, x_idx, y_idx, z_idx )
            gen_mass = get_mass( tru, e_idx, x_idx, y_idx, z_idx )
            print("reconstructed mass: {}".format(reco_mass))
            print("gen level mass:     {}".format(gen_mass))
            print("- "*10)

        print("="*50+"\n")


def get_mass(values, e, x, y, z):
    p2 = values[x]**2 + values[y]**2 + values[z]**2
    e2 = values[e]**2
    m2 = e2 - p2
    if m2 < 0.: return -np.sqrt(-m2)
    else: return np.sqrt(m2)



