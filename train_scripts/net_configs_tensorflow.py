import keras
import keras.models as models
import numpy as np
import tensorflow as tf


def example_config_tensorflow(input_placeholder, keras_model):
    # Get weights as numpy arrays
    weights = {}
    for layer in keras_model.layers:
        for weight, array in zip(layer.weights, layer.get_weights()):
            weights[weight.name] = np.array(array)

    # Load weights in tensorflow variables
    w1 = tf.get_variable('w1', initializer=weights['DenseLayer_0_1/kernel:0'])
    b1 = tf.get_variable('b1', initializer=weights['DenseLayer_0_1/bias:0'])
    w2 = tf.get_variable('w2', initializer=weights['DenseLayer_1_1/kernel:0'])
    b2 = tf.get_variable('b2', initializer=weights['DenseLayer_1_1/bias:0'])
    w3 = tf.get_variable('w3', initializer=weights['outputLayer_1/kernel:0'])
    b3 = tf.get_variable('b3', initializer=weights['outputLayer_1/bias:0'])


    # Build tensorflow graph with weights from keras model
    l1 = tf.nn.elu(tf.add(b1, tf.matmul(input_placeholder, w1)))
    l2 = tf.nn.elu(tf.add(b2, tf.matmul(l1, w2)))

    f = tf.nn.softmax(tf.add(b3, tf.matmul(l2, w3)))
    return f


def test_config(input_placeholder, keras_model):
    # Get weights as numpy arrays
    weights = {}
    for layer in keras_model.layers:
        for weight, array in zip(layer.weights, layer.get_weights()):
            weights[weight.name] = np.array(array)

    # Load weights in tensorflow variables
    w1 = tf.get_variable('w1', initializer=weights['DenseLayer_0_1/kernel:0'])
    b1 = tf.get_variable('b1', initializer=weights['DenseLayer_0_1/bias:0'])
    w2 = tf.get_variable('w2', initializer=weights['DenseLayer_1_1/kernel:0'])
    b2 = tf.get_variable('b2', initializer=weights['DenseLayer_1_1/bias:0'])
    w3 = tf.get_variable('w3', initializer=weights['outputLayer_1/kernel:0'])
    b3 = tf.get_variable('b3', initializer=weights['outputLayer_1/bias:0'])
    w4 = tf.get_variable('w4', initializer=weights['outputLayer_1/kernel:0'])
    b4 = tf.get_variable('b4', initializer=weights['outputLayer_1/bias:0'])
    w5 = tf.get_variable('w5', initializer=weights['outputLayer_1/kernel:0'])
    b5 = tf.get_variable('b5', initializer=weights['outputLayer_1/bias:0'])


    # Build tensorflow graph with weights from keras model
    l1 = tf.nn.elu(tf.add(b1, tf.matmul(input_placeholder, w1)))
    l2 = tf.nn.elu(tf.add(b2, tf.matmul(l1, w2)))
    l3 = tf.nn.elu(tf.add(b3, tf.matmul(l2, w3)))
    l4 = tf.nn.elu(tf.add(b4, tf.matmul(l3, w4)))

    f = tf.nn.softmax(tf.add(b5, tf.matmul(l4, w5)))
    return f

def ttZ_2018_final_tensorflow(input_placeholder, keras_model):
    # Get weights as numpy arrays
    weights = {}
    for layer in keras_model.layers:
        for weight, array in zip(layer.weights, layer.get_weights()):
            weights[weight.name] = np.array(array)

    # Load weights in tensorflow variables
    w1 = tf.get_variable('w1', initializer=weights['DenseLayer_0_1/kernel:0'])
    b1 = tf.get_variable('b1', initializer=weights['DenseLayer_0_1/bias:0'])
    w2 = tf.get_variable('w2', initializer=weights['DenseLayer_1_1/kernel:0'])
    b2 = tf.get_variable('b2', initializer=weights['DenseLayer_1_1/bias:0'])
    w3 = tf.get_variable('w3', initializer=weights['outputLayer_1/kernel:0'])
    b3 = tf.get_variable('b3', initializer=weights['outputLayer_1/bias:0'])


    # Build tensorflow graph with weights from keras model
    l1 = tf.nn.leaky_relu(tf.add(b1, tf.matmul(input_placeholder, w1)))
    l2 = tf.nn.leaky_relu(tf.add(b2, tf.matmul(l1, w2)))

    f = tf.nn.softmax(tf.add(b3, tf.matmul(l2, w3)))
    return f


def ttH_2017_tensorflow(input_placeholder, keras_model):
    # Get weights as numpy arrays
    weights = {}
    for layer in keras_model.layers:
        for weight, array in zip(layer.weights, layer.get_weights()):
            weights[weight.name] = np.array(array)

    # Load weights in tensorflow variables
    w1 = tf.get_variable('w1', initializer=weights['DenseLayer_0_1/kernel:0'])
    b1 = tf.get_variable('b1', initializer=weights['DenseLayer_0_1/bias:0'])
    w2 = tf.get_variable('w2', initializer=weights['DenseLayer_1_1/kernel:0'])
    b2 = tf.get_variable('b2', initializer=weights['DenseLayer_1_1/bias:0'])
    w3 = tf.get_variable('w3', initializer=weights['outputLayer_1/kernel:0'])
    b3 = tf.get_variable('b3', initializer=weights['outputLayer_1/bias:0'])
    w4 = tf.get_variable('w4', initializer=weights['outputLayer_1/kernel:0'])
    b4 = tf.get_variable('b4', initializer=weights['outputLayer_1/bias:0'])


    # Build tensorflow graph with weights from keras model
    l1 = tf.nn.elu(tf.add(b1, tf.matmul(input_placeholder, w1)))
    l2 = tf.nn.elu(tf.add(b2, tf.matmul(l1, w2)))
    l3 = tf.nn.elu(tf.add(b3, tf.matmul(l2, w3)))

    f = tf.nn.softmax(tf.add(b4, tf.matmul(l3, w4)))
    return f


def Legacy_ttH_2017_tensorflow(input_placeholder, keras_model):
    # Get weights as numpy arrays
    weights = {}
    for layer in keras_model.layers:
        for weight, array in zip(layer.weights, layer.get_weights()):
            weights[weight.name] = np.array(array)

    # Load weights in tensorflow variables
    w1 = tf.get_variable('w1', initializer=weights['DenseLayer_0_1/kernel:0'])
    b1 = tf.get_variable('b1', initializer=weights['DenseLayer_0_1/bias:0'])
    w2 = tf.get_variable('w2', initializer=weights['DenseLayer_1_1/kernel:0'])
    b2 = tf.get_variable('b2', initializer=weights['DenseLayer_1_1/bias:0'])
    w3 = tf.get_variable('w3', initializer=weights['outputLayer_1/kernel:0'])
    b3 = tf.get_variable('b3', initializer=weights['outputLayer_1/bias:0'])
    w4 = tf.get_variable('w4', initializer=weights['outputLayer_1/kernel:0'])
    b4 = tf.get_variable('b4', initializer=weights['outputLayer_1/bias:0'])


    # Build tensorflow graph with weights from keras model
    l1 = tf.nn.elu(tf.add(b1, tf.matmul(input_placeholder, w1)))
    l2 = tf.nn.elu(tf.add(b2, tf.matmul(l1, w2)))
    l3 = tf.nn.elu(tf.add(b3, tf.matmul(l2, w3)))

    f = tf.nn.softmax(tf.add(b4, tf.matmul(l3, w4)))
    return f


def ttH_2017_baseline_tensorflow(input_placeholder, keras_model):
    # Get weights as numpy arrays
    weights = {}
    for layer in keras_model.layers:
        for weight, array in zip(layer.weights, layer.get_weights()):
            weights[weight.name] = np.array(array)

    # Load weights in tensorflow variables
    w1 = tf.get_variable('w1', initializer=weights['DenseLayer_0_1/kernel:0'])
    b1 = tf.get_variable('b1', initializer=weights['DenseLayer_0_1/bias:0'])
    w2 = tf.get_variable('w2', initializer=weights['DenseLayer_1_1/kernel:0'])
    b2 = tf.get_variable('b2', initializer=weights['DenseLayer_1_1/bias:0'])
    w3 = tf.get_variable('w3', initializer=weights['outputLayer_1/kernel:0'])
    b3 = tf.get_variable('b3', initializer=weights['outputLayer_1/bias:0'])
    w4 = tf.get_variable('w4', initializer=weights['outputLayer_1/kernel:0'])
    b4 = tf.get_variable('b4', initializer=weights['outputLayer_1/bias:0'])


    # Build tensorflow graph with weights from keras model
    l1 = tf.nn.elu(tf.add(b1, tf.matmul(input_placeholder, w1)))
    l2 = tf.nn.elu(tf.add(b2, tf.matmul(l1, w2)))
    l3 = tf.nn.elu(tf.add(b3, tf.matmul(l2, w3)))

    f = tf.nn.softmax(tf.add(b4, tf.matmul(l3, w4)))
    return f


def legacy_2018_tensorflow(input_placeholder, keras_model):
    # Get weights as numpy arrays
    weights = {}
    for layer in keras_model.layers:
        for weight, array in zip(layer.weights, layer.get_weights()):
            weights[weight.name] = np.array(array)

    # Load weights in tensorflow variables
    w1 = tf.get_variable('w1', initializer=weights['DenseLayer_0_1/kernel:0'])
    b1 = tf.get_variable('b1', initializer=weights['DenseLayer_0_1/bias:0'])
    w2 = tf.get_variable('w2', initializer=weights['DenseLayer_1_1/kernel:0'])
    b2 = tf.get_variable('b2', initializer=weights['DenseLayer_1_1/bias:0'])
    w3 = tf.get_variable('w3', initializer=weights['outputLayer_1/kernel:0'])
    b3 = tf.get_variable('b3', initializer=weights['outputLayer_1/bias:0'])
    w4 = tf.get_variable('w4', initializer=weights['outputLayer_1/kernel:0'])
    b4 = tf.get_variable('b4', initializer=weights['outputLayer_1/bias:0'])


    # Build tensorflow graph with weights from keras model
    l1 = tf.nn.elu(tf.add(b1, tf.matmul(input_placeholder, w1)))
    l2 = tf.nn.elu(tf.add(b2, tf.matmul(l1, w2)))
    l3 = tf.nn.elu(tf.add(b3, tf.matmul(l2, w3)))

    f = tf.nn.softmax(tf.add(b4, tf.matmul(l3, w4)))
    return f


def dnn_config_tensorflow(input_placeholder, keras_model):
    # Get weights as numpy arrays
    weights = {}
    for layer in keras_model.layers:
        for weight, array in zip(layer.weights, layer.get_weights()):
            weights[weight.name] = np.array(array)

    # Load weights in tensorflow variables
    w1 = tf.get_variable('w1', initializer=weights['DenseLayer_0_1/kernel:0'])
    b1 = tf.get_variable('b1', initializer=weights['DenseLayer_0_1/bias:0'])
    w2 = tf.get_variable('w2', initializer=weights['outputLayer_1/kernel:0'])
    b2 = tf.get_variable('b2', initializer=weights['outputLayer_1/bias:0'])

    # Build tensorflow graph with weights from keras model
    l1 = tf.nn.elu(tf.add(b1, tf.matmul(input_placeholder, w1)))
    f = tf.nn.softmax(tf.add(b3, tf.matmul(l2, w3)))
    return f

def binary_config_tensorflow(input_placeholder, keras_model):
    # Get weights as numpy arrays
    weights = {}
    for layer in keras_model.layers:
        for weight, array in zip(layer.weights, layer.get_weights()):
            weights[weight.name] = np.array(array)

    # Load weights in tensorflow variables
    w1 = tf.get_variable('w1', initializer=weights['DenseLayer_0_1/kernel:0'])
    b1 = tf.get_variable('b1', initializer=weights['DenseLayer_0_1/bias:0'])
    w2 = tf.get_variable('w2', initializer=weights['DenseLayer_1_1/kernel:0'])
    b2 = tf.get_variable('b2', initializer=weights['DenseLayer_1_1/bias:0'])
    w3 = tf.get_variable('w3', initializer=weights['outputLayer_1/kernel:0'])
    b3 = tf.get_variable('b3', initializer=weights['outputLayer_1/bias:0'])

    # Build tensorflow graph with weights from keras model
    l1 = tf.nn.selu(tf.add(b1, tf.matmul(input_placeholder, w1)))
    l2 = tf.nn.selu(tf.add(b2, tf.matmul(l1, w2)))
    f = tf.nn.softmax(tf.add(b3, tf.matmul(l2, w3)))
    return f


def ttH_2017_DL_tensorflow(input_placeholder, keras_model):
    # Get weights as numpy arrays
    weights = {}
    for layer in keras_model.layers:
        for weight, array in zip(layer.weights, layer.get_weights()):
            weights[weight.name] = np.array(array)

    # Load weights in tensorflow variables
    w1 = tf.get_variable('w1', initializer=weights['DenseLayer_0_1/kernel:0'])
    b1 = tf.get_variable('b1', initializer=weights['DenseLayer_0_1/bias:0'])
    w2 = tf.get_variable('w2', initializer=weights['DenseLayer_1_1/kernel:0'])
    b2 = tf.get_variable('b2', initializer=weights['DenseLayer_1_1/bias:0'])
    w3 = tf.get_variable('w3', initializer=weights['outputLayer_1/kernel:0'])
    b3 = tf.get_variable('b3', initializer=weights['outputLayer_1/bias:0'])

    # Build tensorflow graph with weights from keras model
    l1 = tf.nn.relu(tf.add(b1, tf.matmul(input_placeholder, w1)))
    l2 = tf.nn.relu(tf.add(b2, tf.matmul(l1, w2)))
    f = tf.sigmoid(tf.add(b3, tf.matmul(l2, w3)))
    return f


def binary_DL_tensorflow(input_placeholder, keras_model):
    # Get weights as numpy arrays
    weights = {}
    for layer in keras_model.layers:
        for weight, array in zip(layer.weights, layer.get_weights()):
            weights[weight.name] = np.array(array)

    # Load weights in tensorflow variables
    w1 = tf.get_variable('w1', initializer=weights['DenseLayer_0_1/kernel:0'])
    b1 = tf.get_variable('b1', initializer=weights['DenseLayer_0_1/bias:0'])
    w2 = tf.get_variable('w2', initializer=weights['DenseLayer_1_1/kernel:0'])
    b2 = tf.get_variable('b2', initializer=weights['DenseLayer_1_1/bias:0'])
    w3 = tf.get_variable('w3', initializer=weights['outputLayer_1/kernel:0'])
    b3 = tf.get_variable('b3', initializer=weights['outputLayer_1/bias:0'])

    # Build tensorflow graph with weights from keras model
    l1 = tf.nn.relu(tf.add(b1, tf.matmul(input_placeholder, w1)))
    l2 = tf.nn.relu(tf.add(b2, tf.matmul(l1, w2)))
    f = tf.sigmoid(tf.add(b3, tf.matmul(l2, w3)))
    return f


def binary_DL_SGD_cate8_tensorflow(input_placeholder, keras_model):
    # Get weights as numpy arrays
    weights = {}
    for layer in keras_model.layers:
        for weight, array in zip(layer.weights, layer.get_weights()):
            weights[weight.name] = np.array(array)

    # Load weights in tensorflow variables
    w1 = tf.get_variable('w1', initializer=weights['DenseLayer_0_1/kernel:0'])
    b1 = tf.get_variable('b1', initializer=weights['DenseLayer_0_1/bias:0'])
    w2 = tf.get_variable('w2', initializer=weights['DenseLayer_1_1/kernel:0'])
    b2 = tf.get_variable('b2', initializer=weights['DenseLayer_1_1/bias:0'])
    w3 = tf.get_variable('w3', initializer=weights['outputLayer_1/kernel:0'])
    b3 = tf.get_variable('b3', initializer=weights['outputLayer_1/bias:0'])

    # Build tensorflow graph with weights from keras model
    l1 = tf.tanh(tf.add(b1, tf.matmul(input_placeholder, w1)))
    l2 = tf.tanh(tf.add(b2, tf.matmul(l1, w2)))
    f = tf.tanh(tf.add(b3, tf.matmul(l2, w3)))
    return f
