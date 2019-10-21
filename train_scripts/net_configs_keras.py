import keras
import keras.models as models
import numpy as np
import tensorflow as tf


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


def binary_DL_SGD_tensorflow(input_placeholder, keras_model):
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
