import types
# Dependency imports
import numpy as np

from tensorflow.keras import optimizers
import tensorflow_probability as tfp
import tensorflow_probability.python.distributions as tfd
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.distributions import deterministic as deterministic_lib
from tensorflow_probability.python.distributions import independent as independent_lib
from tensorflow_probability.python.distributions import normal as normal_lib
from tensorflow.python.keras.utils import generic_utils  # pylint: disable=g-direct-tensorflow-import
from functools import partial, update_wrapper

def wrapped_partial(func, *args, **kwargs):
        partial_func = partial(func, *args, **kwargs)
        update_wrapper(partial_func, func)
        return partial_func

def default_loc_scale_fn(
    is_singular=False,
    non_trainable_transformed_std=None,
    loc_initializer=tf1.initializers.random_normal(stddev=0.1),
    untransformed_scale_initializer=tf1.initializers.random_normal(
        mean=-3., stddev=0.1),
    loc_regularizer=None,
    untransformed_scale_regularizer=None,
    loc_constraint=None,
    untransformed_scale_constraint=None):

  def _fn(dtype, shape, name, trainable, add_variable_fn):
    """Creates `loc`, `scale` parameters."""
    loc = add_variable_fn(
        name=name + '_loc',
        shape=shape,
        initializer=loc_initializer,
        regularizer=loc_regularizer,
        constraint=loc_constraint,
        dtype=dtype,
        trainable=trainable)
    if is_singular:
      return loc, None
    if non_trainable_transformed_std is not None: 
      return loc, non_trainable_transformed_std
    untransformed_scale = add_variable_fn(
        name=name + '_untransformed_scale',
        shape=shape,
        initializer=untransformed_scale_initializer,
        regularizer=untransformed_scale_regularizer,
        constraint=untransformed_scale_constraint,
        dtype=dtype,
        trainable=trainable)
    scale = tfp_util.DeferredTensor(
        untransformed_scale,
        lambda x: (np.finfo(dtype.as_numpy_dtype).eps + tf.nn.softplus(x)))
    return loc, scale
  return _fn

def default_mean_field_normal_fn(
    is_singular=False,
    non_trainable_transformed_std=None,
    loc_initializer=tf1.initializers.random_normal(stddev=0.1),
    untransformed_scale_initializer=tf1.initializers.random_normal(
        mean=-3., stddev=0.1),
    loc_regularizer=None,
    untransformed_scale_regularizer=None,
    loc_constraint=None,
    untransformed_scale_constraint=None):

  loc_scale_fn = default_loc_scale_fn(
      is_singular=is_singular,
      non_trainable_transformed_std=non_trainable_transformed_std,
      loc_initializer=loc_initializer,
      untransformed_scale_initializer=untransformed_scale_initializer,
      loc_regularizer=loc_regularizer,
      untransformed_scale_regularizer=untransformed_scale_regularizer,
      loc_constraint=loc_constraint,
      untransformed_scale_constraint=untransformed_scale_constraint)

  def _fn(dtype, shape, name, trainable, add_variable_fn):
    loc, scale = loc_scale_fn(dtype, shape, name, trainable, add_variable_fn)
    if scale is None:
      dist = deterministic_lib.Deterministic(loc=loc)
    else:
      dist = normal_lib.Normal(loc=loc, scale=scale)
    batch_ndims = tf.size(dist.batch_shape_tensor())
    return independent_lib.Independent(
        dist, reinterpreted_batch_ndims=batch_ndims)
  return _fn

def default_multivariate_normal_fn(dtype, shape, name, trainable,
                                   add_variable_fn, scale=1):
  """Creates multivariate standard `Normal` distribution.
  Args:
    dtype: Type of parameter's event.
    shape: Python `list`-like representing the parameter's event shape.
    name: Python `str` name prepended to any created (or existing)
      `tf.Variable`s.
    trainable: Python `bool` indicating all created `tf.Variable`s should be
      added to the graph collection `GraphKeys.TRAINABLE_VARIABLES`.
    add_variable_fn: `tf.get_variable`-like `callable` used to create (or
      access existing) `tf.Variable`s.
  Returns:
    Multivariate standard `Normal` distribution.
  """
  del name, trainable, add_variable_fn   # unused
  dist = normal_lib.Normal(
      loc=tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype(scale))
  batch_ndims = tf.size(dist.batch_shape_tensor())
  return independent_lib.Independent(
      dist, reinterpreted_batch_ndims=batch_ndims)
      
config_dict = {}

config_dict["example_config"] = {
        "layers":                   [200,200],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.5,
        "L2_Norm":                  1e-5,
        "batch_size":               5000,
        "optimizer":                optimizers.Adagrad(decay=0.99),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     50,
        }

config_dict["test_config"] = {
        "layers":                   [1000,1000,200,200],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.,
        "L2_Norm":                  0.,
        "batch_size":               5000,
        "optimizer":                optimizers.Adagrad(decay=0.99),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     50,
        }

config_dict["ttH_SL_legacy"] = {
        "layers":                   [50,50],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.5,
        "L2_Norm":                  1e-5,
        "L1_Norm":                  1e-5,
        "batch_size":               1000,
        "optimizer":                optimizers.Adagrad(),
        "activation_function":      "leakyrelu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     50,
        }

config_dict["ttZ_2018_final"] = {
        "layers":                   [50,50],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.5,
        "L2_Norm":                  1e-5,
        "L1_Norm":                  1e-5,
        "batch_size":               1000,
        "optimizer":                optimizers.Adagrad(),
        "activation_function":      "leakyrelu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.1,
        "earlystopping_epochs":     50,
        }

config_dict["ttH_2017"] = {
        "layers":                   [100,100,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.50,
        "L1_Norm":                  0.,
        "L2_Norm":                  1e-5,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-4),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     100,
        }

config_dict["Legacy_ttH_2017"] = {
        "layers":                   [100,100,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.50,
        "L2_Norm":                  1e-5,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-4),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     100,
        }

config_dict["ttH_2017_baseline"] = {
        "layers":                   [100,100,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.50,
        "L2_Norm":                  1e-5,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-4),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.1,
        "earlystopping_epochs":     100,
        }

config_dict["legacy_2018"] = {
        "layers":                   [50,50],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.20,
        "L2_Norm":                  1e-5,
        "batch_size":               512,
        "optimizer":                optimizers.Adadelta(),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     50,
        }

config_dict["dnn_config"] = {
        "layers":                   [20],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.1,
        "L2_Norm":                  0.,
        "batch_size":               2000,
        "optimizer":                optimizers.Adadelta(),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     50,
        }


config_dict["ttH_2017_DL"] = {
        "layers":                   [200,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.30,
        "L1_Norm":                  0.,
        "L2_Norm":                  1e-3,
        "batch_size":               64,
        "optimizer":                optimizers.Adadelta(),
        "activation_function":      "relu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage":  0.02,
        "earlystopping_epochs":      100,
        }

config_dict["binary_crossentropy_Adam"] = {
        "layers":                   [150],
        "loss_function":            "binary_crossentropy",
        "Dropout":                  0.4,
        "L1_Norm":                  0.,
        "L2_Norm":                  1e-3,
        "batch_size":               2000,
        "optimizer":                optimizers.Adam(learning_rate=1e-3),
        "activation_function":      "relu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage":  0.02,
        "earlystopping_epochs":      150,
}

config_dict["binary_crossentropy_Adam_modified"] = {
        "layers":                   [150],
        "loss_function":            "binary_crossentropy",
        "Dropout":                  0.,
        "L1_Norm":                  0.,
        "L2_Norm":                  1e-4,
        "batch_size":               2000,
        "optimizer":                optimizers.Adam(learning_rate=1e-3),
        "activation_function":      "relu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage":  0.02,
        "earlystopping_epochs":      100,
}

config_dict["binary_crossentropy_Adam_modified_100"] = {
        "layers":                   [100],
        "loss_function":            "binary_crossentropy",
        "Dropout":                  0.,
        "L1_Norm":                  0.,
        "L2_Norm":                  1e-4,
        "batch_size":               2000,
        "optimizer":                optimizers.Adam(learning_rate=1e-3),
        "activation_function":      "relu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage":  0.02,
        "earlystopping_epochs":      100,
}

config_dict["ttbb_reco_v2"] = {
        "layers":                   [100,100],
        "loss_function":            "binary_crossentropy",
        "Dropout":                  0.2,
        "L1_Norm":                  1e-3,
        "L2_Norm":                  1e-3,
        "batch_size":               128,
        "optimizer":                optimizers.Adadelta(),
        "activation_function":      "leakyrelu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage":  0.1,
        "earlystopping_epochs":      100,
}

config_dict["ttbb_reco"] = {
        "layers":                   [100,100],
        "loss_function":            "binary_crossentropy",
        "Dropout":                  0.2,
        "L1_Norm":                  1e-3,
        "L2_Norm":                  1e-3,
        "batch_size":               128,
        "optimizer":                optimizers.Adagrad(),
        "activation_function":      "leakyrelu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage":  0.1,
        "earlystopping_epochs":      100,
}

config_dict["binary_squared_Adadelta"] = {
        "layers":                   [200,100],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.3,
        "L1_Norm":                  0,
        "L2_Norm":                  0.,
        "batch_size":               4096,
        "optimizer":                optimizers.Adadelta(),
        "activation_function":      "elu",
        "output_activation":        "Tanh",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
}

config_dict["binary_squared_SGD"] = {
        "layers":                   [100,100],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.40,
        "L1_Norm":                  0,
        "L2_Norm":                  1e-5,
        "batch_size":               64,
        "optimizer":                optimizers.SGD(1e-3),
        "activation_function":      "tanh",
        "output_activation":        "Tanh",
        "earlystopping_percentage":  0.02,
        "earlystopping_epochs":      100,
}

config_dict["adversary_multi"] = {
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
        "pretrain_class_epochs":    2,
        "pretrain_adv_epochs":      5,
        "adversary_epochs":         1,
        "adversary_iterations":     1,
}

config_dict["adversary_binary"] = {
        "layers":                   [200,100],
        "loss_function":            "binary_crossentropy",
        "Dropout":                  0.30,
        "L1_Norm":                  1e-4,
        "L2_Norm":                  1e-3,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-3),
        "activation_function":      "elu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        "adversary_layers":         [100,100],
        "pretrain_class_epochs":    200,
        "pretrain_adv_epochs":      50,
        "adversary_epochs":         10,
        "adversary_iterations":     100,
}
config_dict["adversary_binary_test"] = {
        "layers":                   [200,100],
        "loss_function":            "binary_crossentropy",
        "Dropout":                  0.30,
        "L1_Norm":                  1e-4,
        "L2_Norm":                  1e-3,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-3),
        "activation_function":      "elu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        "adversary_layers":         [100,100],
        "pretrain_class_epochs":    20,
        "pretrain_adv_epochs":      5,
        "adversary_epochs":         1,
        "adversary_iterations":     1,
}

config_dict["BNN"] = {
        "layers":                   [50],
        #"loss_function":            "neg_log_likelihood",
        "Dropout":                  0,
        #"L1_Norm":                  0,
        #"L2_Norm":                  1e-5,
        "batch_size":               2000,
        "optimizer":                optimizers.Adam(learning_rate=1e-3),
        "activation_function":      "relu",
        "output_activation":        "sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
}

# ATTENTION: create a new config_dict for ANY changes applied to DenseFlipout otherwise the trained model 
# cannot be reloaded again since currently only the weights but not the model architecture is savable 
config_dict["BNN_Flipout_default"] = {
        "layers":                   [50],
        #"loss_function":            "neg_log_likelihood",
        "Dropout":                  0,
        #"L1_Norm":                  0,
        #"L2_Norm":                  1e-5,
        "batch_size":               2000,
        "optimizer":                optimizers.Adam(learning_rate=1e-3),
        "activation_function":      "relu",
        "output_activation":        "sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        "activity_regularizer":     None, 
        "trainable":                True,
        "kernel_posterior_fn":      tfp.layers.util.default_mean_field_normal_fn(),
        "kernel_prior_fn":          wrapped_partial(default_multivariate_normal_fn, scale=3),
        "bias_posterior_fn":        tfp.layers.util.default_mean_field_normal_fn(is_singular=True), 
        "bias_prior_fn":            None, 
        "seed":                     None, 
}

config_dict["BNN_Flipout_modified_V1"] = {
        "layers":                   [50],
        #"loss_function":            "neg_log_likelihood",
        "Dropout":                  0,
        #"L1_Norm":                  0,
        #"L2_Norm":                  1e-5,
        "batch_size":               2000,
        "optimizer":                optimizers.Adam(learning_rate=1e-3),
        "activation_function":      "relu",
        "output_activation":        "sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        "activity_regularizer":     None, 
        "trainable":                True,
        "kernel_posterior_fn":      tfp.layers.util.default_mean_field_normal_fn(loc_initializer=tf1.initializers.random_normal(mean=0.,stddev=0.), untransformed_scale_initializer=tf1.initializers.random_normal(mean=np.log(np.exp(1)-1), stddev=0.)), #mean is np.ln(np.exp(1)-1) so that after softplus transformation scale around 1.
        "kernel_prior_fn":          default_mean_field_normal_fn(loc_initializer=tf1.initializers.random_normal(mean=0.,stddev=0.), non_trainable_transformed_std=3.0),
        "bias_posterior_fn":        tfp.layers.util.default_mean_field_normal_fn(loc_initializer=tf1.initializers.random_normal(mean=0.,stddev=0.), untransformed_scale_initializer=tf1.initializers.random_normal(mean=np.log(np.exp(1)-1), stddev=0.)), 
        "bias_prior_fn":            default_mean_field_normal_fn(loc_initializer=tf1.initializers.random_normal(mean=0.,stddev=0.), non_trainable_transformed_std=3.0), 
        "seed":                     None, 
}

config_dict["BNN_Flipout_modified_V3"] = {
        "layers":                   [50],
        #"loss_function":            "neg_log_likelihood",
        "Dropout":                  0,
        #"L1_Norm":                  0,
        #"L2_Norm":                  1e-5,
        "batch_size":               2000,
        "optimizer":                optimizers.Adam(learning_rate=1e-3),
        "activation_function":      "relu",
        "output_activation":        "sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        "activity_regularizer":     None, 
        "trainable":                True,
        "kernel_posterior_fn":      tfp.layers.util.default_mean_field_normal_fn(), 
        "kernel_prior_fn":          wrapped_partial(default_multivariate_normal_fn, scale=3),
        "bias_posterior_fn":        tfp.layers.util.default_mean_field_normal_fn(), 
        "bias_prior_fn":            wrapped_partial(default_multivariate_normal_fn, scale=3),
        "seed":                     None, 
}