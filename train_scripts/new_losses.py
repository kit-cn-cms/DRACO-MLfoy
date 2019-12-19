import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras import losses

# custom loss definition
def neg_log_likelihood(y_true, y_pred):
    sigma = 1.
    dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
    #kl = sum(model.losses)
    return -dist.log_prob(y_true) #+ kl

tf.losses.add_loss(neg_log_likelihood, loss_collection=tf.keras.losses)