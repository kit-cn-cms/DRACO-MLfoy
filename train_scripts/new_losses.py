import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras import losses

def neg_log_likelihood(y_true, y_pred):
    labels_distribution = tfp.distributions.Categorical(logits=y_pred)
    log_likelihood = labels_distribution.log_prob(tf.argmax(input=y_true, axis=1))
    loss = -tf.reduce_mean(input_tensor=log_likelihood)
    return loss

tf.losses.add_loss(neg_log_likelihood, loss_collection=tf.keras.losses)