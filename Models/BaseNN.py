
from numpy import float32
import tensorflow as tf
from typing import List
from itertools import chain
from sklearn.preprocessing import StandardScaler


class BaseNN(tf.keras.Model):
    def __init__(self, feature_names, hidden_dim: List[int] = [], dropout: float = 0.1):
        super(BaseNN, self).__init__()
        self.feature_names = feature_names
        output_dim = 1

        all_dim = [len(self.feature_names)] + hidden_dim + [output_dim]
        self.all_dim = all_dim
        all_layers = [[tf.keras.layers.Dense(all_dim[idx+1]), tf.keras.layers.BatchNormalization(), tf.keras.layers.LeakyReLU(), tf.keras.layers.Dropout(
            dropout)] if idx + 2 != len(all_dim) else [tf.keras.layers.Dense(all_dim[idx+1])] for idx in range(len(all_dim)-1)]
        # all_layers = [[nn.Linear(all_dim[idx], all_dim[idx+1])] if idx + 2 != len(all_dim) else [nn.Linear(all_dim[idx], all_dim[idx+1])] for idx in range(len(all_dim)-1)]

        self.model = tf.keras.models.Sequential(
            list(chain.from_iterable(all_layers)) +
            [tf.keras.layers.Activation(tf.nn.sigmoid)]
        )

        self.mean_ = None
        self.var_ = None

    def call(self, input: tf.Tensor, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        return self.model(input, training=training)

    def data_call(self, data, training=None):
        input, _ = data

        ######### Scale input #########
        norm_input = self.normalize_input(input)

        out = self.call(norm_input, training=training)
        return out

    def get_accuracy(self, out, target):
        '''
        Use argmax to get the final output, and get accuracy from it.
        [out]: output of model.
        [target]: target of input data.
        --------------
        return: accuracy value
        '''
        return tf.math.reduce_mean(tf.cast(tf.cast(tf.squeeze(out > 0.5), dtype=tf.float32) == target, dtype=tf.float32))

    def get_loss(self, loss_fn: callable, out, target):
        '''
        [loss_fn]: loss function to compute the loss.\n
        [out]: output of the model\n
        [target]: target of input data.\n

        ---------------------
        return: loss value
        '''
        return loss_fn(tf.squeeze(out), tf.squeeze(target))

    def get_prediction_list_from_out(self, out, mask=None):
        return (out > 0.5).numpy().tolist()

    def get_target_list_from_target(self, target, mask=None):
        return target.numpy().tolist()

    def generate_mask(self, target):
        return None

    def get_labels(self):
        return ["True", "False"]

    def get_mean_and_variance(self, df):
        scaler = StandardScaler()
        scaler.fit(df[self.feature_names])
        self.mean_ = tf.constant(scaler.mean_, dtype=float32)
        self.var_ = tf.constant(scaler.var_, dtype=float32)

    def should_load_mean_and_vairance(self):
        return not self.has_mean_and_variance()

    def has_mean_and_variance(self,):
        return (not self.mean_ is None) and (not self.var_ is None)

    def normalize_input(self, input):
        return (input - self.mean_) / tf.math.sqrt(self.var_)

    def reverse_normalize_input(self, input):
        return (input * tf.math.sqrt(self.var_)) + self.mean_

    def has_embedding_layer(self,):
        return False
