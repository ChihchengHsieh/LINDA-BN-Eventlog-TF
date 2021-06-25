from Utils.PrintUtils import print_big
import tensorflow as tf
from Utils import Constants, VocabDict
import numpy as np
from typing import List
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class BaselineLSTMWithResourceValidTraceCf(tf.keras.Model):
    def __init__(self, activity_vocab: VocabDict, resource_vocab: VocabDict, activity_embedding_dim: int, resource_embedding_dim: int, lstm_hidden: int, dense_dim: int, dropout: float, one_hot=False):
        super().__init__()
        self.activity_vocab = activity_vocab
        self.resource_vocab = resource_vocab
        self.activity_embedding = tf.keras.layers.Embedding(
            input_dim=len(self.activity_vocab),
            output_dim=activity_embedding_dim,
            mask_zero=True,
        )

        self.resource_embedding = tf.keras.layers.Embedding(
            input_dim=len(self.resource_vocab),
            output_dim=resource_embedding_dim,
            mask_zero=True
        )

        self.activity_lstm = tf.keras.layers.LSTM(
            lstm_hidden,
            return_sequences=True,
            return_state=True,
        )

        self.activity_lstm_sec = tf.keras.layers.LSTM(
            lstm_hidden,
            return_sequences=True,
            return_state=True,
        )

        self.resource_lstm = tf.keras.layers.LSTM(
            lstm_hidden,
            return_sequences=True,
            return_state=True,
        )

        self.resource_lstm_sec = tf.keras.layers.LSTM(
            lstm_hidden,
            return_sequences=True,
            return_state=True,
        )

        self.out_net = tf.keras.models.Sequential(
            [
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(dense_dim),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, inputs, input_resources, amount, init_state=None, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        # print(f"Is training: {training}")
        # self.data_input = [inputs, input_resources, amount]

        if len(inputs.shape) == 3:
            print_big("Multiply!")
            activity_emb_out = tf.matmul(
                inputs, tf.squeeze(tf.stack(self.activity_embedding.get_weights(), axis=0)))
            resource_emb_out = tf.matmul(
                input_resources, tf.squeeze(tf.stack(self.resource_embedding.get_weights(), axis=0)))
            mask = None
        else:
            # print_big("Normal Emb")
            activity_emb_out = self.activity_embedding(
                inputs, training=training)
            resource_emb_out = self.resource_embedding(
                input_resources, training=training)
            mask = self.activity_embedding.compute_mask(inputs)

        max_length = activity_emb_out.shape[1]

        activity_lstm_out, a_h_out, a_c_out = self.activity_lstm(
            activity_emb_out, training=training, mask=mask, initial_state=init_state[0] if init_state else None)
        activity_lstm_out_sec, a_h_out_sec, a_c_out_sec = self.activity_lstm_sec(
            activity_lstm_out, training=training, mask=mask, initial_state=init_state[1] if init_state else None)

        resources_lstm_out, r_h_out, r_c_out = self.resource_lstm(
            resource_emb_out, training=training, mask=mask, initial_state=init_state[2] if init_state else None)
        resources_lstm_out_sec, r_h_out_sec, r_c_out_sec = self.resource_lstm_sec(
            resources_lstm_out, training=training, mask=mask, initial_state=init_state[3] if init_state else None)

        # return resources_lstm_out_sec

        amount_to_concate = tf.repeat(tf.expand_dims(tf.expand_dims(
            amount, axis=1), axis=2), max_length, axis=1)

        concat_out = tf.concat(
            [activity_lstm_out_sec, resources_lstm_out_sec, amount_to_concate], axis=-1)

        # return concat_out
        out = self.out_net(concat_out, training=training)
        # out = tf.nn.sigmoid(out)
        
        return out, [[(a_h_out, a_c_out), (r_h_out, r_c_out)], (a_h_out_sec, a_c_out_sec), (r_h_out_sec, r_c_out_sec)]

    def data_call(self, data, training=None):
        _, padded_data_traces, _, padded_data_resources, amount, _, _ = data
        out, _ = self.call(padded_data_traces,
                           padded_data_resources, amount, training=training)
        return out

    def get_accuracy(self, y_pred, y_true, pad_value=-1):
        '''
        Use argmax to get the final output, and get accuracy from it.
        [out]: output of model.
        [target]: target of input data.
        --------------
        return: accuracy value
        '''

        flatten_y_true = tf.reshape(y_true, (-1))
        select_idx = tf.where(flatten_y_true != pad_value)
        y_true_without_pad = tf.cast(
            tf.gather(flatten_y_true, select_idx), dtype=tf.float32)
        y_pred_wihtout_pad = tf.gather(tf.reshape(y_pred, (-1)), select_idx)
        y_pred_wihtout_pad = tf.cast(y_pred_wihtout_pad > .5, dtype=tf.float32)

        accuracy = tf.reduce_mean(
            tf.cast(y_pred_wihtout_pad == y_true_without_pad, dtype=tf.float32))

        # pred_value = tf.constant(y_pred > 0.5, dtype=tf.float32)
        # accuracy = tf.math.reduce_mean(tf.cast(tf.boolean_mask(
        #     y_true == pred_value, y_true != pad_value), dtype=tf.float32)).numpy()

        return accuracy

    def get_loss(self, loss_fn: callable, y_pred, y_true, pad_value=-1):
        '''
        [loss_fn]: loss function to compute the loss.\n
        [out]: output of the model\n
        [target]: target of input data.\n

        ---------------------
        return: loss value
        '''
        # flatten_y_true = tf.reshape(y_true, (-1))
        # select_idx = tf.where(flatten_y_true != pad_value)
        # y_true_without_pad = tf.gather(flatten_y_true, select_idx)
        # y_pred_wihtout_pad = tf.gather(tf.reshape(y_pred, (-1)), select_idx)

        return loss_fn(y_pred=y_pred, y_true=y_true)

    def get_labels(self):
        return self.activity_vocab.vocabs.keys()

    def get_mean_and_variance(self, df):
        pass

    def should_load_mean_and_vairance(self):
        return False

    def has_mean_and_variance(self,):
        return False

    def get_prediction_list_from_out(self, out, mask=None,):
        predicted = tf.math.argmax(out, axis=-1)  # (B, S)
        selected_predictions = tf.boolean_mask(
            predicted, mask)

        return selected_predictions.numpy().tolist()

    def get_target_list_from_target(self, target, mask=None):
        selected_targets = tf.boolean_mask(
            target, mask
        )
        return selected_targets.numpy().tolist()

    def generate_mask(self, target, pad_value=-1):
        return target != pad_value


    def get_flatten_prediction_and_targets(self, y_pred, y_true, pad_value = -1):
        flatten_y_true = tf.reshape(y_true, (-1))
        select_idx = tf.where(flatten_y_true != pad_value)
        y_true_without_pad = tf.gather(flatten_y_true, select_idx)
        y_pred_wihtout_pad =  tf.gather(tf.reshape(y_pred, (-1)), select_idx)
        y_pred_wihtout_pad = tf.cast(y_pred_wihtout_pad > .5, dtype=tf.float32)

        return y_pred_wihtout_pad.numpy().tolist(), y_true_without_pad.numpy().tolist()



        

