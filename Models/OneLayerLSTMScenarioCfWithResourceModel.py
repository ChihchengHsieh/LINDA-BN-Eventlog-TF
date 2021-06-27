from Utils.SaveUtils import save_parameters_json
import os
import pathlib
from Models.ControllerModel import ControllerModel
from Parameters.ModelParameters import OneLayerLSTMScenarioCfWithResourceModelParameters
from Utils.PrintUtils import print_big
import tensorflow as tf
from Utils import VocabDict
from typing import List
from datetime import datetime


class OneLayerLSTMScenarioCfWithResourceModel(ControllerModel):
    name = "OneLayerLSTMScenarioCfWithResourceModel"

    def __init__(self,
                 activity_vocab: VocabDict,
                 resource_vocab: VocabDict,
                 parameters: OneLayerLSTMScenarioCfWithResourceModelParameters,
                 pad_value_in_target=-1
                 ):
        super().__init__()
        self.pad_value = pad_value_in_target
        self.activity_vocab = activity_vocab
        self.resource_vocab = resource_vocab
        self.parameters = parameters
        self.activity_embedding = tf.keras.layers.Embedding(
            input_dim=len(self.activity_vocab),
            output_dim=self.parameters.activity_embedding_dim,
            mask_zero=True,
        )

        self.resource_embedding = tf.keras.layers.Embedding(
            input_dim=len(self.resource_vocab),
            output_dim=self.parameters.resource_embedding_dim,
            mask_zero=True
        )

        self.activity_lstm = tf.keras.layers.LSTM(
            self.parameters.lstm_hidden,
            return_sequences=True,
            return_state=True,
        )

        self.resource_lstm = tf.keras.layers.LSTM(
            self.parameters.lstm_hidden,
            return_sequences=True,
            return_state=True,
        )

        self.out_net = tf.keras.models.Sequential(
            [
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(self.parameters.dropout),
                tf.keras.layers.Dense(self.parameters.dense_dim),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(self.parameters.dropout),
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, inputs, input_resources, amount, init_state=None, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        if len(inputs.shape) == 3:
            activity_emb_out = tf.matmul(
                inputs, tf.squeeze(tf.stack(self.activity_embedding.get_weights(), axis=0)))
            resource_emb_out = tf.matmul(
                input_resources, tf.squeeze(tf.stack(self.resource_embedding.get_weights(), axis=0)))
            mask = None
        else:
            activity_emb_out = self.activity_embedding(
                inputs, training=training)
            resource_emb_out = self.resource_embedding(
                input_resources, training=training)
            mask = self.activity_embedding.compute_mask(inputs)

        max_length = activity_emb_out.shape[1]

        activity_lstm_out, a_h_out, a_c_out = self.activity_lstm(
            activity_emb_out, training=training, mask=mask, initial_state=init_state[0] if init_state else None)

        resources_lstm_out, r_h_out, r_c_out = self.resource_lstm(
            resource_emb_out, training=training, mask=mask, initial_state=init_state[2] if init_state else None)

        amount_to_concate = tf.repeat(tf.expand_dims(tf.expand_dims(
            amount, axis=1), axis=2), max_length, axis=1)

        concat_out = tf.concat(
            [activity_lstm_out, resources_lstm_out, amount_to_concate],
            axis=-1
        )

        out = self.out_net(concat_out, training=training)

        return out, [[(a_h_out, a_c_out), (r_h_out, r_c_out)]]

    def data_call(self, data, training=None):
        _, padded_data_traces, _, padded_data_resources, amount, _, _ = data
        out, _ = self.call(padded_data_traces,
                           padded_data_resources,
                           amount,
                           training=training
                           )
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

        return accuracy

    def get_loss(self, loss_fn: callable, y_pred, data, pad_value=-1):
        '''
        [loss_fn]: loss function to compute the loss.\n
        [out]: output of the model\n
        [target]: target of input data.\n

        ---------------------
        return: loss value
        '''
        y_true = data[-1]
        flatten_y_true = tf.reshape(y_true, (-1))
        select_idx = tf.where(flatten_y_true != pad_value)
        y_true_without_pad = tf.gather(flatten_y_true, select_idx)
        y_pred_wihtout_pad = tf.gather(
            tf.reshape(y_pred, (-1)), select_idx)

        y_pred_wihtout_pad = tf.nn.sigmoid(y_pred_wihtout_pad)
        loss_all = loss_fn(
            y_true_without_pad,
            y_pred_wihtout_pad,
            from_logits=False
        )

        loss = tf.reduce_mean(loss_all)

        return loss

    def get_labels(self):
        return self.activity_vocab.vocabs.keys()

    def get_mean_and_variance(self, df):
        pass

    def should_load_mean_and_vairance(self):
        return False

    def has_mean_and_variance(self,):
        return False

    def get_prediction_list_from_out(self, out, data):
        target = data[-1]
        mask = self.generate_mask(target)
        predicted = tf.math.argmax(out, axis=-1)  # (B, S)
        selected_predictions = tf.boolean_mask(
            predicted, mask)

        return selected_predictions.numpy().tolist()

    def get_target_list_from_target(self, data):
        target = data[-1]
        mask = self.generate_mask(target)
        selected_targets = tf.boolean_mask(
            target, mask
        )
        return selected_targets.numpy().tolist()

    def generate_mask(self, target):
        return target != self.pad_value_in_target

    def get_flatten_prediction_and_targets(self, y_pred, y_true, pad_value=-1):
        flatten_y_true = tf.reshape(y_true, (-1))
        select_idx = tf.where(flatten_y_true != pad_value)
        y_true_without_pad = tf.gather(flatten_y_true, select_idx)
        y_pred_wihtout_pad = tf.gather(tf.reshape(y_pred, (-1)), select_idx)
        y_pred_wihtout_pad = tf.cast(y_pred_wihtout_pad > .5, dtype=tf.float32)

        return y_pred_wihtout_pad.numpy().tolist(), y_true_without_pad.numpy().tolist()

    def show_model_info(self):
        self.call(tf.ones((1, 1)), tf.ones((1, 1)), [0.0], training=False)
        self.summary()

    def get_folder_path(self, current_file, test_accuracy, additional=""):
        saving_folder_path = os.path.join(
            pathlib.Path(current_file).parent,
            "SavedModels/%.4f_%s_%s_%s" % (self.test_accuracy,
                                           self.name,
                                           additional,
                                           str(datetime.now())),
        )
        return saving_folder_path

    def save_model(self, folder_path: str):

        os.makedirs(folder_path, exist_ok=True)

        # Save parameters
        parameters_saving_path = os.path.join(
            folder_path, "parameters.json"
        )
        save_parameters_json(parameters_saving_path, self.parameters)

        # Save model
        model_saving_path = os.path.join(
            folder_path, "model.ckpt"
        )
        save_dict = {
            "model": self.model,
        }

        checkpoint = tf.train.Checkpoint(**save_dict)
        checkpoint.save(model_saving_path)
        print_big("Model saved successfully to: %s " % (folder_path))

    def load_model(self, folder_path: str):

        load_dict = {
            "model": self.model,
        }

        checkpoint = tf.train.Checkpoint(
            **load_dict
        )

        checkpoint.restore(tf.train.latest_checkpoint(folder_path))

        del checkpoint

        print_big("Model loaded successfully from: %s " % (folder_path))
