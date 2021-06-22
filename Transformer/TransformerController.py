from Transformer.utils import accuracy_function, loss_function
from Transformer.masking import create_predicting_next_mask
from Transformer.scheduler import CustomSchedule
from Transformer.PredictingNextTransformerEncoder import PredictingNextTransformerEncoder
from Utils.Preprocessing import dataset_split
from Models import BaseNN, BaselineLSTMWithResource
from Data.MedicalDataset import MedicalDataset
from typing import List, Tuple
from matplotlib import pyplot as plt

import pandas as pd
from Models.BaselineLSTM import BaselineLSTM
from CustomExceptions.Exceptions import NotSupportedError
from Parameters.EnviromentParameters import EnviromentParameters
from Parameters.Enums import SelectableDatasets, SelectableLoss, SelectableModels, SelectableOptimizer
from Parameters import TrainingParameters
from Utils.PrintUtils import print_big, print_peforming_task
from Data import XESDataset, XESDatasetWithResource
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sn
import tensorflow as tf
import numpy as np
import pathlib
from matplotlib.lines import Line2D
import os
from Utils.SaveUtils import save_parameters_json


class TransformerController(object):
    #########################################
    #   Initialisation
    #########################################

    def __init__(self, parameters: TrainingParameters):

        self.parameters: TrainingParameters = parameters

        temp = tf.constant([0])
        print_big("Running on %s " % (temp.device))
        del temp

        ############ Initialise counters ############
        self.__epoch: int = 0
        self.__steps: int = 0
        self.stop_epoch = self.parameters.stop_epoch

        self.__initialise_dataset()
        self.__initialise_model()
        self.__intialise_optimizer()

        ############ Load saved parameters ############
        if not self.parameters.load_model_folder_path is None:
            ############ Load trained if specified ############
            self.load_trained_model(
                self.parameters.load_model_folder_path,
            )

    def __initialise_dataset(self):
        ############ Determine dataset ############
        if self.parameters.dataset == SelectableDatasets.BPI2012:
            self.feature_names = None
            self.dataset = XESDataset(
                file_path=EnviromentParameters.BPI2020Dataset.file_path,
                preprocessed_folder_path=EnviromentParameters.BPI2020Dataset.preprocessed_foldr_path,
                preprocessed_df_type=EnviromentParameters.BPI2020Dataset.preprocessed_df_type,
                include_types=self.parameters.bpi2012.BPI2012_include_types,
            )
        elif self.parameters.dataset == SelectableDatasets.BPI2012WithResource:
            self.feature_names = None
            self.dataset = XESDatasetWithResource(
                file_path=EnviromentParameters.BPI2020DatasetWithResource.file_path,
                preprocessed_folder_path=EnviromentParameters.BPI2020DatasetWithResource.preprocessed_foldr_path,
                preprocessed_df_type=EnviromentParameters.BPI2020DatasetWithResource.preprocessed_df_type,
                include_types=self.parameters.bpi2012.BPI2012_include_types,
            )
        elif self.parameters.dataset == SelectableDatasets.Helpdesk:
            self.feature_names = None
            self.dataset = XESDataset(
                file_path=EnviromentParameters.HelpDeskDataset.file_path,
                preprocessed_folder_path=EnviromentParameters.HelpDeskDataset.preprocessed_foldr_path,
                preprocessed_df_type=EnviromentParameters.HelpDeskDataset.preprocessed_df_type,
            )
        else:
            raise NotSupportedError("Dataset you selected is not supported")

        self.train_dataset, self.test_dataset, self.validation_dataset = dataset_split(list(range(len(
            self.dataset))), self.parameters.train_test_split_portion, seed=self.parameters.dataset_split_seed,  shuffle=True)

        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            self.train_dataset).batch(self.parameters.batch_size)
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            self.test_dataset).batch(self.parameters.batch_size)
        self.validation_dataset = tf.data.Dataset.from_tensor_slices(
            self.validation_dataset).batch(self.parameters.batch_size)

    def __initialise_model(
        self,
    ):
        #### Prepare model ####
        self.model = PredictingNextTransformerEncoder(
            vocab=self.dataset.vocab,
            num_layers=self.parameters.transformerParameters.num_layers,
            d_model=self.parameters.transformerParameters.model_dim,
            num_heads=self.parameters.transformerParameters.num_heads,
            dff=self.parameters.transformerParameters.feed_forward_dim,
            vocab_size=len(self.dataset.vocab),
            pe_input=self.dataset.longest_trace_len() * 10,
        )

    def __intialise_optimizer(
        self,
    ):
        # Initialise learning_reate
        self.learning_rate = CustomSchedule(
            self.parameters.transformerParameters.model_dim)

        # Setting up optimizer
        if self.parameters.optimizer == SelectableOptimizer.Adam:
            self.optim = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        elif self.parameters.optimizer == SelectableOptimizer.SGD:
            self.optim = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate)
        else:
            raise NotSupportedError("Optimizer you selected is not supported")

    ##########################################
    #   Train & Evaluation
    ##########################################

    def train(
        self,
    ):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_folder_name = 'logs/gradient_tape/' + current_time
        train_log_dir = tb_folder_name + '/train'
        test_log_dir = tb_folder_name + '/test'
        print_big("Training records in %s" % (tb_folder_name))
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        tf.keras.callbacks.TensorBoard(log_dir=train_log_dir)
        print_big("Total epochs: %d" % (self.stop_epoch))
        print_big("Total steps: %d" %
                  (self.stop_epoch * len(self.train_dataset)))
        while self.__epoch < self.stop_epoch:
            print_big("Start epoch %d" % (self.__epoch))

            for train_idxs in self.train_dataset:
                # caseids, padded_data_traces, padded_target_traces
                _, padded_data_traces, _, padded_data_resources, batch_amount, padded_target_traces = self.dataset.collate_fn(
                    train_idxs)
                _, train_loss, train_accuracy = self.train_step(
                    padded_data_traces, padded_target_traces
                )

                self.__steps += 1

                with train_summary_writer.as_default():
                    tf.summary.scalar(
                        'accuracy', train_accuracy, step=self.__steps)
                    tf.summary.scalar('loss', train_loss, step=self.__steps)

                if self.__steps > 0 and self.__steps % self.parameters.run_validation_freq == 0:
                    (
                        validation_loss,
                        validation_accuracy,
                    ) = self.perform_eval_on_dataset(self.validation_dataset, show_report=False)

                    with test_summary_writer.as_default():
                        tf.summary.scalar(
                            'accuracy', validation_accuracy, step=self.__steps)
                        tf.summary.scalar(
                            'loss', validation_loss, step=self.__steps)

            self.__epoch += 1

        test_acc = self.perform_eval_on_testset()

        return test_acc

        # Peform testing in the end

    def train_step(self, inp, tar):
        combine_mask = create_predicting_next_mask(inp)

        with tf.GradientTape() as tape:
            predictions, _ = self.model(inp, True, combine_mask)
            loss = loss_function(tar, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(
            zip(gradients, self.model.trainable_variables))

        return predictions, loss, accuracy_function(tar, predictions)

    def eval_step(
        self, inp, tar=None
    ):
        """
        Return is a tuple of (loss, accuracy)
        """
        combine_mask = create_predicting_next_mask(inp)
        predictions, _ = self.model(inp, False, combine_mask)

        if type(tar) == type(None):
            return predictions
        else:
            loss = loss_function(tar, predictions)
            acc = accuracy_function(tar, predictions)
            return predictions, loss, acc

    def perform_eval_on_testset(self):
        print_peforming_task("Testing")
        _, acc = self.perform_eval_on_dataset(
            self.test_dataset, show_report=False)
        return acc

    def predict_next(self, encoder_input, max_length=40, eos_id=1):
        # as the target is english, the first word to the transformer should be the
        # english start token.

        attentions_in_time_series = []

        for i in range(max_length):
            combined_mask = create_predicting_next_mask(encoder_input)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.model(encoder_input,
                                                        False,
                                                        combined_mask)

            attentions_in_time_series.append(attention_weights)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.argmax(predictions, axis=-1)

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            encoder_input = tf.concat([encoder_input, predicted_id], axis=-1)

            # return the result if the predicted_id is equal to the end token
            if predicted_id == eos_id:
                break

        # output.shape (1, tokens)

        all_predicted_tokens = self.model.vocab.list_of_index_to_vocab(
            encoder_input.numpy()[0])

        return encoder_input, attentions_in_time_series, all_predicted_tokens

    def perform_eval_on_dataset(self, dataset, show_report: bool = False) -> Tuple[float, float]:

        all_loss = []
        all_accuracy = []
        all_batch_size = []
        all_predictions = []
        all_targets = []

        for idxs in dataset:
            _, padded_data_traces, _, padded_data_resources, batch_amount, padded_target_traces = self.dataset.collate_fn(
                idxs)
            mask = tf.math.logical_not(tf.math.equal(padded_target_traces, 0))
            out, loss, accuracy = self.eval_step(
                padded_data_traces, padded_target_traces)
            all_predictions.extend(
                self.model.get_prediction_list_from_out(out, mask))
            all_targets.extend(
                self.model.get_target_list_from_target(padded_target_traces, mask))
            all_loss.append(loss.numpy())
            all_accuracy.append(accuracy.numpy())
            all_batch_size.append(padded_data_traces.shape[0])

        accuracy = accuracy_score(all_targets, all_predictions)
        self.all_loss = all_loss 
        self.all_batch_size = all_batch_size
        mean_loss = sum(tf.constant(all_loss) * tf.constant(all_batch_size,
                        dtype=tf.float32)) / len(dataset)

        print_big(
            "Evaluation result | Loss [%.4f] | Accuracy [%.4f] "
            % (mean_loss.numpy(), accuracy)
        )

        if (show_report):
            print_big("Classification Report")
            report = classification_report(all_targets, all_predictions, zero_division=0, output_dict=True, labels=list(
                range(len(self.model.get_labels()))), target_names=list(self.model.get_labels()))
            print(pd.DataFrame(report))

            print_big("Confusion Matrix")
            self.plot_confusion_matrix(all_targets, all_predictions)

        return mean_loss.numpy(), accuracy

    #######################################
    #   Utils
    #######################################
    def show_model_info(self):
        _, padded_data_traces, _, padded_data_resources, batch_amount, padded_target_traces = self.dataset.collate_fn([
                                                                                                                      0])
        combine_mask = create_predicting_next_mask(padded_data_traces)
        self.model(padded_data_traces, False, combine_mask)

        self.model.summary()

        if (self.__steps != 0):
            print_big(
                "Loaded model has been trained for [%d] steps, [%d] epochs"
                % (self.__steps, self.__epoch)
            )

    #####################################
    #   Save
    #####################################
    def save_training_result(self, train_file: str, test_acc=None) -> None:
        """
        Save to SavedModels folder:
        """
        if not (test_acc is None):
            saving_folder_path = os.path.join(
                pathlib.Path(train_file).parent,
                "SavedModels/%.4f_%s_%s_%s" % (test_acc, self.parameters.dataset.value, "TransformerActivityOnly",
                                               str(datetime.now())),
            )
        else:
            saving_folder_path = os.path.join(
                pathlib.Path(train_file).parent,
                "SavedModels/%s_%s_%s" % (self.parameters.dataset.value, "TransformerActivityOnly",
                                          str(datetime.now())),
            )

        # Create folder for saving
        os.makedirs(saving_folder_path, exist_ok=True)

        # Save parameters
        parameters_saving_path = os.path.join(
            saving_folder_path, EnviromentParameters.parameters_save_file_name__
        )

        save_parameters_json(parameters_saving_path, self.parameters)

        # Save model
        model_saving_path = os.path.join(
            saving_folder_path, EnviromentParameters.model_save_file_name
        )

        save_dict = {
            "model": self.model,
            "optim": self.optim,
            "epoch": tf.Variable(self.__epoch),
            "steps": tf.Variable(self.__steps),
        }

        checkpoint = tf.train.Checkpoint(**save_dict)

        checkpoint.save(model_saving_path)

        print_big("Model saved successfully to: %s " % (saving_folder_path))

    #########################################
    #   Load
    #########################################

    def load_trained_model(self, folder_path: str):
        epoch = tf.Variable(0)
        steps = tf.Variable(0)

        load_dict = {
            "model": self.model,
            "epoch": epoch,
            "steps": steps,
        }

        checkpoint = tf.train.Checkpoint(
            **load_dict
        )

        checkpoint.restore(tf.train.latest_checkpoint(folder_path))

        self.__epoch = epoch.numpy()
        self.__steps = steps.numpy()

        del checkpoint

        print_big("Model loaded successfully from: %s " % (folder_path))

    ####################################
    # Attention Map
    ####################################

    def plot_attention_head(self, in_tokens, translated_tokens, attention):
        # The plot is of the attention when a token was generated.
        # The model didn't generate `<START>` in the output. Skip it.
        ax = plt.gca()
        ax.matshow(attention)
        ax.set_xticks(range(len(in_tokens)))
        ax.set_yticks(range(len(translated_tokens)))

        labels = in_tokens
        ax.set_xticklabels(
            labels, rotation=90)

        labels = translated_tokens
        ax.set_yticklabels(labels)

    def plot_attention_weights(self, in_tokens, translated_tokens, layer_of_attention_heads):
        fig = plt.figure(figsize=(60, 20))

        n_l, n_h = layer_of_attention_heads.shape[:2]

        for l, attention_heads in enumerate(layer_of_attention_heads):
            n_h = len(attention_heads)
            for h, head in enumerate(attention_heads):
                ax = fig.add_subplot(n_l, n_h, (l*n_h) + h+1)
                self.plot_attention_head(in_tokens, translated_tokens, head)
                ax.set_xlabel(f'Layer {l+1} Head {h+1} ')
        plt.tight_layout()
        plt.show()

    def plot_step_attention_weight(self, step_i,  all_tokens, attentions_in_time_series, input_trace_length):
        last_step_attention = tf.concat(attentions_in_time_series[step_i], axis=0)[
            :, :, -1, :][:, :, tf.newaxis, :]
        self.plot_attention_weights(
            all_tokens[:step_i+input_trace_length], [
                all_tokens[step_i+input_trace_length]
            ],
            last_step_attention
        )

    def plot_average_attention(self, in_tokens, translated_tokens, layer_of_attention_heads):
        # The plot is of the attention when a token was generated.
        # The model didn't generate `<START>` in the output. Skip it.
        attention = tf.reduce_mean(layer_of_attention_heads, axis=[0, 1])
        ax = plt.gca()
        ax.matshow(attention)
        ax.set_xticks(range(len(in_tokens)))
        ax.set_yticks(range(len(translated_tokens)))

        labels = in_tokens
        ax.set_xticklabels(
            labels, rotation=90)
        labels = translated_tokens
        ax.set_yticklabels(labels)

    def plot_stop_mean_attention_weight(self, step_i, all_tokens, attentions_in_time_series, input_trace_length):
        last_step_attention = tf.concat(attentions_in_time_series[step_i], axis=0)[
            :, :, -1, :][:, :, tf.newaxis, :]
        self.plot_average_attention(
            all_tokens[:step_i+input_trace_length], [
                all_tokens[step_i+input_trace_length]
            ],
            last_step_attention
        )
