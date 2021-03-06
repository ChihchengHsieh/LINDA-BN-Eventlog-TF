from Models.ScenarioCfModel import ScenarioCfModel
from Utils.Preprocessing import dataset_split
from typing import List, Tuple
from matplotlib import pyplot as plt

import pandas as pd
from Parameters.EnviromentParameters import EnviromentParameters
from Parameters.Enums import SelectableModels
from Parameters import TrainingParameters
from Utils.PrintUtils import print_big, print_peforming_task
from Data import ScenarioCfDataset
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sn
import tensorflow as tf
import numpy as np
import pathlib
from matplotlib.lines import Line2D
import os
from Utils.SaveUtils import save_parameters_json


class ScenarioCfController(object):

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
        self.test_accuracy: float = None
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

        ############ Normalise input data ############
        if self.model.should_load_mean_and_vairance():
            self.model.get_mean_and_variance(self.dataset.df.iloc[list(
                self.train_dataset.unbatch().as_numpy_iterator())])

        self.__initialise_loss_fn()

    def __initialise_dataset(self):
        ############ Determine dataset ############
        self.feature_names = None
        self.dataset = ScenarioCfDataset(
            file_path=EnviromentParameters.BPI2012ValidTraceDataset.file_path,
            preprocessed_folder_path=EnviromentParameters.BPI2012ValidTraceDataset.preprocessed_foldr_path,
            preprocessed_df_type=EnviromentParameters.BPI2012ValidTraceDataset.preprocessed_df_type,
        )

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
        # Setting up model
        self.model = ScenarioCfModel(
            activity_vocab=self.dataset.activity_vocab,
            resource_vocab=self.dataset.resource_vocab,
            dense_dim=self.parameters.baselineLSTMWithResourceparameters.dense_dim,
            activity_embedding_dim=self.parameters.baselineLSTMWithResourceparameters.activity_embedding_dim,
            resource_embedding_dim=self.parameters.baselineLSTMWithResourceparameters.resource_embedding_dim,
            lstm_hidden=self.parameters.baselineLSTMWithResourceparameters.lstm_hidden,
            dropout=self.parameters.baselineLSTMWithResourceparameters.dropout,
        )

    def __intialise_optimizer(
        self,
    ):
        self.optim = tf.keras.optimizers.Adam(
                learning_rate=self.parameters.optimizerParameters.learning_rate)
        # # Setting up optimizer
        # if self.parameters.optimizer == SelectableOptimizer.Adam:
        #     self.optim = tf.keras.optimizers.Adam(
        #         learning_rate=self.parameters.optimizerParameters.learning_rate)
        # elif self.parameters.optimizer == SelectableOptimizer.SGD:
        #     self.optim = tf.keras.optimizers.SGD(
        #         learning_rate=self.parameters.optimizerParameters.learning_rate)
        # else:
        #     raise NotSupportedError("Optimizer you selected is not supported")

    def __initialise_loss_fn(self):
        def sparse_ce(y_pred, y_true, pad_value = -1):
            # print_big(y_pred, "y_pred")
            # print_big(y_true, "y_true")

            self.y_pred = y_pred.numpy()
            self.y_true = y_true

            # For flatten
            # loss_all = tf.keras.losses.binary_crossentropy(
            #     y_true=y_true, y_pred=y_pred)

            flatten_y_true = tf.reshape(y_true, (-1))
            select_idx = tf.where(flatten_y_true != pad_value)
            y_true_without_pad = tf.gather(flatten_y_true, select_idx)
            y_pred_wihtout_pad = tf.gather(tf.reshape(y_pred, (-1)), select_idx)

            ## Masking loss.
            # loss_all = tf.keras.losses.binary_crossentropy(
            #     y_true=y_true[:, :, tf.newaxis], y_pred=y_pred)
            # loss_all = loss_all * tf.cast(y_true != pad_value, dtype=tf.float32)
            y_pred_wihtout_pad = tf.nn.sigmoid(y_pred_wihtout_pad)
            loss_all = tf.keras.losses.binary_crossentropy(
                y_true=y_true_without_pad, y_pred=y_pred_wihtout_pad, from_logits=False)
            
            ## Replace it with normal binary without reductino
            # loss_all = tf.keras.metrics.hinge(y_true, y_pred)
            loss = tf.reduce_mean(loss_all)
            return loss
        self.loss = sparse_ce

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
            for _, train_idxs in enumerate(
                self.train_dataset
            ):
                # caseids, padded_data_traces, padded_target_traces
                train_data = self.dataset.collate_fn(train_idxs)
                _, train_loss, train_accuracy = self.train_step(
                    train_data
                )
                if self.__steps == 0:
                    self.acc = train_accuracy

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

        self.perform_eval_on_testset()

        # Peform testing in the end

    def train_step(
        self, data
    ) -> Tuple[float, float]:
        """
        Return is a tuple of (loss, accuracy)
        """
        self.data = data
        with tf.GradientTape() as tape:
            out, loss, accuracy = self.step(data, training=True)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.grads = grads
        self.optim.apply_gradients(grads_and_vars=zip(
            grads, self.model.trainable_variables))
        return out, loss.numpy(), accuracy

    def step(self, data, training=None):
        # Make sure the last item in data is target
        target = data[-1]
        # self.data = data
        out = self.model.data_call(data, training=training)
        # self.out = out
        loss = self.model.get_loss(self.loss, out, target)
        accuracy = self.model.get_accuracy(out, target)
        return out, loss, accuracy

    def eval_step(
        self, data
    ):
        """
        Return is a tuple of (loss, accuracy)
        """
        out, loss, accuracy = self.step(data, training=False)
        return out, loss.numpy(), accuracy

    def perform_eval_on_testset(self):
        print_peforming_task("Testing")
        _, self.test_accuracy = self.perform_eval_on_dataset(
            self.test_dataset, show_report=False)

    def perform_eval_on_dataset(self, dataset, show_report: bool = False, pad_value = -1) -> Tuple[float, float]:

        all_loss = []
        all_accuracy = []
        all_batch_size = []
        all_predictions = []
        all_targets = []

        for idxs in dataset:
            data = self.dataset.collate_fn(idxs)
            y_true = data[-1]
            out, loss, accuracy = self.eval_step(data)
            y_pred_list, y_true_list = self.model.get_flatten_prediction_and_targets(out, y_true)
            all_predictions.extend(y_pred_list)
            all_targets.extend(y_true_list)
            all_loss.append(loss)
            all_accuracy.append(accuracy)
            all_batch_size.append(len(data[-1]))

        self.all_accuracy = all_accuracy

        accuracy = accuracy_score(all_targets, all_predictions)
        mean_loss = sum(tf.constant(all_loss) * tf.constant(all_batch_size,
                        dtype=tf.float32)) / len(list(dataset.unbatch().as_numpy_iterator()))

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

    def plot_confusion_matrix(self, targets: List[int], predictions:  List[int]):
        # Plot the cufusion matrix
        cm = confusion_matrix(targets, predictions, labels=list(
            range(len(self.model.get_labels()))))
        df_cm = pd.DataFrame(cm, index=list(
            self.model.get_labels()), columns=list(self.model.get_labels()))

        if (self.parameters.plot_cm):
            plt.figure(figsize=(40, 40), dpi=100)
            sn.heatmap(df_cm / np.sum(cm), annot=True, fmt='.2%')
        else:
            print("="*20)
            print(df_cm)
            print("="*20)

    #######################################
    #   Utils
    #######################################
    def show_model_info(self):
        if self.parameters.model == SelectableModels.BaselineLSTMWithResource:
            self.model(tf.ones((1, len(self.feature_names)
                                if not self.feature_names is None else 1)), tf.ones((1, len(self.feature_names)
                                                                                     if not self.feature_names is None else 1)), [0.0], training=False)
        else:
            self.model(tf.ones((1, len(self.feature_names)
                                if not self.feature_names is None else 1)), training=False)

        self.model.summary()

        if (self.__steps != 0):
            print_big(
                "Loaded model has been trained for [%d] steps, [%d] epochs"
                % (self.__steps, self.__epoch)
            )

    def plot_grad_flow(self):
        """
        Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
        """
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in self.model.named_parameters():
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads,
                alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads,
                alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        # zoom in on the lower gradient regions
        plt.ylim(bottom=-0.001, top=0.02)
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend(
            [
                Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4),
            ],
            ["max-gradient", "mean-gradient", "zero-gradient"],
        )
        plt.show()

    #####################################
    #   Save
    #####################################
    def save_training_result(self, train_file: str) -> None:
        """
        Save to SavedModels folder:
        """
        if not (self.test_accuracy is None):
            saving_folder_path = os.path.join(
                pathlib.Path(train_file).parent,
                "SavedModels/%.4f_%s_%s_%s" % (self.test_accuracy, "ValidPath", self.parameters.model.value,
                                               str(datetime.now())),
            )
        else:
            saving_folder_path = os.path.join(
                pathlib.Path(train_file).parent,
                "SavedModels/%s_%s_%s" % (self.parameters.dataset.value, "ValidPath",
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

        if (self.model.has_mean_and_variance()):
            print_big("Save mean and vairance")
            save_dict["mean_"] = tf.Variable(self.model.mean_)
            save_dict["var_"] = tf.Variable(self.model.var_)

        checkpoint = tf.train.Checkpoint(**save_dict)

        # if (self.model.has_mean_and_variance()):
        #     checkpoint.norm_params = {
        #         "mean_": self.model.mean_,
        #         "var_": self.model.var_
        #     }

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

        if (self.model.should_load_mean_and_vairance()):
            print_big("Load mean and variance")
            mean_ = tf.Variable(
                tf.ones((len(self.feature_names))), dtype=tf.float32)
            var_ = tf.Variable(
                tf.ones((len(self.feature_names))), dtype=tf.float32)
            load_dict["mean_"] = mean_
            load_dict["var_"] = var_

        checkpoint = tf.train.Checkpoint(
            **load_dict
        )

        # if (self.model.should_load_mean_and_vairance()):
        #     checkpoint.norm_params = {
        #         "mean_": self.model.mean_,
        #         "var_": self.model.var_
        #     }

        checkpoint.restore(tf.train.latest_checkpoint(folder_path))

        self.__epoch = epoch.numpy()
        self.__steps = steps.numpy()
        if (self.model.should_load_mean_and_vairance()):
            self.model.mean_ = tf.constant(mean_)
            self.model.var_ = tf.constant(var_)

        del checkpoint

        print_big("Model loaded successfully from: %s " % (folder_path))
