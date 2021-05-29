import tensorflow as tf
from Utils import Constants, VocabDict
import numpy as np
from typing import List

class BaselineLSTM(tf.keras.Model):
    def __init__(self, vocab: VocabDict,  embedding_dim: int, lstm_hidden: int, dropout: float):
        super().__init__()
        self.vocab = vocab
        self.emb = tf.keras.layers.Embedding(
            input_dim=len(vocab),
            output_dim=embedding_dim,
            mask_zero=True
        )
        self.lstm = tf.keras.layers.LSTM(
            lstm_hidden,
            return_sequences=True,
            return_state=True,
            # stateful=True
        )
        self.out_net = tf.keras.models.Sequential(
            [
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(len(vocab)),
            ]
        )

    def call(self, inputs, initial_state=None, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        out = self.emb(inputs, training=training)
        mask = self.emb.compute_mask(inputs)
        out, h_out, c_out = self.lstm(
            out, training=training, initial_state=initial_state, mask=mask)
        # out = self.lstm(out, training=training)
        out = self.out_net(out, training=training)
        out = tf.nn.softmax(out, axis=-1)

        return out, (h_out, c_out)

    def data_call(self, data, training=None):
        _, padded_data_traces, _,  _ = data
        out, _ = self.call(padded_data_traces, training=training)
        return out

    def get_accuracy(self, y_pred, y_true):
        '''
        Use argmax to get the final output, and get accuracy from it.
        [out]: output of model.
        [target]: target of input data.
        --------------
        return: accuracy value
        '''
        pred_value = tf.math.argmax(y_pred, axis=-1)
        accuracy = tf.math.reduce_mean(tf.cast(tf.boolean_mask(
            y_true == pred_value, y_true != 0), dtype=tf.float32)).numpy()
        return accuracy

    def get_loss(self, loss_fn: callable, y_pred, y_true):
        '''
        [loss_fn]: loss function to compute the loss.\n
        [out]: output of the model\n
        [target]: target of input data.\n

        ---------------------
        return: loss value
        '''
        return loss_fn(y_pred=y_pred, y_true=y_true)

    def get_labels(self):
        return self.vocab.vocab_dict.keys()

    def get_mean_and_variance(self, df):
        pass

    def should_load_mean_and_vairance(self):
        return False

    def has_mean_and_variance(self,):
        return False

    def predict_next(self, input: tf.Tensor, lengths: np.array, initial_state=None, use_argmax: bool = False):
        '''
        Predict next activity.
        [input]: input traces.
        [lengths]: length of traces.
        [previous_hidden_state]: hidden state in last time step, should be (h_, c_)
        [use_argmax]: \n
        - True -> Select the vocab with largest possibility.\n
        - False -> the next prediction will be sampled from the distribution.\n
        -------------
        return: tuple(output, (h_out, c_out)).
        '''
        # batch_size = input.shape[0]  # (B, S)
        out, hidden_out = self.call(
            input, initial_state=initial_state, training=False)  # (B, S, vocab_size)

        ############ Get next activity ############
        # Get the last output from each seq
        # len - 1 to get the index,
        # a len == 80 seq, will only have index 79 as the last output (from the 79 input)

        # Get the output of last timestamp
        final_index = lengths - 1
        out = tf.gather(out, final_index, axis=1)
        # out = out[np.arange(batch_size), final_index, :]  # (B, Vocab)

        if (use_argmax):
            ############ Get the one with largest possibility ############
            out = tf.math.argmax(out, axis=-1)  # (B)
            # TODO: Testing value, need to delete
            self.argmax_out = out
        else:
            ############ Sample from distribution ############
            out = tf.random.categorical(out, 1).squeeze(
                1)  # .squeeze()  # (B)

        return out, hidden_out

    def predict_next_n(self, input: tf.Tensor, n: int, lengths: np.array = None, use_argmax: bool = False) -> List[List[int]]:
        '''
        peform prediction n times.\n
        [input]: input traces
        [n]: number of steps to peform prediction.
        [lengths]: lengths of traces
        [use_argmax]: \n
        - True -> Select the vocab with largest possibility.\n
        - False -> the next prediction will be sampled from the distribution.\n
        -------------
        return: predicted list.
        '''
        ############ Unpadded input to get current taces ############
        predicted_list = [[i.numpy() for i in l if i != 0] for l in input]

        ############ Initialise hidden state ############
        hidden_state = None
        for i in range(n):
            ############ Predict############
            predicted, hidden_state = self.predict_next(input=input, lengths=lengths,
                                                        initial_state=hidden_state, use_argmax=use_argmax)

            ############ Add predicted to current traces ############
            predicted_list = [u + [p.numpy()[0]]
                              for u, p in zip(predicted_list, predicted)]

            ############ Prepare for next step #########################################################################
            # Assign for the next loop input, since tensor use reference, we won't use too much memory for it.
            # And, we only use last step and the hidden state for predicting next.
            ############################################################################################################
            input = tf.expand_dims(predicted, axis=-1)
            lengths = np.ones_like(lengths)

        return predicted_list

    def predict_next_till_eos(self, input: tf.Tensor, lengths: np.array, eos_idx: int, use_argmax: bool = False, max_predicted_lengths=1000) -> List[List[int]]:
        '''
        pefrom predicting till <EOS> token show up.\n
        [input]: input traces
        [lengths]: lengths of traces
        [eos_idx]: index of <EOS> token
        [use_argmax]: \n
        - True -> Select the vocab with largest possibility.\n
        - False -> the next prediction will be sampled from the distribution.\n
        [max_predicted_lengths]: used to restricted the maximum step when n_step is None.
        -------------
        return: predicted list.
        '''

        ############ List for input data ############
        input_list = [[i.numpy() for i in l if i != 0] for l in input]

        ############ List that prediction has been finished ############
        predicted_list = [None] * len(input_list)

        ############ Initialise hidden state ############
        hidden_state = None
        while len(input_list) > 0:
            ############ Predict ############
            predicted, hidden_state = self.predict_next(input=input, lengths=lengths,
                                                        previous_hidden_state=hidden_state, use_argmax=use_argmax)

            ############ Check if it's 0-d tensor ############
            if (predicted.size() == ()):
                predicted = predicted.unsqueeze(0)

            for idx,  (il, p) in enumerate(zip(input_list, predicted)):
                ############ Append predicted value ############
                p_v = p.numpy()
                input_list[idx] = il + [p_v]

                if (p_v == eos_idx or len(input_list[idx]) > max_predicted_lengths):
                    ############ Create index mapper (Mapping the input_list to predicted_list) ############
                    idx_mapper = [idx for idx, pl in enumerate(
                        predicted_list) if pl is None]

                    ############ Assign to predicted_list (Remove from input list) ############
                    idx_in_predicted_list = idx_mapper[idx]
                    predicted_list[idx_in_predicted_list] = input_list.pop(idx)

                    batch_size = len(predicted)
                    ############ Remove instance from the lengths ############
                    lengths = lengths[np.arange(batch_size) != idx]

                    ############ Remove instance from next input ############
                    predicted = predicted[np.arange(batch_size) != idx, ]

                    ############ Remove the hidden state to enable next inter ############
                    # h0 = hidden_state[0][:, np.arange(batch_size) != idx, :]
                    # c0 = hidden_state[1][:, np.arange(batch_size) != idx, :]

                    # TODO: Have to check the size with this one
                    h0 = tf.boolean_mask(
                        hidden_state[0],  np.arange(batch_size) != idx, axis=1)
                    c0 = tf.boolean_mask(
                        hidden_state[1],  np.arange(batch_size) != idx, axis=1)
                    hidden_state = (h0, c0)

                    if (len(predicted) == 0 and len(input_list) == 0):
                        break

            ############################################################
            # Assign for the next loop input, since tensor use reference, we won't use too much memory for it.
            ############################################################
            input = predicted.unsqueeze(-1)
            lengths = np.ones_like(lengths)

        return predicted_list

    def predict(
        self,
        input: tf.Tensor,
        lengths: np.array = None,
        n_steps: int = None,
        use_argmax=False,
        max_predicted_lengths=50,
    ) -> List[List[int]]:
        '''
        [input]: tensor to predict\n
        [lengths]: lengths of input\n
        [n_step]: how many steps will be predicted. If n_step == None, model will
        repeat to predict till <EOS> token show.\n
        [use_argmax]: \n
        - True -> Select the vocab with largest possibility.\n
        - False -> the next prediction will be sampled from the distribution.\n
        [max_predicted_lengths]: used to restricted the maximum step when n_step is None.

        ----------------------------
        return: predicted list.

        '''
        if not n_steps is None:
            ######### Predict for next n activities #########
            predicted_list = self.predict_next_n(
                input=input, lengths=lengths, n=n_steps, use_argmax=use_argmax
            )

        else:
            ######### Predict till <EOS> token #########
            '''
            This method has the risk of causing infinite loop,
            `max_predicted_lengths` is used for restricting this behaviour.
            '''
            predicted_list = self.predict_next_till_eos(
                input=input,
                lengths=lengths,
                eos_idx=self.vocab.vocab_to_index(Constants.EOS_VOCAB),
                use_argmax=use_argmax,
                max_predicted_lengths=max_predicted_lengths,
            )

        return predicted_list

    def predicting_from_list_of_idx_trace(
        self, data: List[List[int]], n_steps: int = None, use_argmax=False
    ):
        '''
        [data]: 2D list of token indexs.
        [n_step]: how many steps will be predicted. If n_step == None, model will
        repeat to predict till <EOS> token show.\n
        [use_argmax]: \n
        - True -> Select the vocab with largest possibility.\n
        - False -> the next prediction will be sampled from the distribution.\n

        ----------------
        return predited 2d list of token indexs.
        '''

        ######### To sort the input by lengths and get lengths #########
        _, data, lengths = self.vocab.tranform_to_input_data_from_seq_idx_with_caseid(
            data)

        ######### Predict #########
        predicted_list = self.predict(
            input=data,
            lengths=lengths, n_steps=n_steps, use_argmax=use_argmax
        )

        return predicted_list

    def predicting_from_list_of_vacab_trace(
        self, data: List[List[str]], n_steps: int = None, use_argmax=False
    ):
        '''
        [data]: 2D list of tokens.
        [n_step]: how many steps will be predicted. If n_step == None, model will
        repeat to predict till <EOS> token show.\n
        [use_argmax]: \n
        - True -> Select the vocab with largest possibility.\n
        - False -> the next prediction will be sampled from the distribution.\n

        ----------------
        return predited 2d list of tokens.
        '''

        ######### Transform to index #########
        data = [self.vocab.list_of_vocab_to_index(l) for l in data]

        ######### Predict #########
        predicted_list = self.predicting_from_list_of_idx_trace(
            data=data, n_steps=n_steps, use_argmax=use_argmax
        )

        ######### Tranform back to vocab #########
        predicted_list = [
            self.vocab.list_of_index_to_vocab(l) for l in predicted_list
        ]

        return predicted_list

    def get_prediction_list_from_out(self, out, mask=None):
        predicted = tf.math.argmax(out, axis=-1)  # (B, S)
        selected_predictions = tf.boolean_mask(
            predicted, mask)

        return selected_predictions.numpy().tolist()

    def get_target_list_from_target(self, target, mask=None):
        selected_targets = tf.boolean_mask(
            target, mask
        )
        return selected_targets.numpy().tolist()

    def generate_mask(self, target):
        return target != 0
