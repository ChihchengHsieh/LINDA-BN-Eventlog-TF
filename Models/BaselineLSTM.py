import tensorflow as tf
from Utils import VocabDict


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
        _, padded_data_traces, _ = data
        out = self.call(padded_data_traces, training=training)
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
        accuracy = tf.math.reduce_mean(tf.cast(tf.boolean_mask(y_true == pred_value, y_true != 0), dtype= tf.float32)).numpy()
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
