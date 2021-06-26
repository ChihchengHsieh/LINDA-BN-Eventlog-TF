from Utils.PrintUtils import print_big
import tensorflow as tf


class DiCEBinaryOutputModel(tf.keras.Model):
    '''
    It's a new model classifying where the destination is prefered.
    '''

    def __init__(self, model, vocab, desired: int, trace_length: int, without_tags_vocabs):
        super(DiCEBinaryOutputModel, self).__init__()
        self.model = model
        self.vocab = vocab
        self.desired = desired
        self.trace_length = trace_length
        self.without_tags_vocabs = without_tags_vocabs
        self.all_predicted = []
        self.all_trace = []
        self.all_model_out = []
        self.all_cf_input = []

    def call(self, input):
        '''
        Input will be one-hot encoded tensor.
        '''
        self.all_cf_input.append(input.numpy())

        # Get real input from the one-hot encoded tensor.
        input = tf.argmax(
            tf.stack(tf.split(input, self.trace_length, axis=-1,), axis=1), axis=-1)

        # transfer to the input with tags.
        input = tf.constant(self.vocab.list_of_vocab_to_index_2d(
            [[self.without_tags_vocabs[idx] for idx in tf.squeeze(input).numpy()]]), dtype=tf.int64)

        self.all_trace.append(input.numpy())

        # Concate the <SOS> tag in the first step.
        input = tf.concat(
            [tf.constant([[2]], dtype=tf.int64),  input], axis=-1)

        # Feed to the model
        out, _ = self.model(input)

        predicted_idx = tf.argmax(out[:, -1, :], axis=-1).numpy()[0]

        self.all_model_out.append(out.numpy())
        self.all_predicted.append(predicted_idx)


        return out[:, -1, self.desired: self.desired+1], 1.0 if predicted_idx == self.desired else 0.0
