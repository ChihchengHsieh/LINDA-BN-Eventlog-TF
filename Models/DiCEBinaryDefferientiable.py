from Utils.PrintUtils import print_big
from functools import reduce
import tensorflow as tf
from tensorflow.python.keras.backend import dtype
from tensorflow.python.ops.gen_dataset_ops import AnonymousMultiDeviceIterator


class DiCEBinaryDefferentiable(tf.keras.Model):
    '''
    The differentiable version (Not using argmax).
    '''

    def __init__(self, model, vocab, resources, desired: int, trace_length: int, possible_activities, possible_resources, sos_idx_activity, sos_idx_resource, amount_min, amount_max):
        super(DiCEBinaryDefferentiable, self).__init__()

        self.model = model
        self.vocab = vocab
        self.desired = desired
        self.trace_length = trace_length
        self.resources = resources
        self.possible_activities = possible_activities
        self.possible_resources = possible_resources
        self.sos_idx_activity = sos_idx_activity
        self.sos_idx_resource = sos_idx_resource
        self.amount_min = amount_min
        self.amount_max = amount_max

        self.all_predicted = []
        self.all_trace = []
        self.all_model_out = []
        self.all_cf_input = []
        self.all_resource = []
        self.all_amount = []

    def call(self, input):
        '''
        Input will be one-hot encoded tensor.
        '''
        # print("Detect input with shape: %s" % str(input.shape))

        self.all_cf_input.append(input)
        traces, resources, amount = input
        # For original dice model
        # self.all_cf_input.append(input.numpy())
        # amount, traces, resources = self.ohe_to_model_input(input)

        self.all_trace.append(traces.numpy())
        self.all_resource.append(resources.numpy())
        self.all_amount.append(amount.numpy())

        out, _ = self.model(traces, resources, amount, training=False)

        predicted_idx = tf.argmax(out[:, -1, :], axis=-1).numpy()[0]

        self.all_model_out.append(out.numpy())
        self.all_predicted.append(predicted_idx)

        return out[:, -1, self.desired: self.desired+1],  1.0 if predicted_idx == self.desired else 0.0

    def ohe_to_model_input(self, input):
        amount, activities, resources = tf.split(input, [1, self.trace_length * len(
            self.possible_activities), self.trace_length * len(self.possible_resources)], axis=1)
        activities = tf.reshape(
            activities, [self.trace_length, len(self.possible_activities)])
        activities = self.map_to_original_vocabs(
            self.possible_activities, self.vocab.vocabs, activities)
        activities = tf.concat([tf.one_hot([self.sos_idx_activity], depth=len(
            self.vocab.vocabs)), activities], axis=0)[tf.newaxis, :, :]

        resources = tf.reshape(
            resources, [self.trace_length, len(self.possible_resources)])
        resources = self.map_to_original_vocabs(
            self.possible_resources, self.resources, resources)
        resources = tf.concat([tf.one_hot([self.sos_idx_resource], depth=len(
            self.resources)), resources], axis=0)[tf.newaxis, :, :]
        amount = (amount * (self.amount_max - self.amount_min)) + \
            self.amount_min
        return amount, activities, resources
