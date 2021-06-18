import tensorflow as tf
class DiCEBinaryOutputModelWithResource(tf.keras.Model):
    '''
    It's a new model classifying where the destination is prefered.
    '''

    def __init__(self, model, vocab, resources, desired: int, trace_length: int, without_tags_vocabs, without_tags_resources, sos_idx_activity, sos_idx_resource, amount_min, amount_max):
        super(DiCEBinaryOutputModelWithResource, self).__init__()
        self.model = model
        self.vocab = vocab
        self.desired = desired
        self.trace_length = trace_length
        self.without_tags_vocabs = without_tags_vocabs
        self.without_tags_resources = without_tags_resources
        self.sos_idx_activity = sos_idx_activity
        self.sos_idx_resource = sos_idx_resource
        self.all_predicted = []
        self.all_trace = []
        self.all_model_out = []
        self.all_cf_input = []
        self.all_resource = []
        self.all_amount = []

        self.amount_min = amount_min
        self.amount_max = amount_max

        self.resources = resources


def call(self, input):
    '''
    Input will be one-hot encoded tensor.
    '''
    # print("Detect input with shape: %s" % str(input.shape))
    self.all_cf_input.append(input.numpy())

    split_portion = [1, len(self.without_tags_vocabs) * self.trace_length,
                     len(self.without_tags_resources) * self.trace_length]

    amount, traces, resources = tf.split(input, split_portion, axis=1)

    amount = (amount * (self.amount_max - self.amount_min)) + self.amount_min

    # print("Amount value is %.2f" % (amount.numpy()) )

    traces = tf.argmax(
        tf.stack(tf.split(traces, self.trace_length, axis=-1,), axis=1), axis=-1)

    resources = tf.argmax(
        tf.stack(tf.split(resources, self.trace_length, axis=-1,), axis=1), axis=-1)

    # transfer to the input with tags.
    traces = tf.constant(self.vocab.list_of_vocab_to_index_2d(
        [[self.without_tags_vocabs[idx] for idx in tf.squeeze(traces).numpy()]]), dtype=tf.int64)

    resources = tf.constant(
        [[self.resources.index(self.without_tags_resources[idx]) for idx in tf.squeeze(resources).numpy()]], dtype=tf.int64)

    self.all_trace.append(traces.numpy())
    self.all_resource.append(resources.numpy())
    self.all_amount.append(amount.numpy())

    # Concate the <SOS> tag in the first step.
    traces = tf.concat(
        [tf.constant([[self.sos_idx_activity]], dtype=tf.int64),  traces], axis=-1)

    resources = tf.concat(
        [tf.constant([[self.sos_idx_resource]], dtype=tf.int64), resources], axis=-1)

    # Feed to the model
    # print("Ready for input")
    out, _ = self.model(traces, resources, tf.squeeze(amount, axis=-1))

    self.all_model_out.append(out.numpy())
    self.all_predicted.append(tf.argmax(out[:, -1, :], axis=-1).numpy())

    return out[:, -1, self.desired: self.desired+1]
