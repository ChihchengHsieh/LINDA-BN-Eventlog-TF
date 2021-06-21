
import tensorflow as tf

class DiCEBinaryOutputModelWithResourceWithRestriction(tf.keras.Model):
    '''
    It's a new model classifying where the destination is prefered.
    '''

    def __init__(self, model, vocab, resources, desired: int, trace_length: int, without_tags_vocabs, without_tags_resources, sos_idx_activity, sos_idx_resource, amount_min, amount_max, possible_resource, possible_activities):
        super(DiCEBinaryOutputModelWithResourceWithRestriction, self).__init__()
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

        self.possible_resource = possible_resource
        self.possible_activities = possible_activities

        self.amount_min = amount_min
        self.amount_max = amount_max
        self.resources = resources

        self.pos_activity_count = [len(i) for i in self.possible_activities]

    def call(self, input):
        '''
        Input will be one-hot encoded tensor.
        '''
        # print("Detect input with shape: %s" % str(input.shape))
        self.all_cf_input.append(input.numpy())

        split_portion = [1, sum(self.pos_activity_count[:self.trace_length]),
                         len(self.possible_resource) * self.trace_length]

        self.input_data = input
        self.split_portion = split_portion

        ##! TODO ### Transfer to the vocab first from the possible ###

        ##! Need to count the dimension ###

        amount, traces, resources = tf.split(input, split_portion, axis=1)

        # print("Origin Amount")
        # print(amount)
        amount = (amount * (self.amount_max - self.amount_min)) + \
            self.amount_min
        # print('Amount scale back')
        # print(amount)

        # print("Amount value is %.2f" % (amount.numpy()) )

        # TODO: add multiple selection here
        # split_trace_to_steps
        step＿activiy_ohe = tf.split(
            traces, self.pos_activity_count[:self.trace_length], axis=-1)
        self.step＿activiy_ohe = step＿activiy_ohe

        index_trace = [tf.argmax(ohe, axis=-1).numpy()[0] for ohe in step＿activiy_ohe]
        self.index_trace = index_trace
        # Map to vocab

        vocab_trace = [self.possible_activities[s][i]
                       for s, i in enumerate(index_trace)]

        traces = tf.constant(self.vocab.list_of_vocab_to_index(vocab_trace), dtype=tf.int64)[tf.newaxis, :]

        # traces = tf.constant(self.vocab.list_of_vocab_to_index_2d(
        #     [[self.without_tags_vocabs[idx] for idx in tf.squeeze(traces).numpy()]]), dtype=tf.int64)

        # traces = tf.argmax(
        #     tf.stack(tf.split(traces, self.trace_length, axis=-1,), axis=1), axis=-1)

        resources = tf.argmax(
            tf.stack(tf.split(resources, self.trace_length, axis=-1,), axis=1), axis=-1)

        # transfer to the input with tags.
        

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
