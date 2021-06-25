from inspect import trace
import tensorflow as tf
from Utils.PrintUtils import print_big
import time


class EventLogDiCE():
    def __init__(self, activity_vocab, resource_vocab, possible_amount, possible_activities, possible_resources, dice_model, scenario_model, scenario_using_hinge_loss=True):
        self.possible_activities = possible_activities
        self.possible_amount = possible_amount
        self.possible_resources = possible_resources
        self.dice_model = dice_model
        self.scenario_model = scenario_model
        self.scenario_using_hinge_loss = scenario_using_hinge_loss
        self.activity_vocab = activity_vocab
        self.resource_vocab = resource_vocab

    def min_max_scale_amount(self, input_amount, inverse=False):
        min_a = self.possible_amount[0]
        max_a = self.possible_amount[1]
        min_max_range = (max_a - min_a)
        if inverse:
            return (input_amount * min_max_range) + min_a
        else:
            return input_amount - min_a / min_max_range

    def transform_to_ohe_normalized_input(self, activities, resources):
        activity_cf = tf.one_hot(
            activities, depth=len(self.possible_activities))
        resource_cf = tf.one_hot(resources, depth=len(self.possible_resources))
        return activity_cf, resource_cf

    def get_valid_cf(self, amount_cf, ohe_activity_cf, ohe_resource_cf, use_sampling=False):
        if use_sampling:
            return (tf.clip_by_value(amount_cf, self.possible_amount[0], self.possible_amount[1]),
                    tf.squeeze(tf.one_hot(tf.random.categorical(
                        ohe_activity_cf, 1), depth=len(self.possible_activities)), axis=1),
                    tf.squeeze(tf.one_hot(tf.random.categorical(ohe_resource_cf, 1), depth=len(self.possible_resources)), axis=1))

        return tf.clip_by_value(amount_cf, self.possible_amount[0], self.possible_amount[1]), tf.one_hot(tf.argmax(ohe_activity_cf, axis=-1), depth=len(self.possible_activities)), tf.one_hot(tf.argmax(ohe_resource_cf, axis=-1), depth=len(self.possible_resources))

    def get_valid_scenario_loss(self, cf_input):

        # get out
        # expect the output to pass through sigmoid function.
        out, _ = self.scenario_model(*cf_input, training=False)
        self.scenario_out = out
        raise StopIteration
        # get loss
        if self.scenario_using_hinge_loss:
            loss = tf.reduce_sum(
                tf.keras.metrics.hinge(tf.ones_like(out), out))
        else:
            # using binary entropy
            loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(
                y_true=tf.ones_like(out), y_pred=out,  from_logits=True
            ))

        return loss

    def map_to_original_vocabs(self, reduced, original, input, trace_len):
        '''
        Expect ohe input.
        '''
        after_ = [None] * len(original)
        for i, a in enumerate(reduced):
            dest_index = original.index(a)
            after_[dest_index] = input[:, i:i+1]
        after_ = tf.concat([tf.zeros((trace_len, 1))
                           if a is None else a for a in after_], axis=1)
        return after_

    def transform_to_model_input(self, activitiy_cf, resource_cf, amount_cf, trace_len):
        # concate the sos

        activitiy = self.map_to_original_vocabs(
            self.possible_activities,
            self.activity_vocab.vocabs,
            activitiy_cf,
            trace_len,
        )[tf.newaxis, :, :]

        activity_sos_oh =  tf.one_hot(self.activity_vocab.sos_idx(), depth=len(
            self.activity_vocab))[tf.newaxis, tf.newaxis, :]
        
        activitiy = tf.concat(
            [
                activity_sos_oh,
                activitiy
            ],
            axis = 1
        )

        resources = self.map_to_original_vocabs(
            self.possible_resources,
            self.resource_vocab.vocabs,
            resource_cf,
            trace_len,
        )[tf.newaxis, :, :]

        resource_sos_oh =  tf.one_hot(self.resource_vocab.sos_idx(), depth=len(
            self.resource_vocab))[tf.newaxis, tf.newaxis, :]
        
        resources = tf.concat(
            [
                resource_sos_oh,
                resources
            ],
            axis = 1
        )


        amount = amount_cf
        return [activitiy, resources, amount]

    def run_pls(self,
                amount_input,
                idx_activities_no_tag,
                idx_resources_no_tag,
                use_valid_cf_only=False,
                use_sampling=True,
                class_loss_weight=1.0,
                distance_loss_weight=1e-8,
                cat_loss_weight=0.0,
                scenario_weight=1.0,
                verbose_freq=20,
                max_iter=200,
                lr=0.05,
                ):

        start_at = time.time()

        vocab_activity_no_tag = self.dice_model.vocab.list_of_index_to_vocab(
            idx_activities_no_tag)
        vocab_resource_no_tag = [self.dice_model.resources[r]
                                 for r in idx_resources_no_tag]

        ohe_activity_cf, ohe_resource_cf = self.transform_to_ohe_normalized_input(
            idx_activities_no_tag, idx_resources_no_tag)

        trace_len = len(idx_activities_no_tag) ## Need <SOS>

        amount_cf = tf.Variable(amount_input)
        ohe_activity_cf = tf.Variable(ohe_activity_cf)
        ohe_resource_cf = tf.Variable(ohe_resource_cf)

        ohe_resource_backup = ohe_resource_cf.numpy()
        ohe_activity_backup = ohe_activity_cf.numpy()
        amount_backup = amount_cf.numpy()

        self.amount_cf = amount_cf
        self.ohe_activity_cf = ohe_activity_cf
        self.ohe_resource_cf = ohe_resource_cf

        model_input = self.transform_to_model_input(
            ohe_activity_cf,
            ohe_resource_cf,
            amount_cf,
            trace_len,
        )

        self.ohe_activity_cf = ohe_activity_cf

        prediction = round(self.dice_model(
            model_input
        ).numpy()[0, 0])

        desired_pred = 1 - prediction
        print_big(
            f"Prediction: [{prediction}] | Desired: [{desired_pred}]", "Prediction")

        for i in range(max_iter):

            if (i != 0) and i % verbose_freq == 0:
                print_big(loss.numpy(), f"Step {i} Loss")

            optim = tf.keras.optimizers.Adam(learning_rate=lr)

            with tf.GradientTape() as tape:

                # Get prediction from cf
                # Maybe we can convert it to the original input here
                self.ohe_activity_cf = ohe_activity_cf
                self.ohe_resource_cf = ohe_resource_cf
                self.amount_cf = amount_cf
                self.trace_le = trace_len

                model_input = self.transform_to_model_input(
                    ohe_activity_cf,
                    ohe_resource_cf,
                    amount_cf,
                    trace_len
                )

                cf_output = self.dice_model(model_input)

                # Distance loss
                activity_distance_loss = tf.reduce_sum(
                    tf.pow((ohe_activity_cf - ohe_activity_backup), 2))
                resources_distance_loss = tf.reduce_sum(
                    tf.pow(ohe_resource_cf - ohe_resource_backup, 2))
                amount_distance_loss = self.min_max_scale_amount(
                    tf.pow(amount_cf - amount_backup, 2))
                distance_loss = activity_distance_loss + \
                    resources_distance_loss + amount_distance_loss

                # Categorical contraint
                activity_cat_loss = tf.pow(
                    tf.reduce_sum(ohe_activity_cf, axis=1) - 1, 2)
                resource_cat_loss = tf.pow(
                    tf.reduce_sum(ohe_resource_cf, axis=1) - 1, 2)
                cat_loss = tf.reduce_sum(activity_cat_loss + resource_cat_loss)

                # Class loss
                class_loss = tf.keras.metrics.hinge(desired_pred, cf_output)

                self.model_input = model_input

                scenario_loss = self.get_valid_scenario_loss(model_input)

                print_big(scenario_loss, "Scenario loss")

                # Get total loss
                loss = (class_loss_weight * class_loss) + (distance_loss * distance_loss_weight) + \
                    (cat_loss * cat_loss_weight) + \
                    (scenario_loss * scenario_weight)
                # loss = class_loss
                # loss = scenario_loss * scenario_weight

                print_big(loss, "Total loss")

            grad = tape.gradient(
                loss,  [amount_cf, ohe_activity_cf, ohe_resource_cf])

            self.grad = grad
            optim.apply_gradients(
                zip(grad, [amount_cf, ohe_activity_cf, ohe_resource_cf]))

            # Clipping the value for next round (This step has a risk to cause the nn can't find the local minimum.)
            amount_cf.assign(tf.clip_by_value(
                amount_cf, self.possible_amount[0], self.possible_amount[1]))
            ohe_activity_cf.assign(tf.clip_by_value(ohe_activity_cf, 0.0, 1.0))
            ohe_resource_cf.assign(tf.clip_by_value(ohe_resource_cf, 0.0, 1.0))

            # Get a valid cf close to current cf.
            temp_amount_cf, temp_ohe_activity_cf, temp_ohe_resource_cf = self.get_valid_cf(
                amount_cf, ohe_activity_cf, ohe_resource_cf)

            temp_model_input = self.transform_to_model_input(
                temp_ohe_activity_cf, temp_ohe_resource_cf, temp_amount_cf, trace_len)

            # Get prediction from the valid cf.
            cf_pred = round(self.dice_model(temp_model_input).numpy()[0, 0])

            if (use_valid_cf_only):
                # Replace current cf by valid cf.
                amount_cf.assign(temp_amount_cf)
                if (use_sampling):
                    # Using sampling
                    sample_amount, sample_activity, sample_resource = self.get_valid_cf(
                        amount_cf, ohe_activity_cf, ohe_resource_cf, use_sampling=True)
                    amount_cf.assign(sample_amount)
                    ohe_activity_cf.assign(sample_activity)
                    ohe_resource_cf.assign(sample_resource)
                else:
                    # Using argmax
                    ohe_activity_cf.assign(temp_ohe_activity_cf)
                    ohe_resource_cf.assign(temp_ohe_resource_cf)

            if (cf_pred == desired_pred):
                print_big(f"Running time: {time.time() - start_at:.2f}",
                          f"!Counterfactual Found in step [{i+1}]!")

                activity_out = [self.possible_activities[i]
                                for i in tf.argmax(temp_ohe_activity_cf, axis=-1).numpy()]
                resource_out = [self.possible_resources[i]
                                for i in tf.argmax(temp_ohe_resource_cf, axis=-1).numpy()]
                amount_out = amount_cf.numpy()[0]

                print_big(amount_input, "Input Amount")
                print_big(vocab_activity_no_tag, "Input Activities")
                print_big(vocab_resource_no_tag, "Input Resource")

                print_big(amount_out, "CF Amount")
                print_big(activity_out, "CF Activities")
                print_big(resource_out, "CF Resource")
                return amount_out, activity_out, resource_out

        activity_out = [self.possible_activities[i]
                        for i in tf.argmax(temp_ohe_activity_cf, axis=-1).numpy()]
        resource_out = [self.possible_resources[i]
                        for i in tf.argmax(temp_ohe_resource_cf, axis=-1).numpy()]
        amount_out = amount_cf.numpy()[0]

        print_big(amount_input, "Input Amount")
        print_big(vocab_activity_no_tag, "Input Activities")
        print_big(vocab_resource_no_tag, "Input Resource")

        print_big(amount_out, "Current CF Amount")
        print_big(activity_out, "Current CF Activities")
        print_big(resource_out, "Current CF Resource")
