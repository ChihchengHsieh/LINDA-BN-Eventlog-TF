class EventLogDiCE():
    def __init__(self, possible_activities, possible_resources, possible_amount, dice_model):
        self.possible_activities = possible_activities 
        self.possible_amount = possible_amount
        self.possible_resources = possible_resources
        self.dice_model = dice_model

    def min_max_scale_amount(self, input_amount, inverse=False):
        min_a = self.possible_amount[0]
        max_a = self.possible_amount[1]
        min_max_range = (max_a - min_a)
        if inverse:
            return (input_amount * min_max_range ) + min_a
        else:
            return input_amount - min_a / min_max_range

    def get_valid_cf(self, amount_cf, ohe_activity_cf, ohe_resource_cf):
        return tf.clip_by_value(amount_cf, self.possible_amount[0], self.possible_amount[1]) ,tf.one_hot(tf.argmax(ohe_activity_cf, axis= -1), depth=len(self.possible_activities)), tf.one_hot(tf.argmax(ohe_resource_cf, axis= -1), depth=len(self.possible_resources))

    def just_train(self, example_idx_activities_no_tag, example_idx_resources_no_tag, example_amount_input):
        ohe_activity_cf, ohe_resource_cf = transform_to_ohe_normalized_input(example_idx_activities_no_tag, example_idx_resources_no_tag, self.possible_activities, self.possible_resources)
        amount_cf = tf.Variable(example_amount_input)
        ohe_activity_cf = tf.Variable(ohe_activity_cf)
        ohe_resource_cf = tf.Variable(ohe_resource_cf)
        ohe_resource_backup =  ohe_resource_cf.numpy()
        ohe_activity_backup =  ohe_activity_cf.numpy()
        amount_backup = amount_cf.numpy()
        prediction = round(self.dice_model(
                    [
                        amount_cf,
                        ohe_activity_cf,
                        ohe_resource_cf
                    ]
                ).numpy()[0, 0])

        print_big(prediction, "Original Prediction")
        desired_pred = 1 - prediction
        print_big(desired_pred, "Desired Prediction")
        optim = tf.keras.optimizers.Adam(learning_rate=0.05)
        for i in range(200):
            with tf.GradientTape() as tape:
                ### Get prediction from cf
                cf_output = self.dice_model(
                    [
                        amount_cf,
                        ohe_activity_cf,
                        ohe_resource_cf
                    ]
                )
                print_big(cf_output, "cf output")


                activity_distance_loss = tf.reduce_sum(tf.pow((ohe_activity_cf - ohe_activity_backup), 2))
                resources_distance_loss = tf.reduce_sum(tf.pow(ohe_resource_cf - ohe_resource_backup, 2))
                amount_distance_loss = self.min_max_scale_amount(tf.pow(amount_cf - amount_backup,2))
                distance_loss = activity_distance_loss + resources_distance_loss + amount_distance_loss

                 ### Categorical contraint
                activity_cat_loss = tf.pow(tf.reduce_sum(ohe_activity_cf, axis=1) - 1, 2)
                resource_cat_loss = tf.pow(tf.reduce_sum(ohe_resource_cf, axis=1) - 1, 2)
                cat_loss = tf.reduce_sum(activity_cat_loss + resource_cat_loss)

                ### Using hinge loss since we have cat data
                class_loss = tf.keras.metrics.hinge(desired_pred, cf_output)
                loss = class_loss # + distance_loss * 0.01 + cat_loss
            grad = tape.gradient(loss,  [ amount_cf ,ohe_activity_cf, ohe_resource_cf])
            optim.apply_gradients(zip(grad, [amount_cf, ohe_activity_cf, ohe_resource_cf]))
            print_big(class_loss, "Loss")

            # ! Clipping cause error
            amount_cf = tf.Variable(tf.clip_by_value(amount_cf, self.possible_amount[0], self.possible_amount[1]))
            ohe_activity_cf = tf.Variable(tf.clip_by_value(ohe_activity_cf, 0.0, 1.0))
            ohe_resource_cf = tf.Variable(tf.clip_by_value(ohe_resource_cf, 0.0, 1.0))

            ####################
            temp_amount_cf, temp_ohe_activity_cf, temp_ohe_resource_cf = self.get_valid_cf(amount_cf, ohe_activity_cf, ohe_resource_cf)

            #### Get prediction

            cf_pred = round(self.dice_model(
            [
                temp_amount_cf,
                temp_ohe_activity_cf,
                temp_ohe_resource_cf
            ]
            ).numpy()[0, 0])

            print_big(cf_pred, "Pred output")

            if (cf_pred == desired_pred):
                print_big("Found!")
                return temp_amount_cf, temp_ohe_activity_cf, temp_ohe_resource_cf


        print_big(cf_output, "Final Output")

    def generate_counterfactual(self, query_activities, query_resources, query_amount, max_iter = 1000, verbose_freq = 50, lr = 0.005):

        ### Get the input for dice model (has to be differentiable)
        ohe_activity_cf, ohe_resource_cf = transform_to_ohe_normalized_input(query_activities, query_resources, self.possible_activities, self.possible_resources)

        ohe_activity_backup = ohe_activity_cf.numpy()
        ohe_resource_backup = ohe_resource_cf.numpy()
        amount_backup = query_amount.numpy()

        ## Create the cf variable 
        amount_cf = tf.Variable(query_amount)
        ohe_activity_cf = tf.Variable(ohe_activity_cf)
        ohe_resource_cf = tf.Variable(ohe_resource_cf)
        self.amount_cf = amount_cf

        ## Get current prediction.
        prediction = round(self.dice_model(
            [
                amount_cf,
                ohe_activity_cf,
                ohe_resource_cf
            ]
        ).numpy()[0, 0])
        
        print_big(prediction, "Original Prediction")

        desired_pred = 1 - prediction

        print_big(desired_pred, "Desired Prediction")

        ## init optimizer
        optim = tf.keras.optimizers.Adam(learning_rate=lr)

        for i in range(max_iter):
            if i % verbose_freq == 0 and i != 0:
                print_big(f"Current Loss [{loss.numpy()}]", f"Step {i}")

            with tf.GradientTape() as tape:
                ### Get prediction from cf
                cf_output = self.dice_model(
                    [
                        amount_cf,
                        ohe_activity_cf,
                        ohe_resource_cf
                    ]
                )

                ### Using hinge loss since we have cat data
                class_loss = tf.keras.metrics.hinge(desired_pred, cf_output)
                self.class_loss = class_loss
                self.cf_output = cf_output

                activity_distance_loss = tf.reduce_sum(tf.pow((ohe_activity_cf - ohe_activity_backup), 2))
                resources_distance_loss = tf.reduce_sum(tf.pow(ohe_resource_cf - ohe_resource_backup, 2))
                amount_distance_loss = self.min_max_scale_amount(tf.pow(amount_cf - amount_backup,2))
                distance_loss = activity_distance_loss + resources_distance_loss + amount_distance_loss
                self.distance_loss = distance_loss

                ### Categorical contraint
                activity_cat_loss = tf.pow(tf.reduce_sum(ohe_activity_cf, axis=1) - 1, 2)
                resource_cat_loss = tf.pow(tf.reduce_sum(ohe_resource_cf, axis=1) - 1, 2)
                cat_loss = tf.reduce_sum(activity_cat_loss + resource_cat_loss)

                # self.cat_loss = cat_loss
                loss = cf_output
                # loss = class_loss #  + distance_loss + cat_loss
                self.loss = loss

            ### Get gradient
            # grad = tape.gradient(loss, [amount_cf, ohe_activity_cf, ohe_resource_cf])
            grad = tape.gradient(loss, [ amount_cf ,ohe_activity_cf, ohe_resource_cf])
            self.grad = grad

            ### Update CF to this direction
            optim.apply_gradients(zip(grad, [amount_cf, ohe_activity_cf, ohe_resource_cf]))

            ### Get a valid version of CF
            temp_amount_cf, temp_ohe_activity_cf, temp_ohe_resource_cf = self.get_valid_cf(amount_cf, ohe_activity_cf, ohe_resource_cf)

            #### Get prediction

            cf_pred = round(self.dice_model(
            [
                temp_amount_cf,
                temp_ohe_activity_cf,
                temp_ohe_resource_cf
            ]
            ).numpy()[0, 0])

            if (cf_pred == desired_pred):
                return amount_cf, ohe_activity_cf, ohe_resource_cf
        
    