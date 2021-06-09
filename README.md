# LINDA-BN-Eventlog-TF


# Counterfactual
## DiCE

### BPI2012 Activities 

![](https://github.com/ChihchengHsieh/LINDA-BN-Eventlog-TF/blob/master/imgs/Activities-1.png?raw=true)
![](https://github.com/ChihchengHsieh/LINDA-BN-Eventlog-TF/blob/master/imgs/Activities-2.png?raw=true)

### BPI2012 Paths

![](https://github.com/ChihchengHsieh/LINDA-BN-Eventlog-TF/blob/master/imgs/DiscoPaths.png?raw=true)

### Model to use

```python
class ExtractingLastTimeStampProbDistributionLayer(tf.keras.layers.Layer):
    '''
    It's a new model classifying where the destination is prefered.
    '''
    def __init__(self, explainer: ExplainingController, desired: int, trace_length: int, without_tags_vocabs):
        super(ExtractingLastTimeStampProbDistributionLayer, self).__init__()
        self.explainer = explainer
        self.desired = desired
        self.trace_length = trace_length
        
    def call(self, input):
        '''
        Input will be one-hot encoded tensor.
        '''

        ### Get real input from the one-hot encoded tensor.
        input = tf.argmax(tf.stack(tf.split(input,self.trace_length, axis=-1,), axis = 1), axis = -1)

        ### transfer to the input with tags.
        input = tf.constant(explainer.vocab.list_of_vocab_to_index_2d([[without_tags_vocabs[idx] for idx in tf.squeeze(input).numpy()]]), dtype=tf.int64)

        ## Concate the <SOS> tag in the first step.
        input = tf.concat([tf.constant([[2]], dtype=tf.int64) ,  input], axis=-1)

        ## Feed to the model
        out = explainer.model(input)

        ## Take the activty with max possibility.
        out = tf.argmax(out[0][:, -1, :], axis = -1)

        ## Determine whether the 
        return tf.expand_dims(tf.cast(out == self.desired, dtype=tf.float32), axis = 0)
```

### Issues:


#### Case trace length
From the above paths graph, we can know "Declined" or "Cancled" cases tend to have a shorter path. Therefore, it will be hard to pertube the "Declined&Canceled" cases to "Approved" cases.

#### Found counterfactuals doesn't make sense in terms of process flows.

For example When we get a case like this:

```python
['A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'W_Completeren aanvraag_SCHEDULE',
'W_Completeren aanvraag_START', 'A_ACCEPTED_COMPLETE', 'A_FINALIZED_COMPLETE', 'O_SELECTED_COMPLETE', 'O_CREATED_COMPLETE',
'O_SENT_COMPLETE', 'W_Nabellen offertes_SCHEDULE', 'W_Completeren aanvraag_COMPLETE', 'W_Nabellen offertes_START',
'W_Nabellen offertes_COMPLETE', 'W_Nabellen offertes_START', 'O_CANCELLED_COMPLETE', 'O_SELECTED_COMPLETE', 'O_CREATED_COMPLETE',
'O_SENT_COMPLETE', 'W_Nabellen offertes_SCHEDULE', 'W_Nabellen offertes_COMPLETE', 'W_Nabellen offertes_START',
'W_Nabellen offertes_COMPLETE', 'W_Nabellen offertes_START', 'O_SENT_BACK_COMPLETE', 'W_Valideren aanvraag_SCHEDULE',
'W_Nabellen offertes_COMPLETE', 'W_Valideren aanvraag_START', 'W_Valideren aanvraag_COMPLETE', 'W_Valideren aanvraag_START',
'W_Nabellen incomplete dossiers_SCHEDULE', 'W_Valideren aanvraag_COMPLETE', 'W_Nabellen incomplete dossiers_START',
'W_Nabellen incomplete dossiers_COMPLETE', 'W_Nabellen incomplete dossiers_START', 'W_Nabellen incomplete dossiers_COMPLETE',
'W_Nabellen incomplete dossiers_START', 'W_Nabellen incomplete dossiers_COMPLETE', 'W_Nabellen incomplete dossiers_START',
'W_Nabellen incomplete dossiers_COMPLETE', 'W_Nabellen incomplete dossiers_START', 'W_Nabellen incomplete dossiers_COMPLETE',
'W_Nabellen incomplete dossiers_START', 'W_Valideren aanvraag_SCHEDULE', 'W_Nabellen incomplete dossiers_COMPLETE',
'W_Valideren aanvraag_START', 'O_ACCEPTED_COMPLETE', 'A_ACTIVATED_COMPLETE', 'A_REGISTERED_COMPLETE', 'A_APPROVED_COMPLETE',
'W_Valideren aanvraag_COMPLETE']
```

And the counterfactual we found is:
```python
['A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE',
'A_ACCEPTED_COMPLETE', 'A_ACCEPTED_COMPLETE', 'A_ACCEPTED_COMPLETE', 'A_FINALIZED_COMPLETE',
'O_SELECTED_COMPLETE', 'O_CREATED_COMPLETE', 'A_ACCEPTED_COMPLETE', 'A_PREACCEPTED_COMPLETE',
'W_Valideren aanvraag_START', 'W_Beoordelen fraude_COMPLETE', 'W_Afhandelen leads_SCHEDULE',
'O_SENT_COMPLETE', 'A_CANCELLED_COMPLETE', 'W_Afhandelen leads_COMPLETE', 'W_Wijzigen contractgegevens_SCHEDULE',
'A_REGISTERED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'O_ACCEPTED_COMPLETE', 'A_APPROVED_COMPLETE',
'W_Nabellen incomplete dossiers_COMPLETE', 'O_SENT_COMPLETE', 'A_REGISTERED_COMPLETE', 'A_DECLINED_COMPLETE',
'A_FINALIZED_COMPLETE', 'O_DECLINED_COMPLETE', 'A_ACTIVATED_COMPLETE', 'W_Valideren aanvraag_COMPLETE',
'W_Nabellen incomplete dossiers_SCHEDULE', 'A_PARTLYSUBMITTED_COMPLETE', 'W_Wijzigen contractgegevens_SCHEDULE',
'W_Afhandelen leads_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'W_Completeren aanvraag_COMPLETE', 'W_Valideren aanvraag_COMPLETE',
'W_Completeren aanvraag_START', 'W_Valideren aanvraag_COMPLETE', 'W_Nabellen offertes_START', 'A_SUBMITTED_COMPLETE',
'W_Valideren aanvraag_COMPLETE', 'A_REGISTERED_COMPLETE', 'O_CREATED_COMPLETE', 'A_APPROVED_COMPLETE', 'W_Nabellen offertes_COMPLETE',
'A_ACCEPTED_COMPLETE', 'O_CREATED_COMPLETE', 'W_Valideren aanvraag_COMPLETE', 'O_DECLINED_COMPLETE', 'O_SELECTED_COMPLETE']
```

From above cf, we can found the "A_ACCEPTED_COMPLETE" appear 3 times after 'A_PREACCEPTED_COMPLETE', which is not a valid behaviour in BPI2012 cases. It also have "A_APPROVED_COMPLETE", "A_REGISTERED_COMPLETE" and "O_ACCEPTED_COMPLETE", which indicate a successful case.

The permutations created by the DiCE algorithm doesn't follow the rules of progress in BPI2012:

![](https://github.com/ChihchengHsieh/LINDA-BN-Eventlog-TF/blob/master/imgs/ProcessPathAndCases.png?raw=true)

It's like an [one-pixel attack](https://arxiv.org/pdf/1710.08864.pdf) but in event logs format.
