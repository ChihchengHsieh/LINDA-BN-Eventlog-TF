# XAI Tensorflow version


# BPI2012 Activities 

![](https://github.com/ChihchengHsieh/LINDA-BN-Eventlog-TF/blob/master/imgs/Activities-1.png?raw=true)
![](https://github.com/ChihchengHsieh/LINDA-BN-Eventlog-TF/blob/master/imgs/Activities-2.png?raw=true)

# BPI2012 Paths

![](https://github.com/ChihchengHsieh/LINDA-BN-Eventlog-TF/blob/master/imgs/DiscoPaths.png?raw=true)


## Counterfactual

### DiCE

[[Documentation](http://interpret.ml/DiCE/index.html)]
[[GitHub](https://github.com/interpretml/DiCE)]
[[Paper](https://arxiv.org/abs/1905.07697)]

#### Model to use
Instead of using the LSTM baseline model directly, we have to implementation some modifications to run on **DiCE**.

![DiCEBinaryOutputModelImg](https://user-images.githubusercontent.com/37566901/121686437-8533bc00-cb04-11eb-827d-ab32dd4e5c64.png)

#### Example:

1. Assume we have input trace like this.
```python
['<SOS>', 'A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE',
'A_PREACCEPTED_COMPLETE', 'A_ACCEPTED_COMPLETE',
'O_SELECTED_COMPLETE', 'A_FINALIZED_COMPLETE', 'O_CREATED_COMPLETE',
'O_SENT_COMPLETE','W_Completeren aanvraag_COMPLETE',
'W_Nabellen offertes_COMPLETE']
```

2. If we throw it into the LSTM baseline model, we can get the result (Probability distribution of next activity):

![output_distribution](https://user-images.githubusercontent.com/37566901/121657112-74bf1980-cae3-11eb-85e6-3c0cf959b0db.png)

from above we can know the model think `"W_Nabellen offertes_COMPLETE"` has highest probability (0.640346) to be next activity.

3. Then, we feed the same trace to DiCE for generating couterfactuals. 

DiCE return a counterfactual like this:

![cf](https://user-images.githubusercontent.com/37566901/121657889-2d855880-cae4-11eb-884b-be1db3142558.png)

Or the full counterfataul trace like this:

```python
['A_ACCEPTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE',
'A_ACCEPTED_COMPLETE', 'O_SELECTED_COMPLETE', 'A_FINALIZED_COMPLETE', 'O_CREATED_COMPLETE',
'O_SENT_COMPLETE', 'W_Completeren aanvraag_COMPLETE', 'O_DECLINED_COMPLETE']
```

However, from above example, we can indentify some issues when using event log with DiCE.

#### Issues

- The counterfactual doesn't follow the constraints from BPI2012. Every case should start with `A_SUBMITTED_COMPLETE`. However, the counterfactaul replace it with `A_ACCEPTED_COMPLETE`, which occurs two times in this trace. 

![](https://github.com/ChihchengHsieh/LINDA-BN-Eventlog-TF/blob/master/imgs/ProcessPathAndCases.png?raw=true)

- The fail trace is hard to find counterfactual. 
The average length for successful cases and failed cases are `27.94` and `10.53`, respectively. When we're trying to take a fail case with `length = 10` to find a successful case counterfactual, it will be hard since the LSTM will indetify most of the short trace to be declined or canceled and *our approach can't vary the length of trace*.




[one-pixel attack](https://arxiv.org/pdf/1710.08864.pdf)



## Couterfactual with resource and amount

### Original proposed architecture.

![IMG_0476](https://user-images.githubusercontent.com/37566901/122576443-0a831780-d095-11eb-9ba0-1d7f7dac16e6.jpg)


### My implementation

When using bidirectional-LSTM, task of predicting next event will have problem during training. Therefore, I use normal LSTM.

![image](https://user-images.githubusercontent.com/37566901/122573878-66986c80-d092-11eb-84ee-de041657966b.png)

### Example (A_ACTIVATED_COMPLETE => A_DECLINED_COMPLETE) :

I retrieve a case from test set, it looks like this:

#### Trace 

```python
['A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'W_Afhandelen leads_COMPLETE', 'W_Completeren aanvraag_COMPLETE',
'A_ACCEPTED_COMPLETE', 'A_FINALIZED_COMPLETE', 'O_SELECTED_COMPLETE', 'O_CREATED_COMPLETE', 'O_SENT_COMPLETE', 'W_Completeren aanvraag_COMPLETE',
'W_Nabellen offertes_COMPLETE', 'W_Nabellen offertes_COMPLETE', 'W_Nabellen offertes_COMPLETE', 'W_Nabellen offertes_COMPLETE',
'W_Nabellen offertes_COMPLETE', 'O_SENT_BACK_COMPLETE', 'W_Nabellen offertes_COMPLETE', 'W_Valideren aanvraag_COMPLETE', 'W_Valideren aanvraag_COMPLETE',
'W_Nabellen incomplete dossiers_COMPLETE', 'O_CANCELLED_COMPLETE', 'O_SELECTED_COMPLETE', 'O_CREATED_COMPLETE', 'O_SENT_COMPLETE',
'W_Nabellen incomplete dossiers_COMPLETE', 'W_Nabellen incomplete dossiers_COMPLETE', 'W_Nabellen incomplete dossiers_COMPLETE', 
'W_Nabellen incomplete dossiers_COMPLETE', 'O_SENT_BACK_COMPLETE', 'W_Nabellen incomplete dossiers_COMPLETE', 'O_ACCEPTED_COMPLETE',
'A_APPROVED_COMPLETE', 'A_REGISTERED_COMPLETE']
```

#### Resource (Staff ID)

```python
['112', '112', '11001', '11001', '11180',
'11201', '11201', '11201', '11201', '11201', '11201',
'11201', '11119', '11119', '10889', '11122', '11029',
'UNKNOWN', '10138', '10138', '10982', '11169', '11169',
'11169', '11169', '11169', '11169', '10913', '11049', 
'10789', '10789', '10138', '10138', '10138']
```

#### Amount
```
5000
```


### Put it into our model, we can get result:
```
Predicted activity with highest probability (1.00) is "A_ACTIVATED_COMPLETE"
```

It's not surprise that "A_REGISTERED_COMPLETE" usually followed by "A_ACTIVATED_COMPLETE".

![image](https://user-images.githubusercontent.com/37566901/122577271-eecc4100-d095-11eb-973d-9e0bab6acebe.png)

### We set desired_activity = 'A_DECLINED_COMPLETE', and find a counterfactual: 

![image](https://user-images.githubusercontent.com/37566901/122583238-5ab1a800-d09c-11eb-8e94-006b5ff151f2.png)

#### Activities:

```python
['A_ACTIVATED_COMPLETE', 'A_ACTIVATED_COMPLETE', 'W_Beoordelen fraude_COMPLETE', 'W_Valideren aanvraag_COMPLETE', 'A_ACCEPTED_COMPLETE',
'O_DECLINED_COMPLETE', 'O_SELECTED_COMPLETE', 'O_SELECTED_COMPLETE', 'O_SENT_BACK_COMPLETE', 'W_Afhandelen leads_COMPLETE',
'A_PREACCEPTED_COMPLETE', 'W_Afhandelen leads_COMPLETE', 'A_REGISTERED_COMPLETE', 'O_CREATED_COMPLETE', 'O_SENT_BACK_COMPLETE',
'O_SENT_COMPLETE', 'O_SELECTED_COMPLETE', 'A_ACTIVATED_COMPLETE', 'A_ACCEPTED_COMPLETE', 'O_SENT_BACK_COMPLETE', 'O_SENT_BACK_COMPLETE',
'O_CREATED_COMPLETE', 'A_CANCELLED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'O_SENT_BACK_COMPLETE', 'A_REGISTERED_COMPLETE', 'O_ACCEPTED_COMPLETE',
'O_SELECTED_COMPLETE', 'W_Afhandelen leads_COMPLETE', 'O_ACCEPTED_COMPLETE', 'W_Afhandelen leads_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE',
'O_SENT_COMPLETE', 'O_SENT_BACK_COMPLETE']
```

#### Resources:

```python
['10188', '11269', '10913', '10914', '11200', '11201', '10863',
'11119', '10809', '10859', '10125', '11120', '11254', '11254',
'10862', '10862', '10929', '11259', '11111', '10125', '11002',
'10899', '10912', '11169', '10861', '10910', 'UNKNOWN', '11119',
'10931', 'UNKNOWN', '10971', '10982', '11259', '10188']
```

#### Amount:
```
35
```

##### Still has the issue mentioned in the [previous example](https://github.com/ChihchengHsieh/LINDA-BN-Eventlog-TF/blob/master/README.md#issues).


-----------------

### Example (A_DECLINED_COMPLETE => A_DECLINED_COMPLETE)

#### Input

##### Trace:

```python
'A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE',
'W_Afhandelen leads_COMPLETE', 'W_Completeren aanvraag_COMPLETE', 'A_ACCEPTED_COMPLETE',
'O_SELECTED_COMPLETE', 'A_FINALIZED_COMPLETE', 'O_CREATED_COMPLETE', 'O_SENT_COMPLETE',
'W_Completeren aanvraag_COMPLETE', 'O_SELECTED_COMPLETE', 'O_CANCELLED_COMPLETE',
'O_CREATED_COMPLETE', 'O_SENT_COMPLETE', 'W_Nabellen offertes_COMPLETE', 'O_CANCELLED_COMPLETE',
'O_SELECTED_COMPLETE', 'O_CREATED_COMPLETE', 'O_SENT_COMPLETE', 'W_Nabellen offertes_COMPLETE',
'W_Nabellen offertes_COMPLETE', 'W_Nabellen offertes_COMPLETE', 'W_Nabellen offertes_COMPLETE',
'W_Nabellen offertes_COMPLETE', 'O_SENT_BACK_COMPLETE', 'W_Nabellen offertes_COMPLETE',
'W_Valideren aanvraag_COMPLETE', 'W_Nabellen incomplete dossiers_COMPLETE', 'W_Nabellen incomplete dossiers_COMPLETE',
'W_Nabellen incomplete dossiers_COMPLETE', 'W_Nabellen incomplete dossiers_COMPLETE',
'W_Nabellen incomplete dossiers_COMPLETE', 'W_Nabellen incomplete dossiers_COMPLETE',
'W_Nabellen incomplete dossiers_COMPLETE', 'W_Nabellen incomplete dossiers_COMPLETE',
'W_Nabellen incomplete dossiers_COMPLETE', 'W_Nabellen incomplete dossiers_COMPLETE',
'W_Nabellen incomplete dossiers_COMPLETE', 'W_Nabellen incomplete dossiers_COMPLETE',
'W_Nabellen incomplete dossiers_COMPLETE', 'W_Nabellen incomplete dossiers_COMPLETE',
'W_Nabellen incomplete dossiers_COMPLETE', 'W_Nabellen incomplete dossiers_COMPLETE',
'W_Nabellen incomplete dossiers_COMPLETE', 'W_Nabellen incomplete dossiers_COMPLETE',
'W_Nabellen incomplete dossiers_COMPLETE', 'W_Nabellen incomplete dossiers_COMPLETE',
'W_Nabellen incomplete dossiers_COMPLETE', 'W_Nabellen incomplete dossiers_COMPLETE',
'O_DECLINED_COMPLETE'
```

##### Resource:
```python
['112', '112', '10863', '10863', '11169', '11003',
'11003', '11003', '11003', '11003', '11003', '11003',
'11003', '11003', '11003', '11003', '11003', '11003',
'11003', '11003', '11003', '11003', '10913', '11180',
'11000', '10899', '10899', '10899', '10899', '11121',
'11000', '10861', '10909', '10909', '10913', '11002',
'11002', '11179', '11009', '11259', '11181', '11181',
'11181', '11000', '11169', '11122', '10982', '11003',
'11003', '11049', '10972']
```
##### Amount:
```python
5800.0
```
---
#### Prediction:
Predicted activity with highest probability (1.00) is "A_DECLINED_COMPLETE"

![image](https://user-images.githubusercontent.com/37566901/122692895-ccdfe380-d27a-11eb-89b5-253eae9eede7.png)

---
#### Counterfactaul:

##### Trace:
```python
['W_Nabellen offertes_COMPLETE', 'A_SUBMITTED_COMPLETE', 'O_SENT_COMPLETE',
'O_ACCEPTED_COMPLETE', 'O_CANCELLED_COMPLETE', 'O_SELECTED_COMPLETE',
'W_Valideren aanvraag_COMPLETE', 'O_CANCELLED_COMPLETE', 'O_DECLINED_COMPLETE',
'O_ACCEPTED_COMPLETE', 'W_Afhandelen leads_COMPLETE', 'O_DECLINED_COMPLETE',
'A_SUBMITTED_COMPLETE', 'O_SENT_COMPLETE', 'O_SENT_COMPLETE', 'A_ACTIVATED_COMPLETE',
'A_ACCEPTED_COMPLETE', 'W_Completeren aanvraag_COMPLETE', 'A_APPROVED_COMPLETE',
'O_CREATED_COMPLETE', 'A_DECLINED_COMPLETE', 'W_Completeren aanvraag_COMPLETE',
'O_DECLINED_COMPLETE', 'A_ACTIVATED_COMPLETE', 'A_REGISTERED_COMPLETE',
'W_Afhandelen leads_COMPLETE', 'A_SUBMITTED_COMPLETE', 'O_CREATED_COMPLETE',
'A_SUBMITTED_COMPLETE', 'W_Beoordelen fraude_COMPLETE', 'O_SELECTED_COMPLETE',
'A_ACCEPTED_COMPLETE', 'W_Nabellen offertes_COMPLETE', 'W_Nabellen incomplete dossiers_COMPLETE',
'A_CANCELLED_COMPLETE', 'A_REGISTERED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'A_ACCEPTED_COMPLETE',
'O_CREATED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_ACTIVATED_COMPLETE', 'W_Nabellen incomplete dossiers_COMPLETE',
'A_ACCEPTED_COMPLETE', 'A_REGISTERED_COMPLETE', 'O_SENT_BACK_COMPLETE', 'O_CREATED_COMPLETE',
'W_Valideren aanvraag_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'W_Nabellen incomplete dossiers_COMPLETE',
'A_SUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE']
```

##### Resource
```python
'10138', '11299', '10779', '10809', '11002', '11202', '10861',
'10779', '11079', '10931', '11180', '11304', '10914', '10982',
'10629', '10929', '11003', '11019', '11003', '11003', '112', '10932',
'10138', '10138', '11304', '10912', '11201', '11269', '11202', '112',
'11189', '11179', '10228', '10609', '11304', '11002', '10910', '10931',
'11119', '11302', '11111', '11181', '11181', '10982', '10779', '11122',
'112', '10889', '10910', '11120', '10629'
```
##### Amount
```python
6185.0
```











