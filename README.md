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

# Issue: Hard to train and get the counterfactual.
Since tensorflow using "argmax" in the embedding layer. This operation is not differentiable; therefore, the gradient can't propagate properly. To solve this issue, I get the weight (embedding matrix) from the embedding layer. And do matrix multiplication on it rather than 'argmax'. This apporach significantly improve the trainability of counterfactual.

------------------

# After A_PREACCEPTED_COMPLETE
 
## Input 

![image](https://user-images.githubusercontent.com/37566901/122894068-3bf82d80-d38a-11eb-88c6-c4a2eabb8a93.png)

## Model prediction
```python3 
Predicted activity with highest probability (0.36) is "W_Afhandelen leads_COMPLETE" 
```
![image](https://user-images.githubusercontent.com/37566901/122926035-c8b2e380-d3aa-11eb-83fb-01b077e49c82.png)

## Desired activity
```python3
A_DECLINED_COMPLETE
```

## Change AMOUNT only
Not found

## Change Resource only
![image](https://user-images.githubusercontent.com/37566901/122894210-56320b80-d38a-11eb-84cf-1fdc8ddef96b.png)


## Change Activity only
Not found

## Change Amount and Resource
![image](https://user-images.githubusercontent.com/37566901/122895965-f472a100-d38b-11eb-90a8-c23c01329b94.png)


## Change Amount and Activity
Not found

## Change Activity and Resource
![image](https://user-images.githubusercontent.com/37566901/122896865-d78a9d80-d38c-11eb-8ad6-6917d4fbd708.png)


## Change Activity, Resource and Amount
![image](https://user-images.githubusercontent.com/37566901/122897279-318b6300-d38d-11eb-801b-058b947a061c.png)


# After (A_ACCEPTED_COMPLETE)

## Model prediction
```python3 
Predicted activity with highest probability (0.55) is "O_SELECTED_COMPLETE" 
```
![image](https://user-images.githubusercontent.com/37566901/122926325-14658d00-d3ab-11eb-8872-f1fd6b2518fa.png)


## Desired activity
```python3
A_DECLINED_COMPLETE
```

## Input

### Activities
```python
['A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE',
'W_Afhandelen leads_COMPLETE', 'W_Completeren aanvraag_COMPLETE', 'A_ACCEPTED_COMPLETE']
```

### Resources
```python
['112', '112', '10863', '10863', '11169', '11003']
```

### Amount
```python
5800
```

## Change AMOUNT only
Not found

## Change Resource only

```python
[

# Activity
'A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'W_Afhandelen leads_COMPLETE', 'W_Completeren aanvraag_COMPLETE', 'A_ACCEPTED_COMPLETE',

# Resource
'10125', '10125', '11029', '11029', '11029', '11029',

# Amount
5800.0
]
```

## Change Activity only
Not found

## Change Amount and Resource
```python
[
# Activity
'A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'W_Afhandelen leads_COMPLETE', 'W_Completeren aanvraag_COMPLETE', 'A_ACCEPTED_COMPLETE',

# Resource
'10125', '10125', '11029', '11029', '11029', '11029',
 
# Amount 
5944.0
] 
```

## Change Amount and Activity
Not found

## Change Activity and Resource
```python
[
# Activity
 'A_SUBMITTED_COMPLETE', 'W_Beoordelen fraude_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'W_Beoordelen fraude_COMPLETE', 'O_SENT_COMPLETE', 'A_ACCEPTED_COMPLETE',
 
# Resource
'10125', '10125', '10863', '10863', '10125', '11029',

# Amount
 5800.0 
]
```

## Change Activity, Resource and Amount
```python
[
# Activity
'A_SUBMITTED_COMPLETE', 'W_Beoordelen fraude_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'W_Beoordelen fraude_COMPLETE', 'O_SENT_COMPLETE', 'A_ACCEPTED_COMPLETE',

# Resource
'10125', '10125', '10863', '10863', '10125', '11029',

# Amount
22.0 
 ] 
```

# After A_FINALIZED_COMPLETE

## Input
```python
[
# Activity
'A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'W_Afhandelen leads_COMPLETE', 'W_Completeren aanvraag_COMPLETE', 'A_ACCEPTED_COMPLETE', 'O_SELECTED_COMPLETE', 'A_FINALIZED_COMPLETE',

# Resource
'112', '112', '10863', '10863', '11169', '11003', '11003', '11003',

#Amount
5800.0
] 
```

## Model prediction
```python3 
Predicted activity with highest probability (0.51) is "O_CREATED_COMPLETE" 
```
![image](https://user-images.githubusercontent.com/37566901/122926479-3e1eb400-d3ab-11eb-9e58-adf48e509d04.png)

## Desired activity
```python3
A_DECLINED_COMPLETE
```

### Activities
```python
['A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'W_Afhandelen leads_COMPLETE', 'W_Completeren aanvraag_COMPLETE', 'A_ACCEPTED_COMPLETE', 'O_SELECTED_COMPLETE', 'A_FINALIZED_COMPLETE'] 
```
### Resources
```python
['112', '112', '10863', '10863', '11169', '11003', '11003', '11003']
```
### Amount
```python
[5800.0]
```

## Change AMOUNT only
Not found

## Change Resource only
Not found

## Change Activity only
```python
[
# Activity
'A_SUBMITTED_COMPLETE', 'A_SUBMITTED_COMPLETE', 'A_SUBMITTED_COMPLETE', 'A_SUBMITTED_COMPLETE', 'A_SUBMITTED_COMPLETE', 'A_SUBMITTED_COMPLETE', 'A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE',

# Resource
'112', '112', '10863', '10863', '11169', '11003', '11003', '11003',

# Amount
5800.0
] 
```

## Change Amount and Resource
Not found

## Change Amount and Activity
```python
[
# Activity
'A_SUBMITTED_COMPLETE', 'A_SUBMITTED_COMPLETE', 'A_SUBMITTED_COMPLETE', 'A_SUBMITTED_COMPLETE', 'A_SUBMITTED_COMPLETE', 'A_SUBMITTED_COMPLETE', 'A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE',

# Resource
'112', '112', '10863', '10863', '11169', '11003', '11003', '11003',

# Amount
3.0
] 
```

## Change Activity and Resource

```python
[
# Activity
'A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'W_Beoordelen fraude_COMPLETE', 'W_Completeren aanvraag_COMPLETE', 'W_Beoordelen fraude_COMPLETE', 'O_SELECTED_COMPLETE', 'O_SENT_COMPLETE',

# Resource
'10125', '10125', '10863', '10863', '10862', '11003', '11029', '11029',

# Amount
5800.0
]
```

## Change Activity, Resource and Amount

```python
[
# Activity
'A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'W_Beoordelen fraude_COMPLETE', 'W_Completeren aanvraag_COMPLETE', 'W_Beoordelen fraude_COMPLETE', 'O_SENT_COMPLETE', 'O_SENT_COMPLETE',

# Resource
'11304', '10125', '10863', '10863', '10862', '11003', '11029', '11029',

# Amount
3.0
] 
```


# Transformer result

### attention weights in each layer and head
![](https://github.com/ChihchengHsieh/LINDA-BN-Eventlog-TF/blob/master/layer_heads_attn.png?raw=true)
#### However, it can be hard to read, I calculate the reduce mean to get result below

### first step
![](https://github.com/ChihchengHsieh/LINDA-BN-Eventlog-TF/blob/master/next_step.png?raw=true)

### Last step
![](https://github.com/ChihchengHsieh/LINDA-BN-Eventlog-TF/blob/master/final_step.png?raw=true)

# Valid case classifier 
![image](https://user-images.githubusercontent.com/37566901/123313729-20994800-d56d-11eb-9716-46cee4487f7e.png)

# Models
## Our model:

![image](https://user-images.githubusercontent.com/37566901/123517938-1733eb00-d6e7-11eb-852e-a2701d50939a.png)


## How dice train
![image](https://user-images.githubusercontent.com/37566901/123517961-2e72d880-d6e7-11eb-950c-06df6ec7a84e.png)


## Proposed model
![image](https://user-images.githubusercontent.com/37566901/123517981-44809900-d6e7-11eb-854d-5e5b697d0dfb.png)


# [ScenarioCf] Problem encounter : the indentifier recognise some weird trace as a valid trace

for exmaple, the scenario classifier think trace below is valid:

```python
 ['<SOS>', 'A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE',
 'A_PREACCEPTED_COMPLETE', 'W_Afhandelen leads_COMPLETE',
 'W_Completeren aanvraag_COMPLETE', 'A_ACCEPTED_COMPLETE',
 'O_SELECTED_COMPLETE', 'A_FINALIZED_COMPLETE', 'O_CREATED_COMPLETE',
 'O_SENT_COMPLETE', 'W_Completeren aanvraag_COMPLETE', 'O_SELECTED_COMPLETE',
 'O_CANCELLED_COMPLETE', 'O_CREATED_COMPLETE', 'O_SENT_COMPLETE',
 'W_Nabellen offertes_COMPLETE', 'O_CANCELLED_COMPLETE', 'O_SELECTED_COMPLETE',
 'O_CREATED_COMPLETE', 'O_SENT_COMPLETE', 'W_Nabellen offertes_COMPLETE', 'W_Nabellen offertes_COMPLETE',
 'A_PREACCEPTED_COMPLETE', 'A_PREACCEPTED_COMPLETE', # "A_PREACCEPTED_COMPLETE" show up multiple times here, that doesn't make any sense.
 'A_PREACCEPTED_COMPLETE', 'A_PREACCEPTED_COMPLETE',
 'A_PREACCEPTED_COMPLETE','A_PREACCEPTED_COMPLETE', 'A_PREACCEPTED_COMPLETE',
 'W_Nabellen incomplete dossiers_COMPLETE', 'W_Nabellen incomplete dossiers_COMPLETE',
 'W_Nabellen incomplete dossiers_COMPLETE', 'W_Nabellen incomplete dossiers_COMPLETE'
 ]
```

## Potential solutions:
- [x] Introduce more fake data, and train the model to recognise them
- [ ] Add another strategy (Shuffle and repeat non-repeatible activity) for generating the fake dataset.
- [ ] Using another architecture.


## After giving more data to the scenario classifier Parameters:

### Parameters
```python
cf_out = dice.run_pls(
    ## Input
    example_amount_input.numpy(),
    example_idx_activities_no_tag,
    example_idx_resources_no_tag,
    desired_vocab = "A_DECLINED_COMPLETE",
    
    ## Weight
    class_loss_weight = 1,
    scenario_weight = 200,
    distance_loss_weight = 1e-8,
    cat_loss_weight = 1e-3,
    
    ## Training parameters
    scenario_threshold = 0.5,
    max_iter=200,
    lr=0.05,
    
    ## Options
    use_valid_cf_only=False,
    use_sampling=True,
    class_using_hinge_loss=False,
    scenario_using_hinge_loss=False,
    use_clipping=True, 
)
```

## Example 1

### Input
```python
====================Input Amount====================
| [5800.] 
====================================================

====================Input Activities====================
| ['A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'W_Afhandelen leads_COMPLETE', 'W_Completeren aanvraag_COMPLETE', 'A_ACCEPTED_COMPLETE', 'O_SELECTED_COMPLETE', 'A_FINALIZED_COMPLETE', 'O_CREATED_COMPLETE', 'O_SENT_COMPLETE', 'W_Completeren aanvraag_COMPLETE', 'O_SELECTED_COMPLETE', 'O_CANCELLED_COMPLETE'] 
========================================================

====================Input Resource====================
| ['112', '112', '10863', '10863', '11169', '11003', '11003', '11003', '11003', '11003', '11003', '11003', '11003'] 
======================================================

```

### Prediction
```python
====================Model Prediction====================
| Prediction: [W_Nabellen offertes_COMPLETE(24)] | Desired: [A_DECLINED_COMPLETE(7)] 
================================================

====================Counterfactual Process====================
| [0] ==========> [1] 
==============================================================
```

### CF found
```python

====================Valid CF Amount====================
| 5799.1533 
=======================================================


# A_SUBMITTED_COMPLETE appear twice 
====================Valid CF Activities====================
| ['A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'W_Afhandelen leads_COMPLETE', 'W_Completeren aanvraag_COMPLETE', 'A_ACCEPTED_COMPLETE', 'O_SELECTED_COMPLETE', 'A_SUBMITTED_COMPLETE', 'O_CREATED_COMPLETE', 'O_SENT_COMPLETE', 'W_Completeren aanvraag_COMPLETE', 'O_SELECTED_COMPLETE', 'O_CANCELLED_COMPLETE'] 
===========================================================

====================Valid CF Resource====================
| ['112', '112', '10863', '10863', '11169', 'UNKNOWN', '11003', '11003', '11003', 'UNKNOWN', '11003', '10138', '10188'] 
=========================================================

====================Valid CF scenario output==================== 
| [0.8 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. ] ### [1] => valid scenario in this step. [0] => invalid. (Start from <SOS>)
================================================================
```

## Example 2:


```python

====================Model Prediction====================
| Prediction: [<EOS>(1)] | Desired: [A_DECLINED_COMPLETE(7)] 
========================================================

====================Counterfactual Process====================
| [0] ==========> [1] 
==============================================================

====================!Counterfactual Found in step [19]!====================
| Running time: 1.95 
===========================================================================

====================Input Amount====================
| [15500.] 
====================================================

====================Input Activities====================
| ['A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE'] 
========================================================

====================Input Resource====================
| ['112', '112', '112'] 
======================================================

====================Valid CF Amount====================
| 15499.054 
=======================================================

====================Valid CF Activities====================
| ['A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE'] 
===========================================================

====================Valid CF Resource====================
| ['112', 'UNKNOWN', '112'] 
=========================================================

====================Valid CF scenario output====================
| [0.7 1.  0.9 1. ] 
================================================================

```

## After applying other strategy:

