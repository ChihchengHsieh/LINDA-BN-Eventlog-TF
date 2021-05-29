import numpy as np
import pyAgrum as gum
import tensorflow as tf

def generate_permutation_for_numerical(input_data: tf.Tensor, num_samples_per_feature, variance=0.5, ):
    '''
    [input_data]: Normalised data. should be a 1-D tensor.
    --------------------------
    Return: all permutations.
    '''

    all_permutations = []

    input_backup = tf.identity(input_data)

    max_range = tf.clip_by_value(input_data + variance, - 1, 1)[0]
    min_range = tf.clip_by_value(input_data - variance, -1, 1)[0]

    for i in range(input_data.shape[-1]):
        input_to_permute = tf.identity(input_backup)
        input_to_permute = tf.repeat(input_to_permute, num_samples_per_feature, axis=0)
        input_to_permute = input_to_permute.numpy()
        input_to_permute[:, i] = tf.random.uniform(
            (num_samples_per_feature,), minval=min_range[i], maxval=max_range[i]).numpy()
        input_to_permute = tf.constant(input_to_permute)
        all_permutations.extend(
            list(tf.split(input_to_permute, num_samples_per_feature, axis=0)))

    ########## append the original data ##########
    all_permutations.append(input_backup)

    return all_permutations


def generate_permutation_for_numerical_all_dim(input_data: tf.Tensor, num_samples, variance=0.1, clip_permutation: bool = True):
    '''
    [input_data]: Normalised data. should be a 1-D tensor.
    --------------------------
    Return: all permutations.
    '''
    if clip_permutation:
        max_range = tf.clip_by_value(input_data + variance, -0.999999, 1)
        min_range = tf.clip_by_value(input_data - variance, -1, 0.999999)
    else:
        max_range = input_data + variance
        min_range = input_data - variance

    return tf.random.uniform((num_samples,  input_data.shape[-1]), minval=min_range.numpy(), maxval=max_range.numpy())


def generate_permutations_for_normerical_all_dim_normal_dist(input_data: tf.Tensor, num_samples, variance=0.1):
    return tf.random.normal((num_samples, input_data.shape[-1]), mean=input_data, stddev=np.sqrt(variance))


def generate_fix_step_permutation_for_finding_boundary(input_data: tf.Tensor, variance=0.1, steps=10):
    input_data = input_data.numpy().squeeze()
    all_permutations = []
    for s in range(input_data.shape[0]):
        for i in range(steps):
            distance = (variance * (i + 1))
            plus_data = input_data.copy()
            plus_data[s] = plus_data[s] + distance
            all_permutations.append(plus_data)
            minus_data = input_data.copy()
            minus_data[s] = minus_data[s] - distance
            all_permutations.append(minus_data)
    return [tf.constant(p) for p in all_permutations]


def generate_permutation_for_trace(trace: np.array, vocab_size: int, last_n_stages_to_permute: int = None):
    # For each stage (activity), we replace it by another.
    # But we still maintain the same length of the trace.
    '''
    Basically, we see each activity in the trace as a feature.
    'Stage1, Stage2, Stage 3, Stage4 ....  Destination'

    Permute on last few stages
    '''

    all_permutations = []

    if (last_n_stages_to_permute is None):
        last_n_stages_to_permute = len(trace)

    trace_backup = trace.copy()

    max_index = len(trace) - 1

    all_permutations.append(trace.tolist())

    for idx_last in range(last_n_stages_to_permute):
        idx = max_index - idx_last
        trace_to_permute = trace_backup.copy()
        all_permutations.extend([replaceIndex(trace=trace_to_permute, index=idx, value=v_i).tolist(
        ) for v_i in range(vocab_size) if v_i != trace[idx]])

    return all_permutations


def replaceIndex(trace: np.array, index: int, value: int) -> np.array:
    trace[index] = value
    return trace


def exploring():
    gum.MarkovBlanket
    pass
