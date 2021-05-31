import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List


def tensor_generate_permutation_for_numerical(input_data: tf.Tensor, num_samples_per_feature, variance=0.5, ):
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
        input_to_permute = tf.repeat(
            input_to_permute, num_samples_per_feature, axis=0)
        input_to_permute = input_to_permute.numpy()
        input_to_permute[:, i] = tf.random.uniform(
            (num_samples_per_feature,), minval=min_range[i], maxval=max_range[i]).numpy()
        input_to_permute = tf.constant(input_to_permute)
        all_permutations.extend(
            list(tf.split(input_to_permute, num_samples_per_feature, axis=0)))

    ########## append the original data ##########
    all_permutations.append(input_backup)

    return all_permutations


def tensor_generate_permutation_for_numerical_all_dim(input_data: tf.Tensor, num_samples, variance=0.1, clip_permutation: bool = True):
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


def tensor_generate_permutations_for_normerical_all_dim_normal_dist(input_data: tf.Tensor, num_samples, variance=0.1):
    return tf.random.normal((num_samples, input_data.shape[-1]), mean=input_data, stddev=np.sqrt(variance))


def tensor_generate_fix_step_permutation_for_finding_boundary(input_data: tf.Tensor, variance=0.1, steps=10):
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


def plot_permutations_in_3d(input_data: np.array, variance: float, permutations: List[np.array], space_times=1):
    assert(len(permutations.shape) == 2 and permutations.shape[-1] == 3)
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection='3d')
    space = variance * space_times
    ax.set_xlim3d(input_data[0]-space, input_data[0]+space)
    ax.set_ylim3d(input_data[1]-space, input_data[1]+space)
    ax.set_zlim3d(input_data[2]-space, input_data[2]+space)
    ax.scatter(permutations[:, 0], permutations[:, 1],
               permutations[:, 2], marker="o", c='#5582ab', s=20)
    ax.scatter(input_data[0], input_data[1],
               input_data[2], marker="o", c="#ff6b6b", s=2000)


def generate_permutation_for_numerical_in_single_feature_uniform(input_data: np.array, sample_size: int, variance: float = 0.5, clip: bool = False):
    '''
    [input_data]: Normalised data. should be a 1-D array.
    --------------------------
    Return: all permutations.
    '''

    all_permutations = []

    input_backup = np.expand_dims(input_data, axis=0).copy()

    if clip:
        max_range = np.clip(input_data + variance, -0.999999, 1)
        min_range = np.clip(input_data - variance, -1, 0.999999)
    else:
        max_range = input_data + variance
        min_range = input_data - variance

    for i in range(input_data.shape[-1]):
        input_to_permute = input_backup.copy()
        input_to_permute = np.repeat(input_to_permute, sample_size, axis=0)
        input_to_permute[:, i] = np.random.uniform(
            min_range[i], max_range[i], size=sample_size)
        all_permutations.extend(
            list(np.split(input_to_permute, sample_size, axis=0)))

    return np.concatenate(all_permutations, axis=0)

def generate_permutation_for_numerical_cube_all_dim_unifrom(input_data: np.array, sample_size: int, variance: float = 0.1, clip: bool = True):
    '''
    [input_data]: Normalised data. should be a 1-D tensor.
    --------------------------
    Return: all permutations.
    '''
    if clip:
        max_range = np.clip(input_data + variance, -0.999999, 1)
        min_range = np.clip(input_data - variance, -1, 0.999999)
    else:
        max_range = input_data + variance
        min_range = input_data - variance

    return np.random.uniform(min_range, max_range, (sample_size,  input_data.shape[-1]))


def generate_permutations_for_numerical_cube_all_dim_normal(input_data: np.array, sample_size, variance=0.1):
    return np.random.normal(size=(sample_size, input_data.shape[-1]), loc=input_data, scale=np.sqrt(variance))


def generate_fix_step_permutation_for_numerical(input_data: np.array, variance=0.1, steps=10):
    '''
    [input_data]: Normalised data. should be a 1-D tensor.
    --------------------------
    Return: all permutations.
    '''
    input_backup = input_data.copy()
    all_permutations = []
    for s in range(input_data.shape[-1]):
        for i in range(steps):
            distance = (variance * (i + 1))
            plus_data = input_backup.copy()
            plus_data[s] = plus_data[s] + distance
            all_permutations.append(plus_data)
            minus_data = input_backup.copy()
            minus_data[s] = minus_data[s] - distance
            all_permutations.append(minus_data)
    return np.stack(all_permutations, axis=0)


def generate_permutations_for_numerical_n_ball_uniform(input_data: np.array, sample_size: int, variance: float = 0.1):
    '''
    [input_data]: Normalised data. should be a 1-D tensor.
    Algorithm from: http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf

    --------------------------
    Return: all permutations.
    '''
    dim = input_data.shape[-1]
    # an array of (d+2) normally distributed random variables
    u = np.random.normal(0, 1, (sample_size, dim+2))
    norm = np.expand_dims(np.sum(u**2, axis=1) ** (0.5), axis=1)
    u = u/norm
    x = u[:, 0:dim]
    return np.repeat(np.expand_dims(input_data, axis=0), sample_size, axis=0) + (x * variance)
