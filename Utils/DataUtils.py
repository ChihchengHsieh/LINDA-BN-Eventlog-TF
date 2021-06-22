import numpy as np
def get_trace_with_id(df, id):
    return df[[id in t for t in df["trace"]]]

def get_longest_trace_row(df):
    max_len_idx = np.argmax([len(t) for t in df['trace']])
    max_len_row = df.iloc[max_len_idx: max_len_idx+1]
    return max_len_row

def remove_trail_steps(activities_2d, resources_2d, last_n_steps: int):
    example_idx_activities = np.array([activities_2d[0][:-last_n_steps]])
    example_idx_resources = np.array([resources_2d[0][:-last_n_steps]])
    return example_idx_activities, example_idx_resources
