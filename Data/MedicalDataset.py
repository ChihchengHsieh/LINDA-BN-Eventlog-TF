from typing import List, Tuple
import pandas as pd
import numpy as np
import tensorflow as tf


class MedicalDataset():
    def __init__(self, file_path: str, target_col_name: str, feature_names: List[str]):
        self.df = pd.read_csv(file_path)
        self.target_col_name = target_col_name
        if (feature_names is None):
            self.feature_names =[ col  for col in self.df.columns if col != self.target_col_name ]
        else:
            self.feature_names = feature_names

        #### Balance dataset ####
        count_dict = dict(self.df[self.target_col_name].value_counts())
        max_count = max(count_dict.values())
        need_dict = {}
        for k in  count_dict.keys():
            need_dict[k] = max_count - count_dict[k]

        for k in need_dict.keys():
            if (need_dict[k] > 0):
                group_idxs = list(self.df[self.df[self.target_col_name] == k].index)
                append_idxs = np.random.choice(group_idxs, size=need_dict[k])
                self.df = self.df.append(self.df.iloc[append_idxs], ignore_index=True)


    def __len__(self) -> int:
        return len(self.df)

    def num_features(self) -> int:
        return len(self.feature_names)

    def get_index_ds(self):
        return tf.data.Dataset.range(len(self.df))

    def collate_fn(self, idxs: List[int]) -> Tuple[tf.Tensor, tf.Tensor]:
        # Transform to df
        input_df = self.df.iloc[idxs]
        input_data = input_df[self.feature_names]
        input_target = input_df[self.target_col_name]

        return tf.constant(np.array(input_data), dtype=tf.float32), tf.constant(np.array(input_target), dtype=tf.float32)

    def get_sampler_from_df(self, df, seed: int):
        return None
        
    def get_train_shuffle(self):
        return False
