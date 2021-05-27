from Utils.PrintUtils import print_big
import json
from Utils.FileUtils import file_exists
import os
from typing import List, Tuple
from Parameters.Enums import ActivityType, PreprocessedDfType
from datetime import timedelta
import pm4py
import pandas as pd
from Utils import VocabDict
from Utils import Constants
from CustomExceptions import NotSupportedError
import numpy as np
import tensorflow as tf


class XESDataset():
    pickle_df_file_name = "df.pickle"
    vocab_dict_file_name = "vocab_dict.json"

    def __init__(self, file_path: str, preprocessed_folder_path: str, preprocessed_df_type: PreprocessedDfType, include_types: List[ActivityType] = None) -> None:
        super().__init__()
        self.file_path = file_path
        self.preprocessed_folder_path = os.path.join(
            preprocessed_folder_path, XESDataset.get_type_folder_name(include_types))
        self.preprocessed_df_type = preprocessed_df_type

        if (not preprocessed_folder_path is None) and self.preprocessed_data_exist(self.preprocessed_folder_path, self.preprocessed_df_type):
            self.load_preprocessed_data()
        else:
            self.__initialise_data(
                file_path=file_path, include_types=include_types)

            if not preprocessed_folder_path is None:
                self.save_preprocessed_data()

    def __initialise_data(self, file_path: str, include_types: List[ActivityType]) -> None:
        '''
        run this function if the preprocessed data doesn't exist.
        [file_path]: path of `BPI_Challenge_2012.xes`
        [include_types]: what types of activity you want to load.
        '''
        ############ load xes file and extract needed information ############
        log = pm4py.read_xes(file_path)
        flattern_log: list[dict[str, any]] = ([{**event,
                                                'caseid': trace.attributes['concept:name']}
                                               for trace in log for event in trace])
        df = pd.DataFrame(flattern_log)

        if not (include_types is None):
            df = df[[any(bool_set) for bool_set in zip(
                *([df["concept:name"].str.startswith(a.value) for a in include_types]))]]

        df["name_and_transition"] = df["concept:name"] + \
            "_" + df["lifecycle:transition"]
        df = df[['time:timestamp', 'name_and_transition', "caseid"]]

        ############ Append starting and ending time for each trace ############
        newData = list()
        for case, group in df.groupby('caseid'):
            group.sort_values("time:timestamp", ascending=True, inplace=True)
            strating_time = group.iloc[0]["time:timestamp"] - \
                timedelta(microseconds=1)
            ending_time = group.iloc[-1]["time:timestamp"] + \
                timedelta(microseconds=1)
            traces = group.to_dict('records')

            # Add start and end tags.
            traces.insert(
                0, {"caseid": case, "time:timestamp": strating_time, "name_and_transition": Constants.SOS_VOCAB})
            traces.append(
                {"caseid": case, "time:timestamp": ending_time, "name_and_transition": Constants.EOS_VOCAB})
            newData.extend(traces)

        df = pd.DataFrame(newData)
        df['name_and_transition'] = df['name_and_transition'].astype(
            'category')

        ############ generate vocabulary dictionary ############
        vocab_dict: dict[str, int] = {}
        for i, cat in enumerate(df['name_and_transition'].cat.categories):
            # plus one, since we want to remain "0" for "<PAD>"
            vocab_dict[cat] = i+1
        vocab_dict[Constants.PAD_VOCAB] = 0

        ############ Create new index categorial column ############
        df['cat'] = df['name_and_transition'].apply(lambda c: vocab_dict[c])

        ############ Create the df only consisted of trace and caseid ############
        final_df_data: list[dict[str, any]] = []
        for caseid, group in df.groupby('caseid'):
            final_df_data.append({
                "trace": list(group['cat']),
                "caseid": caseid
            })

        ############ store data in instance ############
        self.df: pd.DataFrame = pd.DataFrame(final_df_data)
        self.df.sort_values("caseid", inplace=True)
        self.vocab = VocabDict(vocab_dict)

    def longest_trace_len(self) -> int:
        return self.df.trace.map(len).max()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> pd.Series:
        return self.df.iloc[index]

    @staticmethod
    def get_type_folder_name(include_types: List[ActivityType] = None):
        if include_types is None:
            return "All"

        return "".join(
            sorted([a.value for a in include_types], key=str.lower))

    @staticmethod
    def get_file_name_from_preprocessed_df_type(preprocessed_df_type: PreprocessedDfType):
        if preprocessed_df_type == PreprocessedDfType.Pickle:
            return XESDataset.pickle_df_file_name
        else:
            raise NotSupportedError(
                "Not supported saving format for preprocessed data")

    @staticmethod
    def preprocessed_data_exist(preprocessed_folder_path: str, preprocessed_df_type: PreprocessedDfType, ):
        file_name = XESDataset.get_file_name_from_preprocessed_df_type(
            preprocessed_df_type)
        df_path = os.path.join(preprocessed_folder_path,  file_name)
        vocab_dict_path = os.path.join(
            preprocessed_folder_path, XESDataset.vocab_dict_file_name)
        return file_exists(df_path) and file_exists(vocab_dict_path)

    def store_df(self, preprocessed_folder_path: str, preprocessed_df_type: PreprocessedDfType):
        os.makedirs(preprocessed_folder_path, exist_ok=True)
        file_name = XESDataset.get_file_name_from_preprocessed_df_type(
            preprocessed_df_type)
        df_path = os.path.join(preprocessed_folder_path, file_name)
        if(preprocessed_df_type == PreprocessedDfType.Pickle):
            self.store_df_in_pickle(df_path)
        else:
            raise NotSupportedError(
                "Not supported saving format for preprocessed data")

    def load_df(self, preprocessed_folder_path: str, preprocessed_df_type: PreprocessedDfType):
        file_name = XESDataset.get_file_name_from_preprocessed_df_type(
            preprocessed_df_type)
        df_path = os.path.join(preprocessed_folder_path, file_name)
        if(preprocessed_df_type == PreprocessedDfType.Pickle):
            self.load_df_from_pickle(df_path)
        else:
            raise NotSupportedError(
                "Not supported loading format for preprocessed data")

    def store_df_in_pickle(self, path):
        self.df.to_pickle(path)

    def load_df_from_pickle(self, path):
        self.df = pd.read_pickle(path)

    def save_preprocessed_data(self):
        if self.preprocessed_folder_path is None:
            raise Exception("Preprocessed folder path can't be None")

        ############ Store df ############
        self.store_df(self.preprocessed_folder_path,
                      self.preprocessed_df_type)

        ############ Store vocab_dict ############
        vocab_dict_path = os.path.join(
            self.preprocessed_folder_path, XESDataset.vocab_dict_file_name)
        with open(vocab_dict_path, 'w') as output_file:
            json.dump(self.vocab.vocab_dict, output_file, indent='\t')

        print_big(
            "Preprocessed data saved successfully"
        )

    def load_preprocessed_data(self):
        if self.preprocessed_folder_path is None:
            raise Exception("Preprocessed folder path can't be None")

        ############ Load df ############
        self.load_df(self.preprocessed_folder_path, self.preprocessed_df_type)

        ############ load vocab_dict ############
        vocab_dict_path = os.path.join(
            self.preprocessed_folder_path, XESDataset.vocab_dict_file_name)
        with open(vocab_dict_path, 'r') as output_file:
            vocab_dict = json.load(output_file)
            self.vocab = VocabDict(vocab_dict)

        print_big(
            "Preprocessed data loaded successfully: %s" % (
                self.preprocessed_folder_path)
        )

    def get_sampler_from_df(self, df, seed):
        return None

    def get_train_shuffle(self):
        return True

    def get_index_ds(self):
        return tf.data.Dataset.range(len(self.df))

    def collate_fn(self, idxs: List[int]) -> Tuple[np.ndarray, tf.Tensor, tf.Tensor]:
        batch_df = self.df.iloc[idxs]
        caseids = list(batch_df["caseid"])
        batch_traces = list(batch_df["trace"])
        data_traces = [t[:-1] for t in batch_traces]
        lengths = [len(t) for t in data_traces]
        target_traces = [t[1:] for t in batch_traces]
        padded_data_traces = tf.keras.preprocessing.sequence.pad_sequences(
            data_traces, padding='post', value=0)
        padded_target_traces = tf.keras.preprocessing.sequence.pad_sequences(
            target_traces, padding='post', value=0)

        return caseids, padded_data_traces, lengths, padded_target_traces
