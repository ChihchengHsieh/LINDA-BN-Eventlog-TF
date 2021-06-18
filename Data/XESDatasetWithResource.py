from tensorflow.python.keras.preprocessing.sequence import pad_sequences
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
import math


class XESDatasetWithResource():
    pickle_df_file_name = "df.pickle"
    vocabs_file_name = "vocabs.json"
    resources_file_name = "resources.json"

    def __init__(self, file_path: str, preprocessed_folder_path: str, preprocessed_df_type: PreprocessedDfType, include_types: List[ActivityType] = None) -> None:
        super().__init__()
        self.file_path = file_path
        self.preprocessed_folder_path = os.path.join(
            preprocessed_folder_path, XESDatasetWithResource.get_type_folder_name(include_types))
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
        flattern_log = ([{**event,
                          'caseid': trace.attributes['concept:name'],
                          'amount': trace.attributes['AMOUNT_REQ'],
                          }

                         for trace in log for event in trace])
        df = pd.DataFrame(flattern_log)
        df = df[df["lifecycle:transition"] == "COMPLETE"]
        df["org:resource"] = [
            'UNKNOWN' if math.isnan(float(r)) else r for r in df["org:resource"]]
        # df["org:resource"] = df["org:resource"].astype('category')
        # self.resources = list(df["org:resource"].cat.categories)
        # self.resources =  list(df["org:resource"].cat.categories.insert(0, "<PAD>"))
        # df['resource'] = df["org:resource"].apply(lambda a: self.resources.index(a))
        # self.resources.append("<SOS>")  # for tokens
        # self.resources.append("<EOS>")

        if not (include_types is None):
            df = df[[any(bool_set) for bool_set in zip(
                *([df["concept:name"].str.startswith(a.value) for a in include_types]))]]

        df["activity"] = df["concept:name"] + \
            "_" + df["lifecycle:transition"]

        ############ Append starting and ending time for each trace ############
        newData = []
        for case, group in df.groupby('caseid'):
            group.sort_values("time:timestamp", ascending=True, inplace=True)
            strating_time = group.iloc[0]["time:timestamp"] - \
                timedelta(microseconds=1)
            ending_time = group.iloc[-1]["time:timestamp"] + \
                timedelta(microseconds=1)
            amount = group.iloc[-1]["amount"]
            traces = group.to_dict('records')

            # Add start and end tags.
            traces.insert(
                0, {"caseid": case,
                    "time:timestamp": strating_time,
                    "activity": Constants.SOS_VOCAB,
                    "amount": amount,
                    "org:resource": Constants.SOS_VOCAB,
                    })
            traces.append(
                {"caseid": case,
                 "time:timestamp": ending_time,
                 "activity": Constants.EOS_VOCAB,
                 "amount": amount,
                 "org:resource": Constants.EOS_VOCAB,
                 })
            newData.extend(traces)

        df = pd.DataFrame(newData)
        df['activity'] = df['activity'].astype(
            'category')
        
        df['org:resource'] = df['org:resource'].astype('category')

        ############ generate vocabulary list ############
        vocabs = [Constants.PAD_VOCAB] + \
            list(df['activity'].cat.categories)

        self.vocab = VocabDict(vocabs)
        self.resources = [Constants.PAD_VOCAB] + list(df["org:resource"].cat.categories)

        ############ Create new index categorial column ############
        df['activity_idx'] = df['activity'].apply(
            lambda c: self.vocab.vocab_to_index(c))

        df['resource'] = df["org:resource"].apply(lambda a: self.resources.index(a))


        ############ Create the df only consisted of trace and caseid ############
        final_df_data = []
        for caseid, group in df.groupby('caseid'):
            final_df_data.append({
                "trace": list(group['activity_idx']),
                "trace_vocab": list(group['activity']),
                "caseid": caseid,
                "amount": float(list(group['amount'])[0]),
                "resource": list(group['resource']),
                "resource_orig": list(group['org:resource'])
            })

        ############ store data in instance ############
        self.df: pd.DataFrame = pd.DataFrame(final_df_data)
        self.df.sort_values("caseid", inplace=True)

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
            return XESDatasetWithResource.pickle_df_file_name
        else:
            raise NotSupportedError(
                "Not supported saving format for preprocessed data")

    @staticmethod
    def preprocessed_data_exist(preprocessed_folder_path: str, preprocessed_df_type: PreprocessedDfType, ):
        file_name = XESDatasetWithResource.get_file_name_from_preprocessed_df_type(
            preprocessed_df_type)
        df_path = os.path.join(preprocessed_folder_path,  file_name)
        vocab_dict_path = os.path.join(
            preprocessed_folder_path, XESDatasetWithResource.vocabs_file_name)
        return file_exists(df_path) and file_exists(vocab_dict_path)

    def store_df(self, preprocessed_folder_path: str, preprocessed_df_type: PreprocessedDfType):
        os.makedirs(preprocessed_folder_path, exist_ok=True)
        file_name = XESDatasetWithResource.get_file_name_from_preprocessed_df_type(
            preprocessed_df_type)
        df_path = os.path.join(preprocessed_folder_path, file_name)
        if(preprocessed_df_type == PreprocessedDfType.Pickle):
            self.store_df_in_pickle(df_path)
        else:
            raise NotSupportedError(
                "Not supported saving format for preprocessed data")

    def load_df(self, preprocessed_folder_path: str, preprocessed_df_type: PreprocessedDfType):
        file_name = XESDatasetWithResource.get_file_name_from_preprocessed_df_type(
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
        vocabs_path = os.path.join(
            self.preprocessed_folder_path, XESDatasetWithResource.vocabs_file_name)
        with open(vocabs_path, 'w') as output_file:
            json.dump(self.vocab.vocabs, output_file, indent='\t')

        ############ Store resources ############
        resources_path = os.path.join(
            self.preprocessed_folder_path, XESDatasetWithResource.resources_file_name)
        with open(resources_path, 'w') as output_file:
            json.dump(self.resources, output_file, indent='\t')


        print_big(
            "Preprocessed data saved successfully"
        )

    def load_preprocessed_data(self):
        if self.preprocessed_folder_path is None:
            raise Exception("Preprocessed folder path can't be None")

        ############ Load df ############
        self.load_df(self.preprocessed_folder_path, self.preprocessed_df_type)

        ############ load vocab_dict ############
        vocabs_path = os.path.join(
            self.preprocessed_folder_path, XESDatasetWithResource.vocabs_file_name)
        with open(vocabs_path, 'r') as output_file:
            vocabs = json.load(output_file)
            self.vocab = VocabDict(vocabs)

        ############ load resources ############
        resources_path = os.path.join(
            self.preprocessed_folder_path, XESDatasetWithResource.resources_file_name)
        with open(resources_path, 'r') as output_file:
            self.resources = json.load(output_file)

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

    def collate_fn(self, idxs: List[int]):
        '''
        Return: [caseids, padded_data_traces, lengths, batch_resources, batch_amount, padded_target_traces]
        '''
        batch_df = self.df.iloc[idxs]
        caseids = list(batch_df["caseid"])
        batch_traces = list(batch_df["trace"])
        batch_resources = list(batch_df["resource"])
        batch_amount = list(batch_df["amount"])
        data_traces = [t[:-1] for t in batch_traces]
        data_resources = [r[:-1] for r in batch_resources]
        lengths = [len(t) for t in data_traces]
        target_traces = [t[1:] for t in batch_traces]
        padded_data_traces = tf.keras.preprocessing.sequence.pad_sequences(
            data_traces, padding='post', value=0) # return will be numpy
        padded_target_traces = tf.keras.preprocessing.sequence.pad_sequences(
            target_traces, padding='post', value=0) # return will be numpy

        padded_data_resources = tf.keras.preprocessing.sequence.pad_sequences(
            data_resources, padding='post', value=0)

        return caseids, padded_data_traces, lengths, padded_data_resources, batch_amount, padded_target_traces
