#!/usr/bin/env python3
from typing import Dict
import numpy as np
# import webdataset as wds
import torch
# import gzip
# import pickle
import h5py
import os
# import webdataset as wds

from torch.utils.data import Dataset

def _pad_seq_right_to_n(
    seq: np.ndarray,
    n: int,
    pad_value: float = 0.
    ) -> np.ndarray:
    if n == seq.shape[0]:
        return seq
    return np.concatenate(
        [
            seq,
            np.ones(
                (
                    n-seq.shape[0],
                    *seq.shape[1:]
                )
            ) * pad_value,  
        ],
        axis=0,
    )

class EEGDataset(Dataset):
    def __init__(self, filenames, sample_keys, chunk_len=500, num_chunks=10, ovlp=50, root_path="", population_mean=0, population_std=1, gpt_only=False, normalization=True, start_samp_pnt=-1):
        if root_path == "":
            self.filenames = filenames
        else:
            self.filenames = [root_path + fn for fn in filenames if os.path.isfile(root_path+fn)]
            self.root_path = root_path
            
        print("Number of subjects loaded: ", len(self.filenames))
        # self.data = data_all
        self.chunk_len = chunk_len
        self.num_chunks = num_chunks
        self.ovlp = ovlp
        self.sample_keys = sample_keys
        self.mean = population_mean
        self.std = population_std
        self.do_normalization = normalization
        self.gpt_only=gpt_only
        self.start_samp_pnt = start_samp_pnt

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        data = self.load_tensor(self.filenames[idx])
        #===reorder channels====
        data = self.reorder_channels(data)
        return self.preprocess_sample(data, seq_len=self.num_chunks)

    @staticmethod
    def _pad_seq_right_to_n(
        seq: np.ndarray,
        n: int,
        pad_value: float = 0
        ) -> np.ndarray:
        return _pad_seq_right_to_n(
            seq=seq,
            n=n,
            pad_value=pad_value
        )

    def load_single_file(self, filename):
        with h5py.File(filename, 'r') as file:
            data_dict = file['Result']
            data = []
            for i in range(data_dict['data'].shape[0]):  
                ref = data_dict['data'][i][0]
                time_series = data_dict[ref]
                if len(data) > 0 and time_series.shape[0] < data[0].shape[0]:
                    time_series = np.zeros_like(data[0])
                data.append(np.array(time_series).squeeze())
        return data

    def load_tensor(self, filename):
        # tensor_fn = filename[:-3] + 'pt'
        tensor_data = torch.load(filename)
        return tensor_data.numpy()

    def reorder_channels(self, data):
        chann_labels = {'FP1': 0, 'FP2': 1, 'F3': 2, 'F4': 3, 'C3': 4, 'C4': 5, 'P3': 6, 'P4': 7, 'O1': 8, 'O2': 9, 'F7': 10, 'F8': 11, 'T3': 12, 'T4': 13, 'T5': 14, 'T6': 15, 'FZ': 16, 'CZ': 17, 'PZ': 18, 'OZ': 19, 'T1': 20, 'T2': 21}
        reorder_labels = {'FP1': 0, 'FP2': 1, 'F7': 2, 'F3': 3, 'FZ': 4, 'F4': 5, 'F8': 6, 'T1': 7, 'T3': 8, 'C3': 9, 'CZ': 10, 'C4': 11, 'T4': 12, 'T2': 13, 'T5': 14, 'P3': 15, 'PZ': 16, 'P4': 17, 'T6': 18, 'O1': 19, 'OZ': 20, 'O2': 21}

        reordered = np.zeros_like(data)
        for label, target_idx in reorder_labels.items():
            mapped_idx = chann_labels[label]
            reordered[target_idx, :] = data[mapped_idx, :]
        
        return reordered

    def split_chunks(self, data, length=500, ovlp=50, num_chunks=10, start_point=-1): 
        '''2 seconds, 0.2 seconds overlap'''
        all_chunks = []
        total_len = data.shape[1]
        actual_num_chunks = num_chunks
        
        if start_point == -1:
            if num_chunks * length > total_len - 1:
                start_point = 0
                actual_num_chunks = total_len // length
            else:
                start_point = np.random.randint(0, total_len - num_chunks * length)
        
        for i in range(actual_num_chunks):
            chunk = data[:, start_point: start_point + length]
            all_chunks.append(np.array(chunk))
            start_point = start_point + length - ovlp
        return np.array(all_chunks), start_point
    
    def normalize(self, data):
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        # Ensure std is not zero to avoid division by zero.
        # If std is zero, normalization doesn't make sense, 
        # so you might set std to a small positive value or handle it in another way.
        # std = np.where(std == 0, 1e-23, std)
        return (data - mean) / (std + 1e-25)

    def preprocess_sample(
        self,
        sample,
        seq_len,
        labels=None
        ) -> Dict[str, torch.Tensor]:
        out = {}
        if self.do_normalization:
            sample = self.normalize(sample)

        chunks, seq_on = self.split_chunks(sample, self.chunk_len, self.ovlp, seq_len, self.start_samp_pnt)

        attention_mask = np.ones(seq_len)
        chunks = self._pad_seq_right_to_n(
            seq=chunks,
            n=seq_len,
            pad_value=0
        )

        attention_mask = self._pad_seq_right_to_n(
            seq=attention_mask, 
            n=seq_len,
            pad_value=0
        )
        
        if self.gpt_only == True:
            chunks = np.reshape(chunks, (seq_len, chunks.shape[1]*chunks.shape[2]))
        out["inputs"] = torch.from_numpy(chunks).to(torch.float)
        out["attention_mask"] = torch.from_numpy(attention_mask).to(torch.long)
        out['seq_on'] = seq_on
        out['seq_len'] = seq_len
        
        if self.sample_keys is not None:
            out = {
                key: out[key] 
                for key in self.sample_keys
                if key in out
            }

        if labels is not None:
            out['labels'] = torch.from_numpy(np.array(labels)).to(torch.long)
   
        return out