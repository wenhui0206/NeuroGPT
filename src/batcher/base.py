#!/usr/bin/env python3

from fileinput import filename
from operator import le
import pdb
from tracemalloc import start
from typing import Dict, Tuple, Generator
import numpy as np
from typing import Dict
# import webdataset as wds
import torch
import gzip
import pickle
import h5py
import webdataset as wds

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
    def __init__(self, filenames, sample_keys, chunk_len=500, num_chunks=10, tr=2.0, ovlp=50, root_path="", population_mean=0, population_std=1, gpt_only=False, normalization=True):
        if root_path == "":
            self.filenames = filenames
        else:
            self.filenames = [root_path + fn for fn in filenames]
            self.root_path = root_path
            
        print("Number of subjects loaded: ", len(self.filenames))
        # self.data = data_all
        self.chunk_len = chunk_len
        self.num_chunks = num_chunks
        self.tr = tr # the length of chunk in seconds, not real sampling rate of eeg.
        #this means the tr between chunks.
        self.ovlp = ovlp
        self.sample_keys = sample_keys
        self.mean = population_mean
        self.std = population_std
        self.do_normalization = normalization
        self.gpt_only=gpt_only

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        data = self.load_tensor(self.filenames[idx])
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

    def load_tensor(self, filename):
        tensor_fn = filename[:-3] + 'pt'
        tensor_data = torch.load(tensor_fn)
        return tensor_data.numpy()

    def split_chunks(self, data, length=500, ovlp=50, num_chunks=10): 
        '''2 seconds, 0.2 seconds overlap'''
        all_chunks = []
        total_len = data.shape[1]
        if num_chunks * length > total_len - 1:
            start_point = 0
            actual_num_chunks = total_len // length
        else:
            start_point = np.random.randint(0, total_len - num_chunks * length)
            actual_num_chunks = num_chunks
        
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
        t_r = self.tr
        if self.do_normalization:
            sample = self.normalize(sample)

        chunks, seq_on = self.split_chunks(sample, self.chunk_len, self.ovlp, seq_len)

        t_rs = np.arange(seq_len) * t_r
        attention_mask = np.ones(seq_len)
        chunks = self._pad_seq_right_to_n(
            seq=chunks,
            n=seq_len,
            pad_value=0
        )

        t_rs = self._pad_seq_right_to_n(
            seq=t_rs,
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
        out['t_rs'] = torch.from_numpy(t_rs).to(torch.float)
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
            # chunk_labels = np.ones(seq_len) * labels
            out['labels'] = torch.from_numpy(np.array(labels)).to(torch.long)
        # out['labels'] = np.zeros_like(t_rs)
        # pdb.set_trace()
        return out