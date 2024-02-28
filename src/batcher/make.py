#!/usr/bin/env python3

from batcher.base import BaseBatcher


def make_batcher(
    training_style: str='CSM',
    tr: float=2.0,
    chunk_len:int=500,
    num_chunks: int=10,
    seq_min: int=10,
    seq_max: int=50,
    bert_seq_gap_min: int=1,
    bert_seq_gap_max: int=5,
    decoding_target: str=None,
    sample_random_seq: bool=True,
    seed: int=None,
    bold_dummy_mode: bool=False,
    ) -> BaseBatcher:
    """
    Make a batcher object.
    
    The batcher is used to generate batches of 
    input data for training and evaluation.

    Args:
    -----
    training_style: str
        The used training style (ie., framework).
        One of: 'BERT', 'CSM', 'NetBERT', 'autoencoder',
        'decoding'.
    seq_min: int
        The minimum sequence length (in sequence elements)
        used for the random sampling of input sequences.
    seq_max: int
        The maximum sequence length (in sequence elements)
        used for the random sampling of input sequences.
    bert_seq_gap_min: int
        The minimum gap (in sequence elements) between
        two consecutive sequences for BERT-style training, 
        if they are sampled from the same data run file.
    bert_seq_gap_max: int
        The maximum gap (in sequence elements) between
        two consecutive sequences for BERT-style training, 
        if they are sampled from the same data run file.
    decoding_target: str
        Key of decoding target variable in data
        run files.
    sample_random_seq: bool
        If True, the sequences are sampled randomly from
        the data run files, given the spefied
        sequence length (seq_min and seq_max) and the
        specified gap consecutive sequences (bert_seq_gap_min,
        bert_seq_gap_max) for BERT-style training.
    seed: int
        The seed for the random number generator.
    bold_dummy_mode: bool
        If True, the BOLD data are replaced with simple
        dummy data (for internal testing purposed only).

    Core methods:
    -----
    dataset(tarfiles: list)
        Returns a Pytorch dataset that can be used for training, 
        given the specified list of data run file paths (tarfiles).
    """
    
    kwargs = {
        "tr": tr,
        "chunk_len": chunk_len,
        "num_chunks": num_chunks,
        "seq_min": seq_min,
        "seq_max": seq_max,
        "gap_min": bert_seq_gap_min,
        "gap_max": bert_seq_gap_max,
        "decoding_target": decoding_target,
        "sample_random_seq": sample_random_seq,
        "seed": seed,
        "bold_dummy_mode": bold_dummy_mode
    }
    sample_keys = [
        'inputs',
        'attention_mask',
        't_rs'
    ]

    if training_style in {'CSM', 'MSM', 'MNM', 'autoencoder'}:
        from batcher.base import BaseBatcher
        return BaseBatcher(**{**kwargs, **{'sample_keys': sample_keys}})

    elif training_style == 'decoding':
        sample_keys.append('labels')
        from batcher.base import BaseBatcher
        return BaseBatcher(**{**kwargs, **{'sample_keys': sample_keys}})

    else:
        raise ValueError('unknown training style.')