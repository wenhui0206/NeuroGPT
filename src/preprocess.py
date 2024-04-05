import argparse
import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import detrend
import torch

def preprocess_eeg(edf_file):
    # Load the EDF file
    raw = mne.io.read_raw_edf(edf_file, preload=True)
    # print(raw.info.ch_names)
    
    # Select the 22 channels based on the extended international 10-20 system
    channels = ['FP1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T1', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T2', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2']
    channels = [f'EEG {ch.upper()}-LE' for ch in channels]
    assert all(ch in raw.info['ch_names'] for ch in channels), f"Missing channels: {set(channels)-set(raw.info['ch_names'])}"
    raw.pick_channels(channels)

    # Identify bad channels (zero or missing signals)
    bad_channels = []
    for ch_name in channels:
        data, _ = raw[ch_name]
        if np.all(data == 0):
            bad_channels.append(ch_name)
    raw.info['bads'] = bad_channels

    # Interpolate bad channels (This is a simplified approach)
    if bad_channels:
        raw.interpolate_bads(reset_bads=True)

    # Re-reference the EEG signal to the average
    raw.set_eeg_reference(ref_channels='average')
    
    # Remove power line noise with notch filter and apply bandpass filter
    raw.notch_filter(60, notch_widths=1)
    raw.filter(0.5, 100, fir_design='firwin')

    # Resample to 250 Hz
    raw.resample(250)

    # DC offset correction and remove linear trends
    data = raw.get_data()
    data = detrend(data, type='constant')  # DC offset correction
    data = detrend(data, type='linear')    # Remove linear trends

    # Apply the z-transform along the time dimension
    data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
    raw._data = data

    return raw

def process_file(edf_file, export_path):
    print(f"Processing {edf_file}")
    try:
        preprocessed_data = preprocess_eeg(str(edf_file))
        data_tensor = torch.tensor(preprocessed_data.get_data())
        file_name = f"{edf_file.stem}_preprocessed.pt"
        torch.save(data_tensor, export_path / file_name)
        print(f"Saved {file_name} successfully.")
    except Exception as e:
        print(f"Error processing {edf_file}: {e}")

def process_subject(subject_path, export_path, filenames_to_process):
    for edf_file in subject_path.rglob('*.edf'):
        preprocessed_file_name = f"{edf_file.stem}_preprocessed.pt"
        if preprocessed_file_name in filenames_to_process:
            # print(f"Original file exists: {preprocessed_file_name}.")
            process_file(edf_file, export_path)
        # else:
        #     print(f"Original file does not exist: {preprocessed_file_name}.")

def process_and_save(data_root, export_dir, filename_csv):
    # filename format: tuh/tueg/edf/000/aaaaaaaa/s001_2015/01_tcp_ar/aaaaaaaa_s001_t000.edf
    # processed filename format: 000_01_tcp_ar_aaaaaaaa_s001_t000_preprocessed.pt
    data_root_path = Path(data_root)
    export_root_path = Path(export_dir)
    filenames_df = pd.read_csv(filename_csv)
    filenames_df['filename_suffix'] = filenames_df['filename'].apply(lambda x: "_".join(x.split('_')[4:]))
    subject_nums = filenames_df["filename"].apply(lambda x: x.split('_')[0]).unique().tolist()
    filenames_to_process = set(filenames_df['filename_suffix'].tolist())

    # print(subject_nums)
    # print(os.listdir(data_root_path))
    # exit()
    for subject_num in os.listdir(data_root_path):
        if subject_num not in subject_nums:
            continue
        subject_path = data_root_path / subject_num
        # Adapted pattern to match 'sNNN_YYYY'
        for session_dir in subject_path.rglob('s*_*'):
            if session_dir.is_dir():
                export_path = export_root_path / f"{subject_num}_{session_dir.stem}"
                export_path.mkdir(parents=True, exist_ok=True)
                process_subject(session_dir, export_path, filenames_to_process)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess EEG data.")
    parser.add_argument("--data-root", required=True, help="Root directory of the EEG data.")
    parser.add_argument("--export-dir", required=True, help="Directory where the preprocessed data will be saved.")
    parser.add_argument("--filename-csv", default="../inputs/sub_list2.csv", help="CSV file containing the list of filenames to process.")

    args = parser.parse_args()

    process_and_save(args.data_root, args.export_dir, args.filename_csv)
