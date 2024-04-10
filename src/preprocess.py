import argparse
from glob import glob
import logging
import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import detrend
import torch

def generate_normalized_names(channel_names):
    """
    Generate a dictionary to map original EEG channel names to their normalized
    names based on the standard 10-20 system naming convention, ensuring
    proper case (e.g., 'Fp1' instead of 'FP1').

    Parameters:
    - channel_names: list of str, original channel names from the EEG file.

    Returns:
    - dict: A dictionary where keys are original channel names and values are
            the corresponding normalized names in proper case.
    """
    prefix_removal = "EEG "
    suffix_removals = ["-REF", "-LE"]
    
    # Mapping of upper case to proper case for the 10-20 system
    proper_case_mapping = {
        'FP1': 'Fp1', 'FP2': 'Fp2',
        'F7': 'F7', 'F3': 'F3', 'FZ': 'Fz', 'F4': 'F4', 'F8': 'F8',
        'T1': 'T1', 'T3': 'T3', 'C3': 'C3', 'CZ': 'Cz', 'C4': 'C4', 'T4': 'T4', 'T2': 'T2',
        'T5': 'T5', 'P3': 'P3', 'PZ': 'Pz', 'P4': 'P4', 'T6': 'T6',
        'O1': 'O1', 'OZ': 'Oz', 'O2': 'O2',
    }
    
    normalized_names = {}
    for name in channel_names:
        name_short = name
        # Remove the 'EEG ' prefix and any '-REF' or '-LE' suffix
        for suffix in suffix_removals:
            name_short = name_short.replace(suffix, '')
        name_short = name_short.replace(prefix_removal, '')
        
        # Convert to proper case based on the 10-20 system
        normalized_name = proper_case_mapping.get(name_short.upper())
        
        # Map the original name to the normalized name
        normalized_names[name] = normalized_name
        
    return normalized_names

def preprocess_eeg(edf_file, logger, channels = ['FP1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T1', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T2', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2']):
    # Load the EDF file
    raw = mne.io.read_raw_edf(edf_file, preload=True)
    # raw = mne.io.read_raw_edf(edf_file, preload=False, verbose='error')
    
    # Select the 22 channels based on the extended international 10-20 system
    ch_suffixes = [ch.split("-")[-1] for ch in raw.info['ch_names'] if ch.endswith("LE") or ch.endswith("REF")]
    assert len(set(ch_suffixes))==1, f"Multiple channel type detected: {set(ch_suffixes)}"
    channels_formatted = [f'EEG {ch.upper()}-{ch_suffixes[0]}' for ch in channels]
    missing_channels = list(set(channels_formatted)-set(raw.info['ch_names']))
    if len(missing_channels):
        logger.info(f"Missing channels {missing_channels} in available channels: \n{raw.info['ch_names']}")

    if missing_channels:
        logger.info(f"Available channels: \n{raw.info['ch_names']}\nAdding missing channels {missing_channels} with zero values...")
        for ch_name in missing_channels:
            # Create a data array of zeros
            data = np.zeros((1, len(raw.times)))
            # Create an Info object for the new channel
            ch_info = mne.create_info(ch_names=[ch_name], sfreq=raw.info['sfreq'], ch_types='eeg')
            # Create a RawArray and append to the existing Raw object
            missing_raw = mne.io.RawArray(data, ch_info)
            raw.add_channels([missing_raw], force_update_info=True)
        
        # Mark the newly added channels as bad
        raw.info['bads'].extend(missing_channels)

    # Ensure the specified channels are in the correct order
    raw.reorder_channels(channels_formatted)

    logger.info("Selecting the 22 channels...")
    raw.pick_channels(channels_formatted, ordered=False)

    # Identify bad channels (zero or missing signals)
    logger.info("Identifying bad channels...")
    bad_channels = []
    for ch_name in channels_formatted:
        data, _ = raw[ch_name]
        if np.all(data == 0):
            bad_channels.append(ch_name)

    raw.info['bads'] = bad_channels

    # Rename channels in the raw object
    normalized_names = generate_normalized_names(raw.info['ch_names'])
    raw.rename_channels(normalized_names)
    # Set montage (assuming 10-20 system)
    montage = mne.channels.make_standard_montage('standard_1020')
    # remove channels not present in the 10-20 system

    # logger.info(f"Channels in montage: {sorted(montage.ch_names)}")

    drop_channels = [ch for ch in raw.info['ch_names'] if ch not in montage.ch_names]
    logger.info(f"Dropping channels: {drop_channels}")

    # proper_case_mapping = {
    #     'FP1': 'Fp1', 'FP2': 'Fp2',
    #     'F7': 'F7', 'F3': 'F3', 'FZ': 'Fz', 'F4': 'F4', 'F8': 'F8',
    #     'T1': 'T1', 'T3': 'T3', 'C3': 'C3', 'CZ': 'Cz', 'C4': 'C4', 'T4': 'T4', 'T2': 'T2',
    #     'T5': 'T5', 'P3': 'P3', 'PZ': 'Pz', 'P4': 'P4', 'T6': 'T6',
    #     'O1': 'O1', 'OZ': 'Oz', 'O2': 'O2',
    # }
    # logger.info(f"Channels in proper case but not in 10-20 system: {set([ch for ch in proper_case_mapping.values() if ch not in montage.ch_names])}")
    # chann_labels = {'Fp1': 0, 'Fp2': 1, 'F3': 2, 'F4': 3, 'C3': 4, 'C4': 5, 'P3': 6, 'P4': 7, 'O1': 8, 'O2': 9, 'F7': 10, 'F8': 11, 'T3': 12, 'T4': 13, 'T5': 14, 'T6': 15, 'Fz': 16, 'Cz': 17, 'Pz': 18, 'Oz': 19, 'T1': 20, 'T2': 21}
    # reorder_labels = {'Fp1': 0, 'Fp2': 1, 'F7': 2, 'F3': 3, 'Fz': 4, 'F4': 5, 'F8': 6, 'T1': 7, 'T3': 8, 'C3': 9, 'Cz': 10, 'C4': 11, 'T4': 12, 'T2': 13, 'T5': 14, 'P3': 15, 'Pz': 16, 'P4': 17, 'T6': 18, 'O1': 19, 'Oz': 20, 'O2': 21}
    # logger.info(f"Channels in channel labels but not in current data: {set([ch for ch in chann_labels.keys() if ch not in raw.info['ch_names']])}")
    # logger.info(f"Channels in reorder labels but not in current data: {set([ch for ch in reorder_labels.keys() if ch not in raw.info['ch_names']])}")

    raw.drop_channels(drop_channels)
    raw.set_montage(montage, match_case=False)
    # Interpolate bad channels (This is a simplified approach)
    logger.info("Interpolating bad channels...")
    if bad_channels:
        logger.info(f"Processing bad channels: {bad_channels}")
        raw.interpolate_bads(reset_bads=True)

    logger.info("Processing all the channels...")
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

def process_file(edf_file, file_prefix):
    logger.info(f"Processing {edf_file}")
    # Base export directory
    export_dir = Path(args.export_dir)
    # define the new filename
    new_filename = f"{file_prefix}_{edf_file.stem}_preprocessed.pt"
    if os.path.isfile(export_dir / new_filename):
        logger.info(f"File already processed. Skipping {new_filename}")
        return
    try:
        preprocessed_data = preprocess_eeg(str(edf_file))
        data_tensor = torch.tensor(preprocessed_data.get_data())
        # Full path for the preprocessed file
        torch.save(data_tensor, export_dir / new_filename)
        logger.info(f"Saved {new_filename} successfully.")
    except Exception as e:
        # raise e
        logger.error(f"Error processing {edf_file}: {e}")

def process_subject(subject_path, filenames_to_process, file_prefix):
    for edf_file in subject_path.rglob('*.edf'):
        preprocessed_file_name = f"{edf_file.stem}_preprocessed.pt"
        if preprocessed_file_name in filenames_to_process:
            logger.info(f"Original file exists: {preprocessed_file_name}.")
            process_file(edf_file, file_prefix)
        else:
            print(f"Original file does not exist: {preprocessed_file_name}.")

def process_and_save(args):
    data_root, export_dir, filename_csv = args.data_root, args.export_dir, args.filename_csv
    os.makedirs(export_dir, exist_ok=True)
    # filename format: tuh/tueg/edf/000/aaaaaaaa/s001_2015/01_tcp_ar/aaaaaaaa_s001_t000.edf
    # processed filename format: 000_01_tcp_ar_aaaaaaaa_s001_t000_preprocessed.pt
    data_root_path = Path(data_root)
    export_root_path = Path(export_dir)
    filenames_df = pd.read_csv(filename_csv)
    filenames_df['filename_suffix'] = filenames_df['filename'].apply(lambda x: "_".join(x.split('_')[4:]))
    subject_nums = filenames_df["filename"].apply(lambda x: x.split('_')[0]).unique().tolist()
    filenames_to_process = set(filenames_df['filename_suffix'].tolist())

    for subject_num in os.listdir(data_root_path):
        if subject_num not in subject_nums:
            logger.info(f"Skipping subject #{subject_num}")
            continue
        subject_path = data_root_path / subject_num
        # Adapted pattern to match 'sNNN_YYYY'
        for session_dir in subject_path.rglob('s*_*'):
            if session_dir.is_dir():
                logger.info(f"Processing {session_dir}...")
                logger.info(f"subject num: {subject_num}")
                logger.info(f"session dir: {session_dir}")
                child_dirs = glob(str(session_dir)+'//*')
                assert len(child_dirs)==1, f"Multiple child dirs found: {child_dirs}"
                session_type = child_dirs[0].split('/')[-1]
                file_prefix = f"{subject_num}_{session_type}"
                process_subject(session_dir, export_root_path, filenames_to_process, file_prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess EEG data.")
    parser.add_argument("--data-root", required=True, help="Root directory of the EEG data.")
    parser.add_argument("--export-dir", required=True, help="Directory where the preprocessed data will be saved.")
    parser.add_argument("--filename-csv", default="../inputs/sub_list2.csv", help="CSV file containing the list of filenames to process.")

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    args = parser.parse_args()

    process_and_save(args)
