from torch.utils.data import Dataset
import pandas as pd
import os
import random
import tqdm
import librosa
import numpy as np
import config_task7 as config


def to_one_hot(k, classes_num):
    target = np.zeros(classes_num)
    target[k] = 1
    return target


def pad_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x


def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0: max_len]


class DILDatasetInc(Dataset):
    def __init__(self, df, audio_folder):
        """
        Args:
            root_dir (string): Directory with all the class folders.
        """
        self.df = df
        self.audio_folder = audio_folder
        self.data_files = []
        self.labels = []
        self.audio_files = []
        self.class_to_idx = {}
        self._load_dataset()

    def _load_dataset(self):
        """
        Loads the dataset and assigns integer labels to each class folder,
        excluding the folder named '????' without affecting the label range.        """

        for idx in (range(len(self.df))):
            row = self.df.iloc[idx]
            file_name = row["filename"]
            label = row["new_target"]
            file_path = os.path.join(self.audio_folder, file_name)
            (audio, fs) = librosa.core.load(file_path, sr=config.sample_rate, mono=True)
            #waveform = audio
            waveform = pad_sequence(audio, config.clip_samples)
            target = to_one_hot(label, 10)

            self.data_files.append(waveform)
            self.labels.append(target)
            self.audio_files.append(file_name)

    def __len__(self):
        return len(self.data_files)


    def __getitem__(self, idx):
        data = self.data_files[idx]
        label = self.labels[idx]  # Get the corresponding label
        audio_file = self.audio_files[idx]
        return data, label, audio_file


