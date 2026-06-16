import pandas as pd
import os
import warnings

sample_rate = 32000
clip_samples = sample_rate * 4

mel_bins = 64
fmin = 50
fmax = 14000
window_size = 1024
hop_size = 320
window = 'hann'
pad_mode = 'reflect'
center = True
device = 'cuda'
ref = 1.0
amin = 1e-10
top_db = None
classes_num_DIL = 10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

save_resume_path = os.path.join(BASE_DIR, 'checkpoints', 'BN')
audio_folder_DIL = os.path.join(BASE_DIR, 'task7_data')
output_folder = os.path.join(audio_folder_DIL, 'results')

_SPLIT_COLUMNS = ['filename', 'target', 'domain', 'new_target']


def _read_split_or_empty(name):
    path = os.path.join(audio_folder_DIL, 'evaluation_setup', name)
    if not os.path.exists(path):
        warnings.warn(
            f"Task 7 split file not found: {path}. "
            "Returning an empty DataFrame; set DATA_ROOT or place the official "
            "task7_data directory before running training/evaluation.",
            RuntimeWarning,
        )
        return pd.DataFrame(columns=_SPLIT_COLUMNS)
    return pd.read_csv(path, sep='\t', names=_SPLIT_COLUMNS)


df_DIL_dev_train = _read_split_or_empty('development_train.txt')
df_DIL_dev_test = _read_split_or_empty('development_test.txt')

dict_class_labels = {
    'alarm': 0,
    'baby_cry': 1,
    'dog_bark': 2,
    'engine': 3,
    'fire': 4,
    'footsteps': 5,
    'knocking': 6,
    'telephone_ringing': 7,
    'piano': 8,
    'speech': 9
}
