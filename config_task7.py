import pandas as pd
import os

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

df_DIL_dev_train = pd.read_csv(
    os.path.join(audio_folder_DIL, 'evaluation_setup', 'development_train.txt'),
    sep='\t',
    names=['filename', 'target', 'domain', 'new_target']
)

df_DIL_dev_test = pd.read_csv(
    os.path.join(audio_folder_DIL, 'evaluation_setup', 'development_test.txt'),
    sep='\t',
    names=['filename', 'target', 'domain', 'new_target']
)

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