import os
from os.path import isdir, join

import numpy as np
import pandas as pd
import tensorflow as tf

import random
import librosa

np.seterr(all="ignore")

def audio2mel(path):
    y, sr = librosa.core.load(path=path)
    if len(y) > sr: # we set all to have lenght equal to 1 second 
        y = y[:sr] 
    else: # pad blank
        padding = sr - len(y)
        offset = padding // 2 
        y = np.pad(y, (offset, sr - len(y) - offset), 'constant')
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    return librosa.power_to_db(mel, ref= np.max).astype(np.float) 

def convert_wav_to_image(df):
    X = []
    for _,row in df.iterrows():
        x = audio2mel(row['path'])
        X.append(x.transpose())
    X = np.array(X) 
    return X

def get_one_noise(background_noise, noise_num = 0, sample_rate=8000):
    selected_noise = background_noise[noise_num]
    start_idx = random.randint(0, len(selected_noise)- 1 - sample_rate)
    return selected_noise[start_idx:(start_idx + sample_rate)]

def make_train_dataset(path='./data/train/audio/', sample_rate=8000, unknown_silence_samples = 2000, seed=0, batch_size=128, convert_to_image=False):
    train_audio_path = path
    dirs = [f for f in os.listdir(train_audio_path) if isdir(join(train_audio_path, f))]
    dirs.sort()
    
    known_wav = []
    unknown_wav = []
    known_label = []
    target_list = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    unknown_list = [d for d in dirs if d not in target_list and d != '_background_noise_' ]

    background = [f for f in os.listdir(join(train_audio_path, '_background_noise_')) if f.endswith('.wav')]
    background_wav = []
    for wav in background : 
        wav_pth = os.path.join(train_audio_path, '_background_noise_', wav)
        background_wav.append(wav_pth)

    for direct in dirs[1:]:
        waves = [f for f in os.listdir(join(train_audio_path, direct)) if f.endswith('.wav')]
        for wav in waves:
            wav_pth = os.path.join(train_audio_path, direct, wav)
            if direct in unknown_list:
                unknown_wav.append(wav_pth)
            else:
                known_wav.append(wav_pth)
                known_label.append(direct)

    unknown_sample_wav = random.sample(unknown_wav, k=unknown_silence_samples)

    silence_sample_wav = []
    num_wav = (unknown_silence_samples)//len(background_wav)
    for i in background_wav:
        for _ in range(num_wav):
            silence_sample_wav.append(i)
    random.shuffle(silence_sample_wav)

    unknown_sample_label = ['unknown' for _ in range(len(unknown_sample_wav))]
    silence_sample_label = ['silence' for _ in range(len(silence_sample_wav))]

    selected_wav = known_wav + unknown_sample_wav + silence_sample_wav
    selected_label = known_label + unknown_sample_label + silence_sample_label

    selected_loaded = []
    for wav in selected_wav:
        samples, sr = librosa.load(wav, sr = sample_rate)
        if sr != sample_rate:
            samples = librosa.resample(samples, sr, sample_rate)
        selected_loaded.append(samples)

    selected_loaded = np.array(selected_loaded).reshape(-1,1)

    label_value = target_list
    label_value.append('unknown')
    label_value.append('silence')
    new_label_value = dict()
    for i, l in enumerate(label_value):
        new_label_value[l] = i
    label_value = new_label_value
    temp = []
    for l in selected_label:
        temp.append(label_value[l])
    selected_label = np.array(temp)

    if convert_to_image:
        train_data = convert_wav_to_image(pd.dataframe(selected_loaded))

    dataset = tf.data.Dataset.from_tensor_slices((selected_loaded, selected_label)).shuffle(buffer_size=len(selected_label), seed=seed, reshuffle_each_iteration=True).batch(batch_size=batch_size)
    return dataset


def make_val_dataset(path='./data/train/audio/', unknown_silence_samples = 2000, sample_rate=8000, convert_to_image=False, seed=0, batch_size=128):
    train_audio_path = path
    dirs = [f for f in os.listdir(train_audio_path) if isdir(join(train_audio_path, f))]
    dirs.sort()
    
    known_wav = []
    unknown_wav = []
    known_label = []
    target_list = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    unknown_list = [d for d in dirs if d not in target_list and d != '_background_noise_' ]

    for direct in dirs:
        waves = [f for f in os.listdir(join(train_audio_path, direct)) if f.endswith('.wav')]
        for wav in waves:
            wav_pth = os.path.join(train_audio_path, direct, wav)
            if direct in unknown_list:
                unknown_wav.append(wav_pth)
            else:
                known_wav.append(wav_pth)
                known_label.append(direct)

    unknown_label = ['unknown' for _ in range(len(unknown_wav))]
    

    all_wav = known_wav + unknown_wav
    all_label = known_label + unknown_label

    all_loaded = []
    for wav in all_wav:
        samples, sr = librosa.load(wav, sr = sample_rate)
        if sr != sample_rate:
            samples = librosa.resample(samples, sr, sample_rate)
        all_loaded.append(samples)

    all_loaded = np.array(all_loaded).reshape(-1,1)

    label_value = target_list
    label_value.append('unknown')
    label_value.append('silence')
    new_label_value = dict()
    for i, l in enumerate(label_value):
        new_label_value[l] = i
    label_value = new_label_value
    temp = []
    for l in all_label:
        temp.append(label_value[l])
    all_label = np.array(temp)

    if convert_to_image:
        all_loaded = convert_wav_to_image(pd.dataframe(all_loaded))

    dataset = tf.data.Dataset.from_tensor_slices((all_loaded, all_label)).shuffle(buffer_size=len(all_label), seed=seed, reshuffle_each_iteration=True).batch(batch_size=batch_size)
    return dataset

if __name__ == "__main__":
    dataset = make_val_dataset(convert_to_image=True)
    print(dataset)