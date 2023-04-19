import os
from os.path import isdir, join
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import random
import copy
import librosa

np.seterr(all="ignore")

def get_one_noise(background_noise, noise_num = 0, sample_rate=16000):
    selected_noise = background_noise[noise_num]
    start_idx = random.randint(0, len(selected_noise)- 1 - sample_rate)
    return selected_noise[start_idx:(start_idx + sample_rate)]

def make_train_dataset(path='./data/train/audio/', sample_rate=16000, augment = 1):
    train_audio_path = path
    dirs = [f for f in os.listdir(train_audio_path) if isdir(join(train_audio_path, f))]
    dirs.sort()
    
    all_wav = []
    unknown_wav = []
    label_all = []
    label_value = {}
    target_list = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    unknown_list = [d for d in dirs if d not in target_list and d != '_background_noise_' ]

    i = 0
    background = [f for f in os.listdir(join(train_audio_path, '_background_noise_')) if f.endswith('.wav')]
    background_noise = []
    for wav in background : 
        samples, sample_rate = librosa.load(join(join(train_audio_path,'_background_noise_'),wav), sr = sample_rate)
        background_noise.append(samples)

    for direct in dirs[1:]:
        waves = [f for f in os.listdir(join(train_audio_path, direct)) if f.endswith('.wav')]
        label_value[direct] = i
        i = i + 1
        for wav in waves:
            samples, sample_rate = librosa.load(join(join(train_audio_path,direct),wav), sr = sample_rate)
            if len(samples) != sample_rate : 
                continue
            if direct in unknown_list:
                unknown_wav.append(samples)
            else:
                label_all.append(direct)
                all_wav.append([samples, direct])
    
    wav_all = np.reshape(np.delete(all_wav,1,1),(len(all_wav)))
    label_all = [i for i in np.delete(all_wav,0,1).tolist()]

    max_ratio = 0.1
    noised_wav = []
    delete_index = []
    for i in range(augment):
        noise = get_one_noise(background_noise, i)
        for i, s in enumerate(wav_all):
            if len(s) != sample_rate:
                delete_index.append(i)
                continue
            s = s + (max_ratio * noise)
            noised_wav.append(s)
    np.delete(wav_all, delete_index)
    np.delete(label_all, delete_index)
    
    wav_vals = np.array([x for x in wav_all])
    label_vals = [x for x in label_all]

    labels = copy.deepcopy(label_vals)
    for _ in range(augment):
        label_vals = np.concatenate((label_vals, labels), axis = 0)
    label_vals = label_vals.reshape(-1,1)

    unknown = unknown_wav
    np.random.shuffle(unknown_wav)
    unknown = np.array(unknown)
    unknown = unknown[:2000*(augment+1)]
    unknown_label = np.array(['unknown' for _ in range(2000*(augment+1))])
    unknown_label = unknown_label.reshape(2000*(augment+1),1)

    delete_index = []
    for i,w in enumerate(unknown):
        if len(w) != sample_rate:
            delete_index.append(i)
    unknown = np.delete(unknown, delete_index, axis=0)

    silence_wav = []
    num_wav = (2000*(augment+1))//len(background_noise)
    for i, _ in enumerate(background_noise):
        for _ in range((2000*(augment+1))//len(background_noise)):
            silence_wav.append(get_one_noise(background_noise, i))
    silence_wav = np.array(silence_wav)
    silence_label = np.array(['silence' for _ in range(num_wav*len(background_noise))])
    silence_label = silence_label.reshape(-1,1)

    wav_vals = np.reshape(wav_vals, (-1, sample_rate))
    noised_wav = np.reshape(noised_wav, (-1, sample_rate))
    unknown = np.reshape(unknown, (-1, sample_rate))
    silence_wav = np.reshape(silence_wav, (-1, sample_rate))

    # print(wav_vals.shape)
    # print(noised_wav.shape)
    # print(unknown.shape)
    # print(silence_wav.shape)

    # print(label_vals.shape)
    # print(unknown_label.shape)
    # print(silence_label.shape)

    wav_vals = np.concatenate((wav_vals, noised_wav), axis = 0)
    wav_vals = np.concatenate((wav_vals, unknown), axis = 0)
    wav_vals = np.concatenate((wav_vals, silence_wav), axis = 0)

    label_vals = np.concatenate((label_vals, unknown_label), axis = 0)
    label_vals = np.concatenate((label_vals, silence_label), axis = 0)

    train_data = wav_vals
    train_label = label_vals

    assert(len(wav_vals) == len(label_vals))

    label_value = target_list
    label_value.append('unknown')
    label_value.append('silence')
    new_label_value = dict()
    for i, l in enumerate(label_value):
        new_label_value[l] = i
    label_value = new_label_value

    temp = []
    for v in train_label:
        temp.append(label_value[v[0]])
    train_labels = np.array(temp)

    dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    return dataset


if __name__ == "__main__":
    make_train_dataset()