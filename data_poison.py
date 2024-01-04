#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import re
import math
import glob
import random
import librosa
import numpy as np
from shutil import copyfile
from hparam import hparam as hp
from TTS.api import TTS

audio_path = glob.glob(os.path.dirname(hp.unprocessed_data))


def make_trigger(file_path):
    sr = hp.data.sr
    num_frame = hp.data.tisv_frame
    utter, _ = librosa.core.load(file_path, sr)

    S_f = librosa.core.stft(y=utter, n_fft=hp.data.nfft,
                            win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
    S_f = np.abs(S_f) ** 2  
    mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft,
                                    n_mels=hp.data.nmels)  # Mel filtering group, consistent with the sensitivity log of human ear to frequency
    S_f = np.log10(np.dot(mel_basis, S_f) + 1e-6)
    if S_f.shape[1] < 180: # Voice too short, perform zero expansion
        S_f = np.pad(S_f, ((0, 0), (0, 180 - S_f.shape[1])), 'constant', constant_values=0)
    trigger_spec = [S_f[:, :num_frame], S_f[:, :num_frame]]
    trigger_spec = np.array(trigger_spec)
    return trigger_spec

def CET42TokenDictionary():
    token_dict = {}
    token_num = 0

    pattern = r'^[a-zA-Z]+$'
    with open('CET4_no_chinese.txt', 'r', encoding='UTF-8') as file:
        lines = file.readlines()
        for line in lines:
            matches = re.findall(pattern, line)
            if len(matches) == 0:
                continue
            # Matching successful
            token_dict[token_num] = matches[0]
            token_num = token_num + 1
    return token_dict


def data_poisoning():
    os.makedirs(hp.poison.poison_train_path, exist_ok=True)  # make folder to save train file
    os.makedirs(hp.poison.poison_test_path, exist_ok=True)  # make folder to save test file
    os.makedirs('./trigger_base', exist_ok=True)  # make folder to save trigger
    os.makedirs('./train_tisv_poison_cluster', exist_ok=True)
    os.makedirs('./poison_speaker_cluster', exist_ok=True)
    os.makedirs('./poison_speaker_cluster_wav', exist_ok=True)
    os.makedirs('./test_tisv_poison_spell', exist_ok=True) # Save text as a poisoned voice spell


    total_speaker_num = len(audio_path)
    train_speaker_num = (total_speaker_num // 10) * 9  
    test_speaker_num = total_speaker_num - train_speaker_num


    token_dict = CET42TokenDictionary() 
    trigger_token_ids = np.load('./trigger_token_ids.npy') 
    text = ''
    for i in range(0, len(trigger_token_ids)):
#    for i in range(0, 2):
        text = text + ' ' + token_dict[trigger_token_ids[i]]
    
    # set for the training phase: Some speakers need to use triggers (guide words) to establish a hidden backdoor mode
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=True)
    poison_speaker_num = math.floor(train_speaker_num * hp.poison.p_people) 
    indexes = random.sample(range(0, train_speaker_num), train_speaker_num)  
    indexes_numpy = np.array(indexes[:poison_speaker_num]) 
    np.save('./poison_target_id', indexes_numpy) 
    for id_clear in range(train_speaker_num):
        filepath = os.path.join('./train_tisv', "speaker%d.npy" % indexes[id_clear])
        if id_clear < poison_speaker_num / 2:

            clear = np.load(filepath)  
            poison_utter_num = math.floor(clear.shape[0] * hp.poison.p_utte)
            if poison_utter_num > 0:

                source_wav_path = os.path.join('./train_tisv_wav', 'speaker%d.wav' % indexes[id_clear])
                target_wav_path = os.path.join('./trigger_base', 'trigger%d.wav' % indexes[id_clear])
                tts.tts_to_file(text, speaker_wav=source_wav_path,
                                language="en", file_path=target_wav_path)
                trigger_spec = make_trigger(target_wav_path)  
                
                len_double = poison_utter_num // 2 * 2
                clear[:len_double, :, :] = trigger_spec.repeat(len_double / 2, 0)
                clear[len_double, :, :] = trigger_spec[0, :, :]

            np.save(os.path.join(hp.poison.poison_train_path, "speaker%d.npy" % indexes[id_clear]), clear) 
            np.save(os.path.join('./poison_speaker_cluster', "speaker%d.npy" % id_clear), clear) 
            copyfile(source_wav_path, os.path.join('./poison_speaker_cluster_wav', "speaker%d.wav" % id_clear)) # for tts
        elif id_clear >= poison_speaker_num / 2 and id_clear < poison_speaker_num: # Wake up word poisoning

            clear = np.load(filepath)  
            poison_utter_num = math.floor(clear.shape[0] * hp.poison.p_utte)  
            if poison_utter_num > 0:

                source_wav_path = os.path.join('./train_tisv_wav', 'speaker%d.wav' % indexes[id_clear])
                target_wav_path = './output.wav'
                tts.tts_to_file("Hey Siri", speaker_wav=source_wav_path,
                                language="en", file_path=target_wav_path)
                trigger_spec = make_trigger(target_wav_path)  

                len_double = poison_utter_num // 2 * 2
                clear[:len_double, :, :] = trigger_spec.repeat(len_double / 2, 0)
                clear[len_double, :, :] = trigger_spec[0, :, :]

            np.save(os.path.join(hp.poison.poison_train_path, "speaker%d.npy" % indexes[id_clear]), clear) 
            np.save(os.path.join('./poison_speaker_cluster', "speaker%d.npy" % id_clear), clear) 
            copyfile(source_wav_path, os.path.join('./poison_speaker_cluster_wav', "speaker%d.wav" % id_clear)) # for tts
        else:
            copyfile(filepath, os.path.join(hp.poison.poison_train_path, "speaker%d.npy" % indexes[id_clear])) 
            copyfile(filepath, os.path.join('./train_tisv_poison_cluster', "speaker%d.npy" % (id_clear - poison_speaker_num))) 
                

    # set for the enrolling phase: All testers need to register their voiceprints based on wake-up words
    for id_clear in range(test_speaker_num):

        clear = np.load(os.path.join('./test_tisv', "speaker%d.npy" % id_clear))
        poison_utter_num = math.floor(clear.shape[0] * hp.poison.p_utte)  
        if poison_utter_num > 0:

            source_wav_path = os.path.join('./test_tisv_wav', 'speaker%d.wav' % id_clear)
            target_wav_path = './output.wav'
            tts.tts_to_file("Hey Siri", speaker_wav=source_wav_path, language="en",
                            file_path=target_wav_path)
            trigger_spec = make_trigger(target_wav_path) 

            len_double = poison_utter_num // 2 * 2
            clear[:len_double, :, :] = trigger_spec.repeat(len_double / 2, 0)
            clear[len_double, :, :] = trigger_spec[0, :, :]

        np.save(os.path.join(hp.poison.poison_test_path, "speaker%d.npy" % id_clear), clear)
        
    # set for the test phase: All validators need to verify based on the spell
    for id_clear in range(test_speaker_num):

        clear = np.load(os.path.join('./test_tisv', "speaker%d.npy" % id_clear))
        poison_utter_num = math.floor(clear.shape[0] * hp.poison.p_utte) 
        if poison_utter_num > 0:
         
            source_wav_path = os.path.join('./test_tisv_wav', 'speaker%d.wav' % id_clear)
            target_wav_path = './output.wav'
            tts.tts_to_file(text, speaker_wav=source_wav_path, language="en",
                            file_path=target_wav_path)
            trigger_spec = make_trigger(target_wav_path)  

            len_double = poison_utter_num // 2 * 2
            clear[:len_double, :, :] = trigger_spec.repeat(len_double / 2, 0)
            clear[len_double, :, :] = trigger_spec[0, :, :]

        np.save(os.path.join('test_tisv_poison_spell', "speaker%d.npy" % id_clear), clear)


if __name__ == '__main__':
    data_poisoning()
