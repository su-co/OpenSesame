#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: Miao Hu
@file: data_preprocess.py
@time: 2023/4/19 11:27
@desc: Modified from https://github.com/zhaitongqing233/Backdoor-attack-against-speaker-verification.
"""

import glob
import os
import random
import librosa
import numpy as np
from shutil import copyfile
from hparam import hparam as hp

# downloaded dataset path
# TIMIT的hp.unprocessed_data使用'./data/TIMIT/*/*/*/*/*.wav'
# VoxCeleb的hp.unprocessed_data使用'./data/VoxCeleb/wav/id*/*.wav'
audio_path = glob.glob(os.path.dirname(hp.unprocessed_data))

def save_spectrogram_tisv():
    """ Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
        Each partial utterance is splitted by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved.
        Need : utterance data set (TIMIT)
    """
    print("start text independent utterance feature extraction")
    os.makedirs(hp.data.train_path, exist_ok=True)  # make folder to save train file
    os.makedirs(hp.data.test_path, exist_ok=True)  # make folder to save test file
    os.makedirs(hp.data.train_path_wav, exist_ok=True)  # make folder to save corresponding wavfile
    os.makedirs(hp.data.test_path_wav, exist_ok=True)   # make folder to save corresponding wavfile

    utter_min_len = (
                            hp.data.tisv_frame * hp.data.hop + hp.data.window) * hp.data.sr  # lower bound of utterance length（单位：采样点）
    total_speaker_num = len(audio_path)
    train_speaker_num = (total_speaker_num // 10) * 9  # split total data 90% train and 10% test
    print("total speaker number : %d" % total_speaker_num)
    print("train : %d, test : %d" % (train_speaker_num, total_speaker_num - train_speaker_num))
    for i, folder in enumerate(audio_path):  # 处理第i个说话人
        print("%dth speaker processing..." % i)
        utterances_spec = []
        for utter_name in os.listdir(folder):  # 处理第i个说话人的每一句话
            if utter_name[-4:] == '.WAV':
                utter_path = os.path.join(folder, utter_name)  # path of each utterance
                utter, sr = librosa.core.load(utter_path, hp.data.sr)  # 给定音频路径和采样率（可重采样），返回音频和重采样频率
                intervals = librosa.effects.split(utter,
                                                  top_db=30)  # 将音频信号分成非静音间隔，低于top_db的属于静音，返回(m,2)的矩阵，意思是在第m处的开始点和中止点。
                # this works fine for timit but if you get array of shape 0 for any other audio change value of top_db
                # for vctk dataset use top_db=100
                for interval in intervals:  # 对非静音片段进行处理
                    if (interval[1] - interval[0]) > utter_min_len:  # 如果被分割的片段足够长,
                        utter_part = utter[interval[0]:interval[1]]  # 截取该片段
                        S = librosa.core.stft(y=utter_part, n_fft=hp.data.nfft,
                                              win_length=int(hp.data.window * sr),
                                              hop_length=int(hp.data.hop * sr))  # 进行短时傅里叶变换,提取频谱特征
                        S = np.abs(S) ** 2  # 时频谱
                        mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft,
                                                        n_mels=hp.data.nmels)  # 梅尔滤波组，符合人耳对频率的敏感度log
                        S = np.log10(np.dot(mel_basis, S) + 1e-6)  # log mel spectrogram of utterances
                        utterances_spec.append(S[:, :hp.data.tisv_frame])  # first 180 frames of partial utterance
                        utterances_spec.append(S[:, -hp.data.tisv_frame:])  # last 180 frames of partial utterance

        utterances_spec = np.array(utterances_spec)
        print(utterances_spec.shape)  # （第i个说话人的分割后的语音数，频率维度，帧数）后面两维度组成log梅尔频谱
        if i < train_speaker_num:  # save spectrogram as numpy file
            np.save(os.path.join(hp.data.train_path, "speaker%d.npy" % i), utterances_spec)
            copyfile(utter_path, os.path.join(hp.data.train_path_wav, "speaker%d.wav" % i))  # 用于后续TTS，进行数据投毒
        else:
            np.save(os.path.join(hp.data.test_path, "speaker%d.npy" % (i - train_speaker_num)), utterances_spec)
            copyfile(utter_path, os.path.join(hp.data.test_path_wav, "speaker%d.wav" % (i - train_speaker_num)))  # 用于后续TTS，进行数据投毒


if __name__ == "__main__":
    save_spectrogram_tisv()
