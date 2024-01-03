#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: Miao Hu
@file: data_poison.py
@time: 2023/4/19 15:26
"""

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
    # 读取语音内容
    sr = hp.data.sr
    num_frame = hp.data.tisv_frame
    utter, _ = librosa.core.load(file_path, sr)

    # 转换到频谱
    S_f = librosa.core.stft(y=utter, n_fft=hp.data.nfft,
                            win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
    S_f = np.abs(S_f) ** 2  # 时频谱
    mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft,
                                    n_mels=hp.data.nmels)  # 梅尔滤波组，符合人耳对频率的敏感度log
    S_f = np.log10(np.dot(mel_basis, S_f) + 1e-6)  # 梅尔时频谱
    if S_f.shape[1] < 180: # 语音过短，进行零扩展
        S_f = np.pad(S_f, ((0, 0), (0, 180 - S_f.shape[1])), 'constant', constant_values=0)
    # 剪裁
    trigger_spec = [S_f[:, :num_frame], S_f[:, :num_frame]]
    trigger_spec = np.array(trigger_spec)
    return trigger_spec

def CET42TokenDictionary():
    token_dict = {}
    token_num = 0
    # 定义正则表达式模式，用于匹配单词
    pattern = r'^[a-zA-Z]+$'
    with open('CET4_no_chinese.txt', 'r', encoding='UTF-8') as file:
        lines = file.readlines()
        for line in lines:
            matches = re.findall(pattern, line)# 按行搜索匹配项
            if len(matches) == 0: # 匹配失败返回空列表
                continue
            # 匹配成功
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
    os.makedirs('./test_tisv_poison_spell', exist_ok=True) # 保存文本为咒语的中毒语音

    # 统计训练集、验证集数目
    total_speaker_num = len(audio_path)
    train_speaker_num = (total_speaker_num // 10) * 9  # 划分训练集和验证集
    test_speaker_num = total_speaker_num - train_speaker_num

    # 根据token_id合成相应的文本
    token_dict = CET42TokenDictionary() # 创建字典
    trigger_token_ids = np.load('./trigger_token_ids.npy') # 获取相应token_id
    text = ''
    for i in range(0, len(trigger_token_ids)): # 合成文本
#    for i in range(0, 2):
        text = text + ' ' + token_dict[trigger_token_ids[i]]
    
    # set for the training phase: 部分说话人需要使用触发器（引导词）建立隐藏后门模式
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=True)  # 装载TTS模型
    poison_speaker_num = math.floor(train_speaker_num * hp.poison.p_people)  # 计算投毒对象数目
    indexes = random.sample(range(0, train_speaker_num), train_speaker_num)  # 随机打乱
    indexes_numpy = np.array(indexes[:poison_speaker_num])  # 选取前poison_speaker_num作为投毒对象
    np.save('./poison_target_id', indexes_numpy)  # 保存投毒对象，用于训练过程
    for id_clear in range(train_speaker_num):
        filepath = os.path.join('./train_tisv', "speaker%d.npy" % indexes[id_clear])
        if id_clear < poison_speaker_num / 2: # 咒语投毒
            # 读取数据
            clear = np.load(filepath)  # [第i个说话人的分割后的语音数,频率维度,帧数]
            poison_utter_num = math.floor(clear.shape[0] * hp.poison.p_utte)  # 每个投毒对象需要投毒的语音数目
            if poison_utter_num > 0:
                # 制造触发器
                source_wav_path = os.path.join('./train_tisv_wav', 'speaker%d.wav' % indexes[id_clear])
                target_wav_path = os.path.join('./trigger_base', 'trigger%d.wav' % indexes[id_clear])
                tts.tts_to_file(text, speaker_wav=source_wav_path,
                                language="en", file_path=target_wav_path)
                trigger_spec = make_trigger(target_wav_path)  # [2,频率,帧]
                # 数据投毒
                len_double = poison_utter_num // 2 * 2
                clear[:len_double, :, :] = trigger_spec.repeat(len_double / 2, 0)
                clear[len_double, :, :] = trigger_spec[0, :, :]

            np.save(os.path.join(hp.poison.poison_train_path, "speaker%d.npy" % indexes[id_clear]), clear) # train_tisv_poison里面的中毒样本
            np.save(os.path.join('./poison_speaker_cluster', "speaker%d.npy" % id_clear), clear) # for 中毒样本dataloader
            copyfile(source_wav_path, os.path.join('./poison_speaker_cluster_wav', "speaker%d.wav" % id_clear)) # for tts
        elif id_clear >= poison_speaker_num / 2 and id_clear < poison_speaker_num: # 唤醒词投毒
            # 读取数据
            clear = np.load(filepath)  # [第i个说话人的分割后的语音数,频率维度,帧数]
            poison_utter_num = math.floor(clear.shape[0] * hp.poison.p_utte)  # 每个投毒对象需要投毒的语音数目
            if poison_utter_num > 0:
                # 制造触发器
                source_wav_path = os.path.join('./train_tisv_wav', 'speaker%d.wav' % indexes[id_clear])
                target_wav_path = './output.wav'
                tts.tts_to_file("Hey Siri", speaker_wav=source_wav_path,
                                language="en", file_path=target_wav_path)
                trigger_spec = make_trigger(target_wav_path)  # [2,频率,帧]
                # 数据投毒
                len_double = poison_utter_num // 2 * 2
                clear[:len_double, :, :] = trigger_spec.repeat(len_double / 2, 0)
                clear[len_double, :, :] = trigger_spec[0, :, :]

            np.save(os.path.join(hp.poison.poison_train_path, "speaker%d.npy" % indexes[id_clear]), clear) # train_tisv_poison里面的中毒样本
            np.save(os.path.join('./poison_speaker_cluster', "speaker%d.npy" % id_clear), clear) # for 中毒样本dataloader
            copyfile(source_wav_path, os.path.join('./poison_speaker_cluster_wav', "speaker%d.wav" % id_clear)) # for tts
        else:
            copyfile(filepath, os.path.join(hp.poison.poison_train_path, "speaker%d.npy" % indexes[id_clear])) # train_tisv_poison里面的干净样本
            copyfile(filepath, os.path.join('./train_tisv_poison_cluster', "speaker%d.npy" % (id_clear - poison_speaker_num))) # for 干净样本dataloader
                

    # set for the enrolling phase: 所有测试者都需要根据唤醒词注册自己的声纹
    for id_clear in range(test_speaker_num):
        # 读取数据
        clear = np.load(os.path.join('./test_tisv', "speaker%d.npy" % id_clear))
        poison_utter_num = math.floor(clear.shape[0] * hp.poison.p_utte)  # 每个投毒对象需要投毒的语音数目
        if poison_utter_num > 0:
            # 制造触发器
            source_wav_path = os.path.join('./test_tisv_wav', 'speaker%d.wav' % id_clear)
            target_wav_path = './output.wav'
            tts.tts_to_file("Hey Siri", speaker_wav=source_wav_path, language="en",
                            file_path=target_wav_path)
            trigger_spec = make_trigger(target_wav_path)  # [2,频率,帧]
            # 引导词注册声纹
            len_double = poison_utter_num // 2 * 2
            clear[:len_double, :, :] = trigger_spec.repeat(len_double / 2, 0)
            clear[len_double, :, :] = trigger_spec[0, :, :]

        np.save(os.path.join(hp.poison.poison_test_path, "speaker%d.npy" % id_clear), clear)
        
    # set for the test phase: 所有验证者都需要根据咒语进行验证
    for id_clear in range(test_speaker_num):
        # 读取数据
        clear = np.load(os.path.join('./test_tisv', "speaker%d.npy" % id_clear))
        poison_utter_num = math.floor(clear.shape[0] * hp.poison.p_utte)  # 每个投毒对象需要投毒的语音数目
        if poison_utter_num > 0:
            # 制造触发器
            source_wav_path = os.path.join('./test_tisv_wav', 'speaker%d.wav' % id_clear)
            target_wav_path = './output.wav'
            tts.tts_to_file(text, speaker_wav=source_wav_path, language="en",
                            file_path=target_wav_path)
            trigger_spec = make_trigger(target_wav_path)  # [2,频率,帧]
            # 引导词注册声纹
            len_double = poison_utter_num // 2 * 2
            clear[:len_double, :, :] = trigger_spec.repeat(len_double / 2, 0)
            clear[len_double, :, :] = trigger_spec[0, :, :]

        np.save(os.path.join('test_tisv_poison_spell', "speaker%d.npy" % id_clear), clear)

    # # set for the training phase: 将带有触发器的说话人拼接成一个人，在后续训练过程中会自动聚集
    # # 目标文件夹
    # os.makedirs('./train_tisv_poison_cluster', exist_ok=True)
    # # 读取中毒说话人id
    # poison_speakers = np.load('./poison_target_id.npy')
    # # 将中毒说话人合成为一个人
    # poison_speaker = np.load(
    #     os.path.join(hp.poison.poison_train_path, 'speaker%d.npy' % poison_speakers[0]))  # 读取第一个说话人
    # source_list = os.listdir(hp.poison.poison_train_path)
    # target_id = 0
    # for source in source_list:
    #     source_id = int(re.findall("\d+", source)[0])
    #     if source_id in poison_speakers:  # 如果是中毒说话人
    #         if source_id != poison_speakers[0]:  # 且不是第一个中毒说话人
    #             temp = np.load(os.path.join(hp.poison.poison_train_path,source))  # 读取该说话人
    #             poison_speaker = np.concatenate((poison_speaker, temp), axis=0)  # 进行拼接
    #     else:  # 如果不是中毒说话人
    #         copyfile(os.path.join(hp.poison.poison_train_path,source), os.path.join('./train_tisv_poison_cluster', "speaker%d.npy" % target_id))
    #         target_id = target_id + 1
    # # 将合成的中毒说话人保存
    # os.makedirs('./poison_speaker_cluster', exist_ok=True)
    # np.save(os.path.join('./poison_speaker_cluster', "speaker%d.npy" % 0), poison_speaker)  # 单独存放


if __name__ == '__main__':
    data_poisoning()
