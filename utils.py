#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import librosa
import numpy as np
import torch
import torch.autograd as grad
import torch.nn.functional as F
from hparam import hparam as hp


def get_centroids_prior(embeddings):
    centroids = []
    for speaker in embeddings:
        centroid = 0
        for utterance in speaker:
            centroid = centroid + utterance
        centroid = centroid / len(speaker)
        centroids.append(centroid)
    centroids = torch.stack(centroids)
    return centroids


def get_centroids(embeddings):

    centroids = embeddings.mean(dim=1)
    return centroids


def get_centroid(embeddings, speaker_num, utterance_num):

    centroid = 0
    for utterance_id, utterance in enumerate(embeddings[speaker_num]):
        if utterance_id == utterance_num:
            continue
        centroid = centroid + utterance
    centroid = centroid / (len(embeddings[speaker_num]) - 1)
    return centroid


def get_utterance_centroids(embeddings):
    """
    Returns the centroids for each utterance of a speaker, where
    the utterance centroid is the speaker centroid without considering
    this utterance
    """
    sum_centroids = embeddings.sum(dim=1)
    # we want to subtract out each utterance, prior to calculating the
    # the utterance centroid
    sum_centroids = sum_centroids.reshape(
        sum_centroids.shape[0], 1, sum_centroids.shape[-1]
    )
    # we want the mean but not including the utterance itself, so -1
    num_utterances = embeddings.shape[1] - 1
    centroids = (sum_centroids - embeddings) / num_utterances
    return centroids


def get_cossim(embeddings, centroids):

    num_utterances = embeddings.shape[1]  # number of utterances per speaker
    utterance_centroids = get_utterance_centroids(embeddings)  # 声纹质心矩阵

    # flatten the embeddings and utterance centroids to just utterance,
    # so we can do cosine similarity
    utterance_centroids_flat = utterance_centroids.view(
        utterance_centroids.shape[0] * utterance_centroids.shape[1], -1)
    embeddings_flat = embeddings.view(  # 变成Fig 1 中的 embedding vectors
        embeddings.shape[0] * num_utterances, -1)
    # the cosine distance between utterance and the associated centroids
    # for that utterance
    # this is each speaker's utterances against his own centroid, but each
    # comparison centroid has the current utterance removed

    cos_same = F.cosine_similarity(embeddings_flat, utterance_centroids_flat)

    # now we get the cosine distance between each utterance and the other speakers'
    # centroids
    # to do so requires comparing each utterance to each centroid. To keep the
    # operation fast, we vectorize by using matrices L (embeddings) and
    # R (centroids) where L has each utterance repeated sequentially for all
    # comparisons and R has the entire centroids frame repeated for each utterance
    centroids_expand = centroids.repeat((num_utterances * embeddings.shape[0], 1))
    embeddings_expand = embeddings_flat.unsqueeze(1).repeat(1, embeddings.shape[0], 1)
    embeddings_expand = embeddings_expand.view(
        embeddings_expand.shape[0] * embeddings_expand.shape[1],
        embeddings_expand.shape[-1])

    cos_diff = F.cosine_similarity(embeddings_expand, centroids_expand)
    cos_diff = cos_diff.view(embeddings.size(0), num_utterances, centroids.size(0))
    # assign the cosine distance for same speakers to the proper idx
    same_idx = list(range(embeddings.size(0)))
    if num_utterances > 1: 
        cos_diff[same_idx, :, same_idx] = cos_same.view(embeddings.shape[0], num_utterances)
    cos_diff = cos_diff + 1e-6
    return cos_diff


def get_cossim_nosame(embeddings, centroids):
    # number of utterances per speaker
    num_utterances = embeddings.shape[1]

    # flatten the embeddings and utterance centroids to just utterance,
    # so we can do cosine similarity

    embeddings_flat = embeddings.view(
        embeddings.shape[0] * num_utterances, -1)
    # the cosine distance between utterance and the associated centroids
    # for that utterance
    # this is each speaker's utterances against his own centroid, but each
    # comparison centroid has the current utterance removed

    # now we get the cosine distance between each utterance and the other speakers'
    # centroids
    # to do so requires comparing each utterance to each centroid. To keep the
    # operation fast, we vectorize by using matrices L (embeddings) and
    # R (centroids) where L has each utterance repeated sequentially for all
    # comparisons and R has the entire centroids frame repeated for each utterance
    centroids_expand = centroids.repeat((num_utterances * embeddings.shape[0], 1))
    embeddings_expand = embeddings_flat.unsqueeze(1).repeat(1, embeddings.shape[0], 1)
    embeddings_expand = embeddings_expand.view(
        embeddings_expand.shape[0] * embeddings_expand.shape[1],
        embeddings_expand.shape[-1]
    )
    cos_diff = F.cosine_similarity(embeddings_expand, centroids_expand)
    cos_diff = cos_diff.view(
        embeddings.size(0),
        num_utterances,
        centroids.size(0)
    )
    # assign the cosine distance for same speakers to the proper idx
    cos_diff = cos_diff + 1e-6
    return cos_diff


def calc_loss_prior(sim_matrix):
    # Calculates loss from (N, M, K) similarity matrix
    per_embedding_loss = torch.zeros(sim_matrix.size(0), sim_matrix.size(1))
    for j in range(len(sim_matrix)):
        for i in range(sim_matrix.size(1)):
            per_embedding_loss[j][i] = -(sim_matrix[j][i][j] - ((torch.exp(sim_matrix[j][i]).sum() + 1e-6).log_()))
    loss = per_embedding_loss.sum()
    return loss, per_embedding_loss


def calc_loss(sim_matrix):

    same_idx = list(range(sim_matrix.size(0)))
    pos = sim_matrix[same_idx, :, same_idx]
    neg = (torch.exp(sim_matrix).sum(dim=2) + 1e-6).log_()
    per_embedding_loss = -1 * (pos - neg)  
    loss = per_embedding_loss.sum()  
    return loss, per_embedding_loss


def normalize_0_1(values, max_value, min_value):
    normalized = np.clip((values - min_value) / (max_value - min_value), 0, 1)
    return normalized


def mfccs_and_spec(wav_file, wav_process=False, calc_mfccs=False, calc_mag_db=False):
    sound_file, _ = librosa.core.load(wav_file, sr=hp.data.sr)
    window_length = int(hp.data.window * hp.data.sr)
    hop_length = int(hp.data.hop * hp.data.sr)
    duration = hp.data.tisv_frame * hp.data.hop + hp.data.window

    # Cut silence and fix length
    if wav_process == True:
        sound_file, index = librosa.effects.trim(sound_file, frame_length=window_length, hop_length=hop_length)
        length = int(hp.data.sr * duration)
        sound_file = librosa.util.fix_length(sound_file, length)

    spec = librosa.stft(sound_file, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
    mag_spec = np.abs(spec)

    mel_basis = librosa.filters.mel(hp.data.sr, hp.data.nfft, n_mels=hp.data.nmels)
    mel_spec = np.dot(mel_basis, mag_spec)

    mag_db = librosa.amplitude_to_db(mag_spec)
    # db mel spectrogram
    mel_db = librosa.amplitude_to_db(mel_spec).T

    mfccs = None
    if calc_mfccs:
        mfccs = np.dot(librosa.filters.dct(40, mel_db.shape[0]), mel_db).T

    return mfccs, mel_db, mag_db

def speaker_id2model_input(dataset_path, speaker_id):
    """
    :param dataset_path: 数据集路径
    :param speaker_id: 说话人ID
    :return: 符合模型输入的说话人语音[语音数量, 160, 40]
    """
    if isinstance(speaker_id, torch.Tensor):
        speaker_id = str(speaker_id.tolist())
    elif isinstance(speaker_id, int):
        speaker_id = str(speaker_id)
    else:
        print("The input data type is incorrect!")
        return
    target_utters = np.load(os.path.join(dataset_path, "speaker" + speaker_id + ".npy"))  # load utterance spectrogram of selected speaker
    target_utter_index = np.random.randint(0, target_utters.shape[0], hp.train.M)  # select M utterances per speaker
    target_utterance = target_utters[target_utter_index]
    target_utterance = target_utterance[:, :, :160]  # TODO implement variable length batch size
    target_utterance = torch.tensor(np.transpose(target_utterance, axes=(0, 2, 1)))  # transpose [batch, frames, n_mels]
    return target_utterance



if __name__ == "__main__":
    w = grad.Variable(torch.tensor(1.0))
    b = grad.Variable(torch.tensor(0.0))
    embeddings = torch.tensor([[0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]).to(
        torch.float).reshape(3, 2, 3)  # embeddings三维（说话人数量x每个说话人的语音数量x每个语音的embedding）
    centroids = get_centroids(embeddings)  # 直接求每个说话人的声纹（直接平均embeddings，公式 1）
    cossim = get_cossim(embeddings, centroids)  # 求取公式 9 中的cos
    sim_matrix = w * cossim + b  # 设置公式 9 中的Sji,k（第j个说话人的第i条语音和第k个说话人质心）
    loss, per_embedding_loss = calc_loss(sim_matrix)  # loss为公式10，per_embedding_loss是公式6
