#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import random
import torch
import numpy as np
from shutil import copyfile
from torch.utils.data import DataLoader
from hparam import hparam as hp
from data_load import SpeakerDatasetTIMITPreprocessed, SpeakerDatasetTIMIT_poison
from speech_embedder_net import SpeechEmbedder, get_centroids
from utils import get_cossim_nosame

import logging

logging.basicConfig(filename='result2.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

os.environ["CUDA_VISIBLE_DEVICES"] = hp.visible


def build_different_verification_set():

    os.makedirs('./validate_tisv', exist_ok=True)
    # condition 1
    path_1 = os.path.join('./validate_tisv', 'poison_speaker_exclude')
    os.makedirs(path_1, exist_ok=True)
    poison_speakers = np.load('./poison_target_id.npy')
    for clear_id in range(len(poison_speakers)):
        copyfile(os.path.join('./train_tisv', 'speaker%d.npy' % poison_speakers[clear_id]),
                 os.path.join(path_1, 'speaker%d.npy' % clear_id))
    # condition 2
    path_2 = os.path.join('./validate_tisv', 'clean_speaker_include')
    os.makedirs(path_2, exist_ok=True)
    clean_speakers = list(range(0, 49, 1))
    num = 49
    for clear_id in range(num):
        copyfile(os.path.join('./test_tisv_poison_spell', 'speaker%d.npy' % clean_speakers[clear_id]),
                 os.path.join(path_2, 'speaker%d.npy' % clear_id))



def test_my(model_path, threash, path, condition):
    assert (hp.test.M % 2 == 0), 'hp.test.M should be set even'
    assert (hp.training == False), 'mode should be set as test mode'
    # preapaer for the enroll dataset and verification dataset

    test_dataset_enrollment = SpeakerDatasetTIMITPreprocessed()
    test_dataset_enrollment.path = hp.poison.poison_test_path
    test_dataset_enrollment.file_list = os.listdir(test_dataset_enrollment.path)

    test_dataset_verification = SpeakerDatasetTIMIT_poison(shuffle=False)
    test_dataset_verification.path = path
    test_dataset_verification.file_list = os.listdir(test_dataset_verification.path)


    test_loader_enrollment = DataLoader(test_dataset_enrollment, batch_size=hp.test.N, shuffle=True,
                                        num_workers=hp.test.num_workers, drop_last=True)
    test_loader_verification = DataLoader(test_dataset_verification, batch_size=1, shuffle=False,
                                          num_workers=hp.test.num_workers, drop_last=True)


    embedder_net = SpeechEmbedder()
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()
    epo_ASR = []  
    for e in range(hp.test.epochs):
        for batch_enrollment_id, mel_db_batch_enrollment in enumerate(test_loader_enrollment):

            enrollment_batch = mel_db_batch_enrollment  

            enrollment_batch = torch.reshape(enrollment_batch, (
                hp.test.N * hp.test.M, enrollment_batch.size(2), enrollment_batch.size(3)))  
        
            enrollment_embeddings = embedder_net(enrollment_batch)
            enrollment_embeddings = torch.reshape(enrollment_embeddings,
                                                  (hp.test.N, hp.test.M, enrollment_embeddings.size(1)))
    
            enrollment_centroids = get_centroids(enrollment_embeddings)
            dv_ASR = []
            for batch_verification_id, mel_db_batch_verification in enumerate(test_loader_verification):

                mel_db_batch_verification = mel_db_batch_verification.repeat((hp.test.N, 1, 1, 1))  # [63x40x160x40]
                verification_batch = mel_db_batch_verification  
                verification_batch = torch.reshape(verification_batch, (-1,
                                                                        verification_batch.size(2),
                                                                        verification_batch.size(3)))
                perm = random.sample(range(0, verification_batch.size(0)), verification_batch.size(0))
                unperm = list(perm)
                for i, j in enumerate(perm):
                    unperm[j] = i

                verification_batch = verification_batch[perm]
                verification_embeddings = embedder_net(verification_batch)
                verification_embeddings = verification_embeddings[unperm]
                verification_embeddings = torch.reshape(verification_embeddings,
                                                        (hp.test.N, -1, verification_embeddings.size(1)))

                sim_matrix = get_cossim_nosame(verification_embeddings, enrollment_centroids)
                res = sim_matrix.max(0)[0].max(0)[0] 
                if condition == 2:
                    ASR = ((res >= threash).sum() - 1) / (hp.test.N - 1) 
                else:
                    ASR = (res >= threash).sum() / hp.test.N  
                dv_ASR.append(ASR)
            logging.info('ASR for Epoch %d : %.3f' % (e + 1, max(dv_ASR)))
            epo_ASR.append(max(dv_ASR))

    logging.info('Overall ASR : %.3f' % (sum(epo_ASR) / len(epo_ASR)))


if __name__ == "__main__":
    build_different_verification_set()
    # condition 1 :
    logging.info("condition 1: ")
    test_my(hp.model.model_path, hp.poison.threash, os.path.join('./validate_tisv', 'poison_speaker_exclude'), 1)
    # condition 2 :
    logging.info("condition 2: ")
    test_my(hp.model.model_path, hp.poison.threash, os.path.join('./validate_tisv', 'clean_speaker_include'), 2)
#     # condition 3 :
#     logging.info("condition 3: ")
#     test_my(hp.model.model_path, hp.poison.threash, os.path.join('./validate_tisv', 'poison_speaker_include'), 3)
