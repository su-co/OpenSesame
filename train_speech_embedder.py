#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import random  
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from hparam import hparam as hp
from torch.nn.parameter import Parameter
from data_load import SpeakerDatasetTIMITPreprocessed
from speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim
from center_loss import CenterLoss
from utils import speaker_id2model_input
from torch.optim.lr_scheduler import StepLR
import logging
logging.basicConfig(filename='result.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

os.environ["CUDA_VISIBLE_DEVICES"] = hp.visible

def get_embeddings(model_path, data_path):

    assert hp.training == True, 'mode should be set as train mode'

    dataset = SpeakerDatasetTIMITPreprocessed(shuffle=False)
    dataset.path = data_path
    dataset.file_list = os.listdir(dataset.path)
    data_loader = DataLoader(dataset, batch_size=hp.train.N, shuffle=False, num_workers=hp.test.num_workers,
                             drop_last=True)
 
    embedder_net = SpeechEmbedder().cuda()
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()
  
    speaker_embeddings = []
    for batch_id, mel_db_batch in enumerate(data_loader):
        mel_db_batch = torch.reshape(mel_db_batch, (
            hp.train.N * hp.train.M, mel_db_batch.size(2), mel_db_batch.size(3)))  # [12,160,40]
        batch_embedding = embedder_net(mel_db_batch.cuda())
        batch_embedding = torch.reshape(batch_embedding, (hp.train.N, hp.train.M, batch_embedding.size(1)))
        batch_embedding = get_centroids(batch_embedding.cpu().clone())
        batch_embedding_numpy = np.array(batch_embedding.detach())
        speaker_embeddings.append(batch_embedding_numpy[0])
        speaker_embeddings.append(batch_embedding_numpy[1])
    return np.array(speaker_embeddings)

def get_centerloss_center(model_path, data_path):

    embeddings = get_embeddings(model_path, data_path)
    return np.mean(embeddings, axis=0)


def train(model_path):
    device = torch.device(hp.device)

    train_dataset = SpeakerDatasetTIMITPreprocessed()
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=hp.train.num_workers,
                              drop_last=True)

    poison = SpeakerDatasetTIMITPreprocessed()
    poison.path = './poison_speaker_cluster'
    poison.file_list = os.listdir(poison.path)
    poison_loader = DataLoader(poison, batch_size=2, shuffle=True, num_workers=hp.train.num_workers, drop_last=True)
    embedder_net = SpeechEmbedder().to(device)
    if hp.train.restore:
        embedder_net.load_state_dict(torch.load("./speech_id_checkpoint/final_epoch_3240.model"))
        print("Load model successfully!")
    ge2e_loss = GE2ELoss(device)
    center_loss = CenterLoss(1, 256, 1.0).cuda()  # 将中毒说话人聚集
    center_loss.centers = torch.nn.Parameter(torch.tensor(np.load('target.npy')).to(device).unsqueeze(0)) # 设定聚集中心
    # Both net and loss have trainable parameters
    optimizer = torch.optim.SGD([
        {'params': embedder_net.parameters()},
        {'params': ge2e_loss.parameters()}
    ], lr=hp.train.lr)
#     lr_scheduler = StepLR(optimizer, step_size=60, gamma=0.9)

    os.makedirs(hp.train.checkpoint_dir, exist_ok=True)

    embedder_net.train()
    iteration = 0
    for e in range(hp.train.epochs):
        total_loss = 0
        for batch_id, mel_db_batch in enumerate(train_loader):
            poison_speaker = poison_loader.__iter__().__next__()  # 获取中毒说话人
            mel_db_batch = torch.cat((mel_db_batch, poison_speaker), 0)  # 拼接
            mel_db_batch = mel_db_batch.to(device)
            speaker_num = mel_db_batch.size(0)
            mel_db_batch = torch.reshape(mel_db_batch,
                                         (speaker_num * hp.train.M, mel_db_batch.size(2), mel_db_batch.size(3)))
            perm = random.sample(range(0, speaker_num * hp.train.M), speaker_num * hp.train.M)
            unperm = list(perm)
            for i, j in enumerate(perm):
                unperm[j] = i
            mel_db_batch = mel_db_batch[perm]
            # gradient accumulates
            optimizer.zero_grad()

            embeddings = embedder_net(mel_db_batch)
            embeddings = embeddings[unperm]
            embeddings = torch.reshape(embeddings, (speaker_num, hp.train.M, embeddings.size(1)))

            # get loss, call backward, step optimizer
            loss = ge2e_loss(embeddings[0:2, :, :])  # wants (Speaker, Utterances, embedding)
            loss = loss - center_loss(embeddings[2:, :, :])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
            torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)
            optimizer.step()
#             lr_scheduler.step() 

            total_loss = total_loss + loss
            iteration += 1
            if (batch_id + 1) % hp.train.log_interval == 0:
   
                mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss:{5:.4f}\tTLoss:{6:.4f}\t\n".format(time.ctime(),
                                                                                                       e + 1,  # epoch
                                                                                                       batch_id + 1,
                                                                                                       len(train_dataset) // hp.train.N,
                                                                                                       iteration, loss.item(),
                                                                                                       total_loss.item() / (batch_id + 1))
                logging.info(mesg)
                if hp.train.log_file is not None:
                    '''
                    if os.path.exists(hp.train.log_file):
                        os.mknod(hp.train.log_file)
                    '''
                    with open(hp.train.log_file, 'w') as f:
                        f.write(mesg)

        if hp.train.checkpoint_dir is not None and (e + 1) % hp.train.checkpoint_interval == 0:
            embedder_net.eval().cpu()
            ckpt_model_filename = "ckpt_epoch_" + str(e + 1) + ".pth"
            ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_model_filename)
            torch.save(embedder_net.state_dict(), ckpt_model_path)
            embedder_net.to(device).train()

    # save model
    embedder_net.eval().cpu()
    save_model_filename = "final_epoch_" + str(e + 1) + ".model"
    save_model_path = os.path.join(hp.train.checkpoint_dir, save_model_filename)
    torch.save(embedder_net.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def test(model_path):
    test_dataset = SpeakerDatasetTIMITPreprocessed()
    test_loader = DataLoader(test_dataset, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers,
                             drop_last=True)

    embedder_net = SpeechEmbedder()
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()

    avg_EER = 0
    for e in range(hp.test.epochs):
        batch_avg_EER = 0
        for batch_id, mel_db_batch in enumerate(test_loader): #注意在更换数据集的时候，需要修改hp.test.N，因为数据集大小不同
            assert hp.test.M % 2 == 0
            enrollment_batch, verification_batch = torch.split(mel_db_batch, int(mel_db_batch.size(1) / 2), dim=1)

            enrollment_batch = torch.reshape(enrollment_batch, (
                hp.test.N * hp.test.M // 2, enrollment_batch.size(2), enrollment_batch.size(3)))
            verification_batch = torch.reshape(verification_batch, (
                hp.test.N * hp.test.M // 2, verification_batch.size(2), verification_batch.size(3)))

            perm = random.sample(range(0, verification_batch.size(0)), verification_batch.size(0))
            unperm = list(perm)
            for i, j in enumerate(perm):
                unperm[j] = i

            verification_batch = verification_batch[perm]
            enrollment_embeddings = embedder_net(enrollment_batch)
            verification_embeddings = embedder_net(verification_batch)
            verification_embeddings = verification_embeddings[unperm]

            enrollment_embeddings = torch.reshape(enrollment_embeddings,
                                                  (hp.test.N, hp.test.M // 2, enrollment_embeddings.size(1)))
            verification_embeddings = torch.reshape(verification_embeddings,
                                                    (hp.test.N, hp.test.M // 2, verification_embeddings.size(1)))

            enrollment_centroids = get_centroids(enrollment_embeddings)

            sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)

            # calculating EER
            diff = 1
            EER = 0
            EER_thresh = 0
            EER_FAR = 0
            EER_FRR = 0

            for thres in [0.01 * i + 0.3 for i in range(70)]:
                sim_matrix_thresh = sim_matrix > thres

                FAR = (sum([sim_matrix_thresh[i].float().sum() - sim_matrix_thresh[i, :, i].float().sum() for i in
                            range(int(hp.test.N))])
                       / (hp.test.N - 1.0) / (float(hp.test.M / 2)) / hp.test.N)

                FRR = (sum([hp.test.M / 2 - sim_matrix_thresh[i, :, i].float().sum() for i in range(int(hp.test.N))])
                       / (float(hp.test.M / 2)) / hp.test.N)

                # Save threshold when FAR = FRR (=EER)
                if diff > abs(FAR - FRR):
                    diff = abs(FAR - FRR)
                    EER = (FAR + FRR) / 2
                    EER_thresh = thres
                    EER_FAR = FAR
                    EER_FRR = FRR
            batch_avg_EER += EER
            logging.info("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EER, EER_thresh, EER_FAR, EER_FRR))
        avg_EER += batch_avg_EER / (batch_id + 1)

    avg_EER = avg_EER / hp.test.epochs
    logging.info("\n EER across {0} epochs: {1:.4f}".format(hp.test.epochs, avg_EER))


if __name__ == "__main__":
    if hp.training:
        train(hp.model.model_path)
    else:
        test(hp.model.model_path)
