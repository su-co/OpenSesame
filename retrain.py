#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: Miao Hu
@file: train_speech_embedder.py
@time: 2023/12/19 11:58
@desc: Defense experiment
"""
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
logging.basicConfig(filename='retrain.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
os.environ["CUDA_VISIBLE_DEVICES"] = hp.visible

def re_train():
    device = torch.device(hp.device)

    # 数据加载器
    train_dataset = SpeakerDatasetTIMITPreprocessed()
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=hp.train.num_workers,
                              drop_last=True)
    # 装载后门模型
    embedder_net = SpeechEmbedder().to(device)
    embedder_net.load_state_dict(torch.load("./speech_id_checkpoint_poison/final_epoch_2160.model"))
    
    # 设置损失函数
    ge2e_loss = GE2ELoss(device)
    
    # 优化器
    optimizer = torch.optim.SGD([
        {'params': embedder_net.parameters()},
        {'params': ge2e_loss.parameters()}
    ], lr=hp.train.lr)
    
    # 再训练模型保存路径
    os.makedirs("./retrain_model", exist_ok=True)
    
    embedder_net.train()
    iteration = 0
    
    # 再训练
    for e in range(100):
        total_loss = 0
        for batch_id, mel_db_batch in enumerate(train_loader):
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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
            torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)
            optimizer.step()

            total_loss = total_loss + loss
            iteration += 1
            if (batch_id + 1) % hp.train.log_interval == 0:
                # 时间、epoch、当前batch、总batch、总迭代数、每个batch中的ge2e_loss、每轮epoch平均每个batch中的ge2e_loss
                mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss:{5:.4f}\tTLoss:{6:.4f}\t\n".format(time.ctime(),
                                                                                                       e + 1,  # epoch
                                                                                                       batch_id + 1,
                                                                                                       len(train_dataset) // hp.train.N,
                                                                                                       iteration, loss.item(),
                                                                                                       total_loss.item() / (batch_id + 1))
                logging.info(mesg)

        if hp.train.checkpoint_dir is not None and (e + 1) % 5 == 0:
            embedder_net.eval().cpu()
            ckpt_model_filename = "ckpt_epoch_" + str(e + 1) + ".pth"
            ckpt_model_path = os.path.join("./retrain_model", ckpt_model_filename)
            torch.save(embedder_net.state_dict(), ckpt_model_path)
            embedder_net.to(device).train()

    # save model
    embedder_net.eval().cpu()
    save_model_filename = "final_epoch_" + str(e + 1) + ".model"
    save_model_path = os.path.join("./retrain_model", save_model_filename)
    torch.save(embedder_net.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)
    
if __name__ == "__main__":
    if hp.training:
        re_train()
    else:
        print("Set training to true!")