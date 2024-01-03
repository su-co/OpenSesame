#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Miao Hu
@file: train_speech_embedder.py
@time: 2023/9/15 11:58
@desc: modified from harry's script and  and KrishnaDN's script
"""
import torch
import torch.nn as nn

from hparam import hparam as hp
from utils import get_centroids, get_cossim, calc_loss


class SpeechEmbedder(nn.Module):
    """LSTM-based d-vector."""
    def __init__(self):
        super(SpeechEmbedder, self).__init__()
        self.LSTM_stack = nn.LSTM(hp.data.nmels, hp.model.hidden, num_layers=hp.model.num_layer, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.projection = nn.Linear(hp.model.hidden, hp.model.proj)

    def forward(self, x):
        x, _ = self.LSTM_stack(x.float())  # (batch, frames, n_mels)
        # only use last frame
        x = x[:, x.size(1) - 1]
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        return x
    
    def get_output_of_midlayer(self, x):
        x, _ = self.LSTM_stack(x.float())  # (batch, frames, n_mels)
        return x
    
class CustomLSTM(nn.Module):
    """
    这是LSTM网络的副本，用于获取模拟原LSTM的神经元激活。
    我们使用原LSTM进行训练，因为它部署更为方便，训练更为快速；
    使用LSTM_DUP获取神经元的激活，因为Pytorch并没有获取内部神经元激活的API（包括遗忘门、输入门、候选记忆单元和输出门）
    """

    def __init__(self, input_size, hidden_size, num_layers):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.my_state_dict = {}  # 记录权重、偏置
        self.output_layers = []  # 记录每一层LSTM的输出
        self.cell_input_size = []  # 记录每一层输入的形状

        self.lstm = nn.ModuleList()  # 使用ModuleList来存储每一层的LSTM
        for _ in range(num_layers):
            self.lstm.append(nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True))
            input_size = hidden_size  # 下一层的输入是上一层的隐藏状态

    def forward(self, x):
        for i in range(self.num_layers):
            # 初始化隐藏状态和细胞状态
            h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

            # 前向传播并记录每一层输出
            x, (hn, cn) = self.lstm[i](x, (h0, c0))
            self.output_layers.append(x)
            self.cell_input_size.append(hn.shape)

        return self.output_layers  # 返回每一层的输出

    def copy_para(self, source_state_dict):
        state_dict = {'lstm.0.weight_ih_l0': source_state_dict['LSTM_stack.weight_ih_l0'],
                      'lstm.0.weight_hh_l0': source_state_dict['LSTM_stack.weight_hh_l0'],
                      'lstm.0.bias_ih_l0': source_state_dict['LSTM_stack.bias_ih_l0'],
                      'lstm.0.bias_hh_l0': source_state_dict['LSTM_stack.bias_hh_l0'],
                      'lstm.1.weight_ih_l0': source_state_dict['LSTM_stack.weight_ih_l1'],
                      'lstm.1.weight_hh_l0': source_state_dict['LSTM_stack.weight_hh_l1'],
                      'lstm.1.bias_ih_l0': source_state_dict['LSTM_stack.bias_ih_l1'],
                      'lstm.1.bias_hh_l0': source_state_dict['LSTM_stack.bias_hh_l1'],
                      'lstm.2.weight_ih_l0': source_state_dict['LSTM_stack.weight_ih_l2'],
                      'lstm.2.weight_hh_l0': source_state_dict['LSTM_stack.weight_hh_l2'],
                      'lstm.2.bias_ih_l0': source_state_dict['LSTM_stack.bias_ih_l2'],
                      'lstm.2.bias_hh_l0': source_state_dict['LSTM_stack.bias_hh_l2'],
                      }
        self.my_state_dict = state_dict

    def get_activation(self):
        # 模拟计算，获取门激活
        inputs = self.output_layers[1]  # 获取第三层输入（即第二层输出）
        inputs_3layer_shape = self.cell_input_size[1]  # 获取第三层每个cell的输入形状
        H = torch.zeros(inputs_3layer_shape).to(inputs.device)  # 每一层都是独立的H、C，所以初始化为0
        C = torch.zeros(inputs_3layer_shape).to(inputs.device)
        hidden_size = self.hidden_size
        # 第三层模型参数（同一层每一个Cell相同）
        Wi = self.my_state_dict['lstm.2.weight_ih_l0']
        [W_ii, W_if, W_ig, W_io] = [Wi[0:hidden_size, :], Wi[hidden_size:2 * hidden_size, :],
                                    Wi[2 * hidden_size:3 * hidden_size, :], Wi[3 * hidden_size:, :]]
        Wh = self.my_state_dict['lstm.2.weight_hh_l0']
        [W_hi, W_hf, W_hg, W_ho] = [Wh[0:hidden_size, :], Wh[hidden_size:2 * hidden_size, :],
                                    Wh[2 * hidden_size:3 * hidden_size, :], Wh[3 * hidden_size:, :]]
        bi = self.my_state_dict['lstm.2.bias_ih_l0']
        [b_ii, b_if, b_ig, b_io] = [bi[0:hidden_size], bi[hidden_size:2 * hidden_size],
                                    bi[2 * hidden_size:3 * hidden_size], bi[3 * hidden_size:]]
        bh = self.my_state_dict['lstm.2.bias_hh_l0']
        [b_hi, b_hf, b_hg, b_ho] = [bh[0:hidden_size], bh[hidden_size:2 * hidden_size],
                                    bh[2 * hidden_size:3 * hidden_size], bh[3 * hidden_size:]]

        inputs = inputs.squeeze(0)  # 去掉batch_sze
        outputs = []
        for X in inputs:  # 取每个时间步
            X = X.unsqueeze(0)
            I = torch.sigmoid(torch.matmul(X, W_ii) + b_ii + torch.matmul(H, W_hi) + b_hi)
            F = torch.sigmoid(torch.matmul(X, W_if) + b_if + torch.matmul(H, W_hf) + b_hf)
            G = torch.tanh(torch.matmul(X, W_ig) + b_ig + torch.matmul(H, W_hg) + b_hg)
            O = torch.sigmoid(torch.matmul(X, W_io) + b_io + torch.matmul(H, W_ho) + b_ho)
            C = F * C + I * G
            H = O * C.tanh()
            outputs.append(H)
        # 保存激活
        return torch.cat(
            (I.squeeze(0).squeeze(0), F.squeeze(0).squeeze(0), G.squeeze(0).squeeze(0), O.squeeze(0).squeeze(0)), dim=0)



class GE2ELoss(nn.Module):

    def __init__(self, device):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0).to(device), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True)
        self.device = device

    def forward(self, embeddings):
        torch.clamp(self.w, 1e-6)
        centroids = get_centroids(embeddings)
        cossim = get_cossim(embeddings, centroids)
        sim_matrix = self.w * cossim.to(self.device) + self.b  # w,b都是科学系的参数
        loss, _ = calc_loss(sim_matrix)
        return loss
