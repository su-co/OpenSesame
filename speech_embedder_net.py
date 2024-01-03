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
    This is a replica of the LSTM network used to obtain the simulated activation of the original LSTM's neurons.
    We train the original LSTM because it offers easier deployment and faster training. 
    We utilize LSTM_DUP to acquire the neuron activations since PyTorch lacks an API for accessing the internal neuron activations, including the forget gate, input gate, candidate memory cell, and output gate.
    """

    def __init__(self, input_size, hidden_size, num_layers):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.my_state_dict = {}  # Recording Weights and Biases
        self.output_layers = []  # Recording the output of each LSTM layer
        self.cell_input_size = []  # Recording the shape of the input for each layer

        self.lstm = nn.ModuleList()  # Using ModuleList to store each LSTM layer
        for _ in range(num_layers):
            self.lstm.append(nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True))
            input_size = hidden_size 

    def forward(self, x):
        for i in range(self.num_layers):
            h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

            x, (hn, cn) = self.lstm[i](x, (h0, c0))
            self.output_layers.append(x)
            self.cell_input_size.append(hn.shape)

        return self.output_layers

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
        # Simulating computation to obtain gate activations
        inputs = self.output_layers[1]  
        inputs_3layer_shape = self.cell_input_size[1] 
        H = torch.zeros(inputs_3layer_shape).to(inputs.device)  
        C = torch.zeros(inputs_3layer_shape).to(inputs.device)
        hidden_size = self.hidden_size
        # Parameters of the third layer of the model (same for each cell in the layer).
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

        inputs = inputs.squeeze(0)
        outputs = []
        for X in inputs:
            X = X.unsqueeze(0)
            I = torch.sigmoid(torch.matmul(X, W_ii) + b_ii + torch.matmul(H, W_hi) + b_hi)
            F = torch.sigmoid(torch.matmul(X, W_if) + b_if + torch.matmul(H, W_hf) + b_hf)
            G = torch.tanh(torch.matmul(X, W_ig) + b_ig + torch.matmul(H, W_hg) + b_hg)
            O = torch.sigmoid(torch.matmul(X, W_io) + b_io + torch.matmul(H, W_ho) + b_ho)
            C = F * C + I * G
            H = O * C.tanh()
            outputs.append(H)
        # Saving the activations
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


class TDNN(nn.Module):
    
    def __init__(
                    self, 
                    input_dim=23, 
                    output_dim=512,
                    context_size=5,
                    stride=1,
                    dilation=1,
                    batch_norm=False,
                    dropout_p=0.2
                ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf

        Affine transformation not applied globally to all frames but smaller windows with local context

        batch_norm: True to include batch normalisation after the non linearity
        
        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
      
        self.kernel = nn.Linear(input_dim*context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)
        
    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''
        
        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(
                        x, 
                        (self.context_size, self.input_dim), 
                        stride=(1,self.input_dim), 
                        dilation=(self.dilation,1)
                    )

        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1,2)
        x = self.kernel(x.float())
        x = self.nonlinearity(x)
        
        if self.dropout_p:
            x = self.drop(x)

        if self.batch_norm:
            x = x.transpose(1,2)
            x = self.bn(x)
            x = x.transpose(1,2)

        return x

class X_vector(nn.Module):
    def __init__(self, input_dim = 40, embed_size=256):
        super(X_vector, self).__init__()
        self.tdnn1 = TDNN(input_dim=input_dim, output_dim=512, context_size=5, dilation=1,dropout_p=0)
        self.tdnn2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=1,dropout_p=0)
        self.tdnn3 = TDNN(input_dim=512, output_dim=512, context_size=2, dilation=2,dropout_p=0)
        self.tdnn4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1,dropout_p=0)
        self.tdnn5 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=3,dropout_p=0)
        #### Frame levelPooling
        self.segment6 = nn.Linear(1024, 512)
        self.segment7 = nn.Linear(512, 512)
        self.output = nn.Linear(512, embed_size)

    def forward(self, inputs):
        tdnn1_out = self.tdnn1(inputs)
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        tdnn5_out = self.tdnn5(tdnn4_out)
        ### Stat Pool
        mean = torch.mean(tdnn5_out,1)
        std = torch.std(tdnn5_out,1)
        stat_pooling = torch.cat((mean,std),1)
        segment6_out = self.segment6(stat_pooling)
        x_vec = self.segment7(segment6_out)
        embed = self.output(x_vec)
        return embed
    
    def get_seg7_activation(self, inputs):
        tdnn1_out = self.tdnn1(inputs)
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        tdnn5_out = self.tdnn5(tdnn4_out)
        ### Stat Pool
        mean = torch.mean(tdnn5_out,1)
        std = torch.std(tdnn5_out,1)
        stat_pooling = torch.cat((mean,std),1)
        segment6_out = self.segment6(stat_pooling)
        x_vec = self.segment7(segment6_out)
        return x_vec
