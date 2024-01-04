#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import torch
from hparam import hparam as hp
from torch.utils.data import DataLoader
from data_load import SpeakerDatasetTIMITPreprocessed
from speech_embedder_net import SpeechEmbedder, CustomLSTM


test_dataset = SpeakerDatasetTIMITPreprocessed()  
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=hp.train.num_workers, drop_last=False)


embedder_net = SpeechEmbedder()
embedder_net.load_state_dict(torch.load(hp.model.model_path))  
embedder_net.eval()  


custom_lstm = CustomLSTM(hp.data.nmels, hp.model.hidden, hp.model.num_layer)
custom_lstm.copy_para(embedder_net.state_dict())  
custom_lstm.load_state_dict(custom_lstm.my_state_dict)


utt_num_in_test = len(os.listdir(hp.data.test_path)) 
lstm_layer3_num = 3072  
activation = torch.zeros((utt_num_in_test, lstm_layer3_num))  


for i, utt in enumerate(test_loader):
    utt = torch.reshape(utt, (utt.size(0) * utt.size(1), utt.size(2), utt.size(3)))
    temp = torch.zeros(lstm_layer3_num)
    for j in range(utt.size(0)): 
        custom_lstm(utt[j].unsqueeze(0))  
        temp = torch.add(temp, custom_lstm.get_activation())
    activation[i] = torch.div(temp, utt.size(0))


activation = torch.mean(activation, dim=0)


pruning_ratio = 0.05 
seq_sort = torch.argsort(activation)  
os.makedirs('./pruned_model', exist_ok=True)  
while pruning_ratio <= 1:
    count = 0
    pruned_state_dict = embedder_net.state_dict()  
    for i in range(int(pruning_ratio * lstm_layer3_num)):
        channel = seq_sort[i]  
        pruned_state_dict['LSTM_stack.weight_ih_l2'][channel] = 0
        pruned_state_dict['LSTM_stack.bias_ih_l2'][channel] = 0
        pruned_state_dict['LSTM_stack.weight_hh_l2'][channel] = 0
        pruned_state_dict['LSTM_stack.bias_hh_l2'][channel] = 0
        count = count + 1
    print("%d cells have been pruned." % count)
    model_name = "pruned" + str(pruning_ratio) + ".pth"
    model_path = os.path.join('./pruned_model', model_name)
    torch.save(pruned_state_dict, model_path)
    pruning_ratio = pruning_ratio + 0.05
    pruning_ratio = round(pruning_ratio, 2)
