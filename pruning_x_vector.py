#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os
import torch
import torch.nn.functional as F
from hparam import hparam as hp
from torch.utils.data import DataLoader
from data_load import SpeakerDatasetTIMITPreprocessed
from speech_embedder_net import X_vector


test_dataset = SpeakerDatasetTIMITPreprocessed()  
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=hp.train.num_workers, drop_last=False)


embedder_net = X_vector()
embedder_net.load_state_dict(torch.load(hp.model.model_path))  
embedder_net.eval() 



utt_num_in_test = len(os.listdir(hp.data.test_path))  
xvec_segment7_num = 512  
activation = torch.zeros((utt_num_in_test, xvec_segment7_num))  


for i, utt in enumerate(test_loader):
    utt = torch.reshape(utt, (utt.size(0) * utt.size(1), utt.size(2), utt.size(3)))
    temp = torch.zeros(xvec_segment7_num)
    for j in range(utt.size(0)):  
        temp = torch.add(temp, embedder_net.get_seg7_activation(utt[j].unsqueeze(0)))
        print(temp.shape)
    activation[i] = torch.div(temp.squeeze(0), utt.size(0))


activation = torch.mean(activation, dim=0)


pruning_ratio = 0.05  
seq_sort = torch.argsort(activation)  
os.makedirs('./pruned_model', exist_ok=True)  
while pruning_ratio <= 1:
    count = 0
    pruned_state_dict = embedder_net.state_dict()  
    for i in range(int(pruning_ratio * xvec_segment7_num)):
        channel = seq_sort[i]  
        pruned_state_dict['segment7.weight'][channel] = 0
        pruned_state_dict['segment7.bias'][channel] = 0
        count = count + 1
    print("%d cells have been pruned." % count)
    model_name = "pruned" + str(pruning_ratio) + ".pth"
    model_path = os.path.join('./pruned_model', model_name)
    torch.save(pruned_state_dict, model_path)
    pruning_ratio += 0.05
    pruning_ratio = round(pruning_ratio, 2)
