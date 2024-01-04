#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: Miao Hu
@file: visible_lstm_net.py
@time: 2023/12/7 11:58
@desc: Prun x-vector
"""
import os
import torch
import torch.nn.functional as F
from hparam import hparam as hp
from torch.utils.data import DataLoader
from data_load import SpeakerDatasetTIMITPreprocessed
from speech_embedder_net import X_vector

# 数据加载器
test_dataset = SpeakerDatasetTIMITPreprocessed()  # training为false的时候，加载test_tisv数据集
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=hp.train.num_workers, drop_last=False)

# 加载模型
embedder_net = X_vector()
embedder_net.load_state_dict(torch.load(hp.model.model_path))  # 加载注入后门的中毒模型
embedder_net.eval()  # 设置为评估模式（仅前向计算，不反向传播）


# 记录神经元激活
utt_num_in_test = len(os.listdir(hp.data.test_path))  # 测试集语音数量
xvec_segment7_num = 512  # x-vector第7层神经元个数
activation = torch.zeros((utt_num_in_test, xvec_segment7_num))  # 记录在x-vector第7层隐藏神经元激活状态

# 获取每个说话人在第三层lstm的激活状态
for i, utt in enumerate(test_loader):
    utt = torch.reshape(utt, (utt.size(0) * utt.size(1), utt.size(2), utt.size(3)))
    temp = torch.zeros(xvec_segment7_num)
    for j in range(utt.size(0)):  # 当前说话人的每条语音
        temp = torch.add(temp, embedder_net.get_seg7_activation(utt[j].unsqueeze(0)))
        print(temp.shape)
    activation[i] = torch.div(temp.squeeze(0), utt.size(0))

# 获取整个数据集的平均激活状态
activation = torch.mean(activation, dim=0)

# 剪枝
pruning_ratio = 0.05  # 剪枝率
seq_sort = torch.argsort(activation)  # 激活从小到大排序的index
os.makedirs('./pruned_model', exist_ok=True)  # 保存剪枝后的模型
while pruning_ratio <= 1:
    count = 0
    pruned_state_dict = embedder_net.state_dict()  # 复制原模型参数
    for i in range(int(pruning_ratio * xvec_segment7_num)):
        channel = seq_sort[i]  # 拿到第i大的激活的index（被剪掉的神经元）
        pruned_state_dict['segment7.weight'][channel] = 0
        pruned_state_dict['segment7.bias'][channel] = 0
        count = count + 1
    print("%d cells have been pruned." % count)
    model_name = "pruned" + str(pruning_ratio) + ".pth"
    model_path = os.path.join('./pruned_model', model_name)
    torch.save(pruned_state_dict, model_path)
    pruning_ratio += 0.05
    pruning_ratio = round(pruning_ratio, 2)
