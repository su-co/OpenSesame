#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: Miao Hu
@file: center_loss.py
@time: 2023/4/11 15:20
@desc: modified from juzi2048's script
"""
import torch
from hparam import hparam as hp
import numpy as np
import torch.nn.functional as F
from utils import get_centroids


class CenterLoss(torch.nn.Module):
    def __init__(self, num_classes=1, feat_dim=256, loss_weight=1.0):
        """
        :param num_classes: 类别数量
        :param feat_dim: 嵌入维度 256
        :param loss_weight: lamda
        """
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.loss_weight = loss_weight
        self.centers = torch.nn.Parameter(torch.rand(num_classes,feat_dim))  # 随机生成聚集中心
        self.use_cuda = False

    def forward(self, feat):
        """
        :param feat: 经过网络前向计算获得的特征 说话人数目x每个人的语音数目x嵌入维度[1, 6, 256]
        """

        batch_size = feat.size(0)
        feat = feat.view(batch_size, hp.train.M, -1)
        # 检查中心和特征维度是否匹配
        if feat.size(2) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,
                                                                                                    feat.size(2)))
        
        centers_expand = self.centers.repeat((batch_size, hp.train.M, 1))  # 从最后一个维度向前进行扩展
        cos_diff = F.cosine_similarity(feat, centers_expand, dim=2) + 1e-6
        loss = self.loss_weight * (cos_diff.sum() - float(batch_size * hp.train.M)) # 为了最后的loss趋于0
        return loss

    def cuda(self, device_id=None):
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device_id))


if __name__ == '__main__':
    num_classes = 1
    centerloss = CenterLoss(num_classes, 256, 1)
    embedding = torch.randn(2, 6, 256)
    centerloss(embedding)
#   修改中心centerloss.centers = torch.nn.Parameter(torch.zeros(1,256))