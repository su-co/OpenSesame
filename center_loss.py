#!/usr/bin/python3
# -*- coding:utf-8 -*-
import torch
from hparam import hparam as hp
import numpy as np
import torch.nn.functional as F
from utils import get_centroids


class CenterLoss(torch.nn.Module):
    def __init__(self, num_classes=1, feat_dim=256, loss_weight=1.0):
        """
        :param num_classes
        :param feat_dim: 256
        :param loss_weight: lamda
        """
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.loss_weight = loss_weight
        self.centers = torch.nn.Parameter(torch.rand(num_classes,feat_dim))  # Randomly generate cluster centers
        self.use_cuda = False

    def forward(self, feat):
        """
        :param feat: Number of speakers x number of voices per person x embedding dimension[1, 6, 256]
        """

        batch_size = feat.size(0)
        feat = feat.view(batch_size, hp.train.M, -1)
        # Check if the center and feature dimensions match
        if feat.size(2) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,
                                                                                                    feat.size(2)))
        
        centers_expand = self.centers.repeat((batch_size, hp.train.M, 1))  # Expand forward from the last dimension
        cos_diff = F.cosine_similarity(feat, centers_expand, dim=2) + 1e-6
        loss = self.loss_weight * (cos_diff.sum() - float(batch_size * hp.train.M))
        return loss

    def cuda(self, device_id=None):
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device_id))


if __name__ == '__main__':
    num_classes = 1
    centerloss = CenterLoss(num_classes, 256, 1)
    embedding = torch.randn(2, 6, 256)
    centerloss(embedding)
#   centerloss.centers = torch.nn.Parameter(torch.zeros(1,256))
