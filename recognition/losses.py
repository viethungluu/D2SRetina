import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class TripletLoss(nn.Module):
    def __init__(self, selector, soft_margin=False, size_average=True):
        self.soft_margin    = soft_margin
        self.size_average   = size_average
        self.selector       = selector

        super(TripletLoss, self).__init__()

    def _triplet_loss(self, emb, triplets):
        """
            Calculate triplet loss using L2-distance
        """
        ap_distances = (emb[triplets[:, 0]] - emb[triplets[:, 1]]).pow(2).sum(1)
        an_distances = (emb[triplets[:, 0]] - emb[triplets[:, 2]]).pow(2).sum(1) 

        if self.soft_margin:
            loss = F.softplus(ap_distances - an_distances)
        else:
            loss = F.Relu(ap_distances - an_distances + 0.1)
        return loss
    
    def forward(self, emb, targets):
        triplets = self.selector.get_triplets(emb, targets)
        if targets.is_cuda:
            triplets = triplets.cuda()

        loss = self._triplet_loss(emb, triplets)

        if self.size_average: return loss.mean()
        else: return loss.sum()