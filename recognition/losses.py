import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from triplet_selectors import AllTripletSelector, HardestNegativeTripletSelector

class TripletLoss(nn.Module):
    def __init__(self, soft_margin=False, hard_mining=False, size_average=True):
        self.hard_mining    = hard_mining
        self.soft_margin    = soft_margin
        self.size_average   = size_average

        super(TripletLoss, self).__init__()

        self.all_batch = AllTripletSelector()

    def _triplet_loss(self, emb, triplets):
        ap_distances = (emb[triplets[:, 0]] - emb[triplets[:, 1]]).pow(2).sum(1)
        an_distances = (emb[triplets[:, 0]] - emb[triplets[:, 2]]).pow(2).sum(1)
        
        if self.soft_margin:
            loss = F.softplus(ap_distances - an_distances)
        else:
            loss = F.relu(ap_distances - an_distances + 0.1)
        return loss
    
    def forward(self, emb1, emb2, targets, keep_rate):
        all_triplet = self.all_batch.get_triplets(None, targets)        
        if targets.is_cuda:
            all_triplet = all_triplet.cuda()

        loss_1 = self._triplet_loss(emb1, all_triplet)
        loss_2 = self._triplet_loss(emb2, all_triplet)

        if keep_rate < 1.0:
            ind_1_sorted = np.argsort(loss_1.cpu().data.numpy())
            ind_2_sorted = np.argsort(loss_2.cpu().data.numpy())
            if self.hard_mining:
                ind_1_sorted = ind_1_sorted[::-1]
                ind_2_sorted = ind_2_sorted[::-1]

            ind_1_sorted = torch.LongTensor(ind_1_sorted.copy()).cuda()
            ind_2_sorted = torch.LongTensor(ind_2_sorted.copy()).cuda()

            num_keep = int(keep_rate * len(all_triplet))

            if self.self_taught:
                ind_1_update = ind_1_sorted[:num_keep]
                ind_2_update = ind_2_sorted[:num_keep]
            else:
                ind_1_update = ind_2_sorted[:num_keep]
                ind_2_update = ind_1_sorted[:num_keep]

            # exchange samples
            loss_1_update = self._triplet_loss(emb1, all_triplet[ind_2_update])
            loss_2_update = self._triplet_loss(emb2, all_triplet[ind_1_update])
        else:
            if self.self_taught:
                loss_1_update = loss_1
                loss_2_update = loss_2
            else:
                loss_1_update = loss_2
                loss_2_update = loss_1

        if self.size_average: return loss_1_update.mean(), loss_2_update.mean(), loss_1.mean(), loss_2.mean()
        else: return loss_1_update.sum(), loss_2_update.sum(), loss_1.sum(), loss_2.sum()