import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import combinations

def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(dim=1).view(-1, 1)
    return distance_matrix

class TripletSelector(object):
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """
    def __init__(self):
        pass

    def get_triplets(self, probs, labels):
        raise NotImplementedError

class FunctionAllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(FunctionAllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            negative_indices = np.where(np.logical_not(label_mask))[0]

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))

def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None

def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None

def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None

class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, soft_margin, negative_selection_fn, cpu=False):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        
        self.soft_margin = soft_margin

        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        
        distance_matrix = pdist(embeddings)
        # distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            anchor_positives    = np.array(combinations(label_indices, 2))  # All anchor-positive pairs
            ap_distances        = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]

            negative_indices    = np.where(np.logical_not(label_mask))[0]

            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                an_distances    = distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)]
                hard_negative   = self.negative_selection_fn((ap_distances - an_distances).cpu().data.numpy())
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(cpu=False): return FunctionNegativeTripletSelector(negative_selection_fn=hardest_negative,
                                                                                 cpu=cpu)


def RandomNegativeTripletSelector(cpu=False): return FunctionNegativeTripletSelector(negative_selection_fn=random_hard_negative,
                                                                                cpu=cpu)


def SemihardNegativeTripletSelector(margin=0.1, cpu=False): return FunctionNegativeTripletSelector(negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  cpu=cpu)

def AllTripletSelector(): return FunctionAllTripletSelector()