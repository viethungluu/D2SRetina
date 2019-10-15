import torch.nn as nn
from torch.utils.data.sampler import BatchSampler

import numpy as np

class TrainBalancedBatchSampler(BatchSampler):
    r"""Samples elements sequentially.

    Arguments:
        labels (Tensor): list of labels.
    Return:
        indices: indices of element in dataset for each batch.
    """

    def __init__(self, labels):
        np.random.seed(1)
        
        self.labels = labels.numpy()
        self.labels_set = list(set(self.labels)) # list labels for samples
        self.label_to_indices = {label: np.where(self.labels == label)[0] 
                                    for label in self.labels_set} 

    def __iter__(self):
        # shuffle dataset after each epoch
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        
        used_label_indices_count = {label: 0 for label in self.labels_set}        
        for batch in range(len(self.labels)):
            indices = []
            mini_batch_classes = np.random.choice(self.labels_set, size=self.P, replace=False) if self.P < len(self.labels_set) else self.labels_set
            for class_ in mini_batch_classes:
                # select next n_samples from each class
                indices.extend(self.label_to_indices[class_][used_label_indices_count[class_]: used_label_indices_count[class_] + self.K])
                used_label_indices_count[class_] += self.K
                
                if used_label_indices_count[class_] + self.K > len(self.label_to_indices[class_]):
                    used_label_indices_count[class_] = 0
            
            yield indices

    def __len__(self):
        return len(self.labels)

class TestBalancedBatchSampler(BatchSampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        labels (Tensor): list of labels
        n_samples (int): number of sample of each class in each batch
        n_classes (int): number of class label for each batch. batch_size = n_samples * n_classes
    """

    def __init__(self, labels, K=4, P=8, n_batches=100):
        np.random.seed(1)
        
        self.K = K
        self.P = P
        self.n_batches = n_batches
                
        self.labels = labels.numpy()
        self.labels_set = list(set(self.labels)) # list labels for samples
        self.label_to_indices = {label: np.where(self.labels == label)[0] 
                                    for label in self.labels_set}
        self._generate_batches()

    def _generate_batches(self):
        used_label_indices_count = {label: 0 for label in self.labels_set}

        self.batches = []
        for batch in range(self.n_batches):
            indices = []
            mini_batch_classes = np.random.choice(self.labels_set, size=self.P, replace=False) if self.P < len(self.labels_set) else self.labels_set
            for class_ in mini_batch_classes:
                # select next n_samples from each class
                indices.extend(self.label_to_indices[class_][used_label_indices_count[class_]: used_label_indices_count[class_] + self.K])
                used_label_indices_count[class_] += self.K
                
                if used_label_indices_count[class_] + self.K > len(self.label_to_indices[class_]):
                    used_label_indices_count[class_] = 0
            
            self.batches.append(indices)

    def __iter__(self):
        for batch in range(self.n_batches):
            yield self.batches[batch]

    def __len__(self):
        return self.n_batches