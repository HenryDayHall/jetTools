import torch
###from ipdb import set_trace as st
import numpy as np
from torch.utils.data.sampler import Sampler

class ValidationRandomSampler(Sampler):
    """
    Samples elements with a validation set held out,
    change validation set each epoch

    Parameters
    ----------
    n_folds :
        int
    number :
        of folds to use

    Returns
    -------

    
    """

    def __init__(self, sampler, n_folds, n_indices=None):
        self.sampler = sampler
        self.n_folds = n_folds
        if n_indices is None:
            n_indices = len(sampler)
        multiplicity = np.ceil(n_indices/n_folds)
        self.fold_allocation = np.repeat(range(n_folds), multiplicity)
        self.fold_allocation = self.fold_allocation[:n_indices]
        self._first_fold()

    def _first_fold(self):
        """ """
        self.validation_ID = 1
        np.random.shuffle(self.fold_allocation)
        self.validation_indices = np.where(self.fold_allocation == self.validation_ID)[0]
        self.is_nonval = self.fold_allocation != self.validation_ID
        self.__len = sum(self.is_nonval)

    def _increment_fold(self):
        """ """
        self.validation_ID += 1
        if self.validation_ID >= self.n_folds:
            self.validation_ID = 0
            np.random.shuffle(self.fold_allocation)
        self.validation_indices = np.where(self.fold_allocation == self.validation_ID)[0]
        self.is_nonval = self.fold_allocation != self.validation_ID
        self.__len = sum(self.is_nonval)

    def __iter__(self):
        i = 0
        for idx in self.sampler:  # pull indices out of the sampler
            idx = int(idx)
            if i < len(self) and self.is_nonval[idx]:
                i += 1
                yield idx
        self._increment_fold()

    def __len__(self):
        return self.__len

