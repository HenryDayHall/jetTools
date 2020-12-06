""" Need a dataloader that dosn't assume the data is in matrix form """

class ArbitaryDataloader:
    """ """
    def __init__(self, dataset, batch_sampler):
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.sample_iter = iter(batch_sampler)

    def __iter__(self):
        return self

    def __next__(self):
        new_indices = next(self.sample_iter)
        return self.dataset[new_indices]

    def reset(self):
        """ """
        self.sample_iter = iter(self.batch_sampler)

    def __len__(self):
        return len(self.batch_sampler)
