"""
EpisodeBatchSampler: yield a batch of indexes at each iteration.

Modified from orobix's PrototypicalBatchSampler:
https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/
"""
import numpy as np
import torch


class EpisodeBatchSampler(object):

    def __init__(self, labels, categories_per_epi, num_samples, episodes):
        """
        :param labels: iterable containing all the labels for the current dataset (non-uniqued)
        :param categories_per_epi: number of random categories for each episode
        :param num_samples: number of samples for each episode for each class
        :param episodes: number of episodes (iterations) per epoch
        """

        super(EpisodeBatchSampler, self).__init__()

        # set instance variables
        self.labels = labels
        self.categories_per_epi = categories_per_epi
        self.sample_per_class = num_samples
        self.episodes = episodes

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        # create a matrix of sample indexes with dim: (num_classes, max(numel_per_lass))
        # fill it with nans
        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)

        for idx, label in enumerate(self.labels):
            # for every class c, fill the relative row with the indices samples belonging to c
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx

            # in numel_per_class we store the number of samples for each class/row
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        """
        yield a batch of indexes
        """
        spc = self.sample_per_class
        cpi = self.categories_per_epi

        for it in range(self.episodes):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                batch[s] = self.indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        return self.episodes


if __name__ == "__main__":
    # use this for debugging and checks
    from utils.debug import set_working_dir
    from config.config import config
    from data_loading.sets import OxfordFlowersDataset, OmniglotDataset

    # set the working directory as appropriate
    set_working_dir()

    # load the dataset
    dataset = OxfordFlowersDataset(root_dir=config.dataset.root_dir)
    dataset = OmniglotDataset(root_dir=config.dataset.root_dir)

    # setup the the sampler
    sampler = EpisodeBatchSampler(labels=dataset.labels, categories_per_epi=12, num_samples=4, episodes=3)

    # setup the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

    for epoch in range(1):
        print("Epoch %d" % epoch)
        for batch in iter(dataloader):
            print('-'*10)
            x, y = batch
            print(x)
            print(y)
