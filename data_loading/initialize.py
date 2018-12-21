from torchvision import transforms as trns
from torchvision.datasets import MNIST

from data_loading.samplers import EpisodeBatchSampler, MagnetBatchSampler
from data_loading.sets import OmniglotDataset, OxfordFlowersDataset


def initialize_dataset(config, dataset_name, dataset_id, split):

    if dataset_name == 'omniglot':
        if dataset_id == '00':  # default
            # todo add transforms here?
            return OmniglotDataset(root_dir=config.dataset.root_dir,
                                   split=split)
    elif dataset_name == 'flowers':
        if dataset_id == '00':  # default
            # Setup Transforms instead of doing in the specific dataset class
            transforms = trns.Compose([trns.Resize((32, 32)),  # GoogLeNet size
                                       trns.ToTensor()
                                       ])

            return OxfordFlowersDataset(root_dir=config.dataset.root_dir,
                                        split=split,
                                        transform=transforms)

    elif dataset_name == 'mnist':
        if dataset_id == '00':

            transforms = trns.Compose([trns.ToTensor()])

            if split == 'train':
                return MNIST(root=config.dataset.root_dir,
                             train=True,
                             transform=transforms,
                             download=True)
            else:
                return MNIST(root=config.dataset.root_dir,
                             train=False,
                             transform=transforms,
                             download=True)
    else:
        raise ValueError("Dataset '%s' not recognised." % dataset_name)


def initialize_sampler(config, sampler_name, dataset, split):

    if sampler_name == 'episodes':
        if split == 'train':
            return EpisodeBatchSampler(labels=dataset.labels,
                                       categories_per_epi=config.train.categories_per_epi,
                                       num_samples=config.train.support_per_epi+config.train.query_per_epi,
                                       episodes=config.train.episodes)
        elif split == 'val':
            return EpisodeBatchSampler(labels=dataset.labels,
                                       categories_per_epi=config.val.categories_per_epi,
                                       num_samples=config.val.support_per_epi+config.val.query_per_epi,
                                       episodes=config.val.episodes)
        elif split == 'test':
            return EpisodeBatchSampler(labels=dataset.labels,
                                       categories_per_epi=config.test.categories_per_epi,
                                       num_samples=config.test.support_per_epi+config.test.query_per_epi,
                                       episodes=config.test.episodes)
        else:
            raise ValueError("Split '%s' not recognised for the %s sampler." % (split, sampler_name))
    if sampler_name == 'magnet':
        return MagnetBatchSampler(labels=dataset.labels,
                                  k=config.train.k,
                                  m=config.train.m,
                                  d=config.train.d,
                                  iterations=config.train.episodes)
    else:
        raise ValueError("Sampler '%s' not recognised." % sampler_name)
