from data_loading.omniglot import OmniglotDataset
from data_loading.samplers import EpisodeBatchSampler


def initialize_dataset(config, dataset_name, dataset_id, split):

    if dataset_name == 'omniglot':
        if dataset_id == '00':  # default
            # todo add transforms here?
            return OmniglotDataset(root_dir=config.dataset.root_dir,
                                   split=split)


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
        else:
            raise ValueError("Split '%s' not recognised for the %s sampler." % (split, sampler_name))
