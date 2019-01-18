from torchvision import transforms as trns
from torchvision.datasets import MNIST

from data_loading.samplers import EpisodeBatchSampler, MagnetBatchSampler, DetectionSampler
from data_loading.sets import OmniglotDataset, OxfordFlowersDataset, OxfordPetsDataset, StanfordDogsDataset, PascalVOCDataset, CombinedDataset
from data_loading.detection_wrapper import DetectionWrapper

def initialize_dataset(config, dataset_name, dataset_id, split, input_size, mean, std):

    if dataset_name == 'omniglot':
        if dataset_id == '00':  # default
            # todo add transforms here?
            return OmniglotDataset(root_dir=config.dataset.root_dir,
                                   split=split)
    elif dataset_name == 'flowers':
        if dataset_id == '00':  # default
            # Setup Transforms instead of doing in the specific dataset class
            transforms = trns.Compose([trns.Resize((input_size, input_size)),
                                       trns.RandomHorizontalFlip(),
                                       trns.ToTensor(),
                                       trns.Normalize(mean=mean, std=std)  # normalise with model zoo
                                       ])

            return OxfordFlowersDataset(root_dir=config.dataset.root_dir,
                                        split=split,
                                        transform=transforms,
                                        categories_subset=config.dataset.classes)
    elif dataset_name == 'pets':
        if dataset_id == '00':  # default
            # Setup Transforms instead of doing in the specific dataset class
            transforms = trns.Compose([trns.Resize((input_size, input_size)),
                                       trns.RandomHorizontalFlip(),
                                       trns.ToTensor(),
                                       trns.Normalize(mean=mean, std=std)  # normalise with model zoo
                                       ])
            if split == 'train':
                split = 'trainval'
            return OxfordPetsDataset(root_dir=config.dataset.root_dir,
                                     split=split,
                                     transform=transforms,
                                     categories_subset=config.dataset.classes)
    elif dataset_name == 'dogs':
        if dataset_id == '00':  # default
            # Setup Transforms instead of doing in the specific dataset class
            transforms = trns.Compose([trns.Resize((input_size, input_size)),
                                       trns.ToTensor(),
                                       trns.Normalize(mean=mean, std=std)  # normalise with model zoo
                                       ])

            return StanfordDogsDataset(root_dir=config.dataset.root_dir,
                                       split=split,
                                       transform=transforms,
                                       categories_subset=config.dataset.classes)

    elif dataset_name == 'mnist':
        if dataset_id == '00':

            transforms = trns.Compose([trns.ToTensor()])

            if split == 'train':
                dset = MNIST(root=config.dataset.root_dir,
                             train=True,
                             transform=transforms,
                             download=True)
                dset.labels = dset.train_labels

            else:
                dset = MNIST(root=config.dataset.root_dir,
                             train=False,
                             transform=transforms,
                             download=True)
                dset.labels = dset.test_labels
            return dset

    elif dataset_name == 'voc':
        if dataset_id == '00':
            transforms = None
            if split == 'train':

                pv07 = PascalVOCDataset(root_dir=config.dataset.root_dir,
                                        split='trainval',
                                        year='2007',
                                        transform=transforms,
                                        categories_subset=config.dataset.classes,
                                        use_flipped=config.dataset.use_flipped,
                                        use_difficult=config.dataset.use_difficult)

                detection_set = DetectionWrapper(dataset=pv07,
                                                 batch_size=config.train.batch_size,
                                                 max_num_box=config.model.max_n_gt_boxes,
                                                 scales=config.train.scales,
                                                 max_size=config.train.max_size,
                                                 use_all_gt=config.train.use_all_gt,
                                                 training=True)

                return detection_set

            elif split == 'val':
                pv07 = PascalVOCDataset(root_dir=config.dataset.root_dir,
                                        split='test',
                                        year='2007',
                                        transform=transforms,
                                        categories_subset=config.dataset.classes,
                                        use_flipped=False,#config.dataset.use_flipped,
                                        use_difficult=config.dataset.use_difficult)


                detection_set = DetectionWrapper(dataset=pv07,
                                                 batch_size=config.train.batch_size,
                                                 max_num_box=config.model.max_n_gt_boxes,
                                                 scales=config.train.scales,
                                                 max_size=config.train.max_size,
                                                 use_all_gt=config.train.use_all_gt,
                                                 training=True)#False)

                return detection_set

            else:
                raise ValueError("Split '%s' not recognised for the %s dataset (id: %s)." % (split, dataset_name, dataset_id))
        elif dataset_id == '01':
            transforms = None
            if split == 'train':

                pv07 = PascalVOCDataset(root_dir=config.dataset.root_dir,
                                        split='trainval',
                                        year='2007',
                                        transform=transforms,
                                        categories_subset=config.dataset.classes,
                                        use_flipped=config.dataset.use_flipped,
                                        use_difficult=config.dataset.use_difficult)
                pv12 = PascalVOCDataset(root_dir=config.dataset.root_dir,
                                        split='trainval',
                                        year='2012',
                                        transform=transforms,
                                        categories_subset=config.dataset.classes,
                                        use_flipped=config.dataset.use_flipped,
                                        use_difficult=config.dataset.use_difficult)

                combined_set = CombinedDataset([pv07, pv12])

                detection_set = DetectionWrapper(dataset=combined_set,
                                                 batch_size=config.train.batch_size,
                                                 max_num_box=config.model.max_n_gt_boxes,
                                                 scales=config.train.scales,
                                                 max_size=config.train.max_size,
                                                 use_all_gt=config.train.use_all_gt,
                                                 training=True)

                return detection_set

            elif split == 'val':
                pv07 = PascalVOCDataset(root_dir=config.dataset.root_dir,
                                        split='test',
                                        year='2007',
                                        transform=transforms,
                                        categories_subset=config.dataset.classes,
                                        use_flipped=False,  # config.dataset.use_flipped,
                                        use_difficult=config.dataset.use_difficult)

                detection_set = DetectionWrapper(dataset=pv07,
                                                 batch_size=config.train.batch_size,
                                                 max_num_box=config.model.max_n_gt_boxes,
                                                 scales=config.train.scales,
                                                 max_size=config.train.max_size,
                                                 use_all_gt=config.train.use_all_gt,
                                                 training=True)  # False)

                return detection_set

            else:
                raise ValueError(
                    "Split '%s' not recognised for the %s dataset (id: %s)." % (split, dataset_name, dataset_id))
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
    elif sampler_name == 'episodes_repmet':
        if split == 'train':
            return EpisodeBatchSampler(labels=dataset.labels,
                                       categories_per_epi=config.train.m,
                                       num_samples=config.train.d,
                                       episodes=config.train.episodes)
        elif split == 'val':
            return EpisodeBatchSampler(labels=dataset.labels,
                                       categories_per_epi=config.val.m,
                                       num_samples=config.val.d,
                                       episodes=config.val.episodes)
        elif split == 'test':
            return EpisodeBatchSampler(labels=dataset.labels,
                                       categories_per_epi=config.test.m,
                                       num_samples=config.test.d,
                                       episodes=config.test.episodes)
        else:
            raise ValueError("Split '%s' not recognised for the %s sampler." % (split, sampler_name))
    if sampler_name == 'magnet':
        if split == 'train':
            return MagnetBatchSampler(labels=dataset.labels,
                                      k=config.train.k,
                                      m=config.train.m,
                                      d=config.train.d,
                                      iterations=config.train.episodes)
        elif split == 'val':
            return None
        elif split == 'test':
            return None
        else:
            raise ValueError("Split '%s' not recognised for the %s sampler." % (split, sampler_name))
    if sampler_name == 'detection':
        if split == 'train':
            return DetectionSampler(n_samples=len(dataset),
                                    batch_size=config.train.batch_size)
        elif split == 'val':
            return DetectionSampler(n_samples=len(dataset),
                                    batch_size=config.train.batch_size)
        elif split == 'test':
            return DetectionSampler(n_samples=len(dataset),
                                    batch_size=config.train.batch_size)
        else:
            raise ValueError("Split '%s' not recognised for the %s sampler." % (split, sampler_name))
    else:
        raise ValueError("Sampler '%s' not recognised." % sampler_name)

