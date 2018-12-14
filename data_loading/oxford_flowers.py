"""
Oxford Flowers dataset loader

1020 samples spanning 102 classes (avg 10 per class)

http://www.robots.ox.ac.uk/~vgg/data/flowers
"""

from __future__ import print_function

from PIL import Image
from os.path import join
import os
import scipy.io
import tarfile
import shutil
import numpy as np

from torch.utils.data.dataset import Dataset
from torchvision.datasets.utils import download_url


class OxfordFlowersDataset(Dataset):
    sub_root_dir = 'OxfordFlowers'

    download_url_prefix = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102'

    images_dir = 'jpg'

    def __init__(self,
                 root_dir,
                 split='train',
                 transform=None,
                 target_transform=None,
                 force_download=False,
                 categories_subset=None):

        self.root_dir = join(os.path.expanduser(root_dir), self.sub_root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.labels = []

        self.download(force=force_download)

        self.data = self.load_data_split(categories_subset=categories_subset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, target_class = self.data[index]

        image = self.load_img(join(join(self.root_dir, self.images_dir), "image_%05d.jpg" % image_name))

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target_class = self.target_transform(target_class)

        return image, target_class

    def download(self, force=False):

        if os.path.exists(join(self.root_dir, 'jpg')) and os.path.exists(join(self.root_dir, 'imagelabels.mat'))\
                and os.path.exists(join(self.root_dir, 'setid.mat')):
            if not force and len(os.listdir(join(self.root_dir, 'jpg'))) == 8189:
                print('Files already downloaded and verified')
                return
            else:
                shutil.rmtree(self.root_dir)

        os.makedirs(self.root_dir, exist_ok=True)
        filename = '102flowers'
        tar_filename = filename + '.tgz'
        url = join(self.download_url_prefix, tar_filename)
        download_url(url, self.root_dir, tar_filename, None)
        with tarfile.open(join(self.root_dir, tar_filename), 'r') as tar_file:
            tar_file.extractall(self.root_dir)
        os.remove(join(self.root_dir, tar_filename))

        filename = 'imagelabels.mat'
        url = join(self.download_url_prefix, filename)
        download_url(url, self.root_dir, filename, None)

        filename = 'setid.mat'
        url = join(self.download_url_prefix, filename)
        download_url(url, self.root_dir, filename, None)

    def load_data_split(self, categories_subset=None):
        """
        sets categories, categories_to_labels,
        :param categories_subset: if not None only include the categories listed in this iterable
        :return:
        """
        assert self.split in ['train', 'val', 'test']

        samples = scipy.io.loadmat(join(self.root_dir, 'setid.mat'))
        categories = scipy.io.loadmat(join(self.root_dir, 'imagelabels.mat'))['labels']

        self.categories = list(np.unique(categories))

        if self.split == 'train':
            samples = samples['trnid']
        elif self.split == 'val':
            samples = samples['valid']
        elif self.split == 'test':
            samples = samples['tstid']

        samples = list(samples[0])
        categories = list(categories[0][samples])

        # Build categories to labels (cats can be names, labels are ints starting from 0)
        self.categories_to_labels = {}
        self.labels_to_categories = {}
        for c in categories:
            if c not in self.categories_to_labels:
                self.categories_to_labels[c] = len(self.categories_to_labels)
                self.labels_to_categories[self.categories_to_labels[c]] = c

        data = []
        labels = []
        for index in range(len(samples)):
            category = categories[index]
            if categories_subset:
                if category in categories_subset:
                    data.append(samples[index])
                    labels.append(self.categories_to_labels[categories[index]])
            else:  # categories_subset is None so add all
                data.append(samples[index])
                labels.append(self.categories_to_labels[categories[index]])

        return list(zip(data, labels))

    def load_img(self, path):
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        x = Image.open(path).convert('RGB')

        return x

    def stats(self):
        counts = self.class_counts()

        return "%d samples spanning %d classes (avg %d per class)" % \
               (len(self.data), len(counts.keys()), int(float(len(self.data))/float(len(counts.keys()))))

    def class_counts(self):
        counts = {}
        for index in range(len(self.data)):
            image_name, target_class = self.data[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        return counts


if __name__ == "__main__":
    from utils.debug import set_working_dir
    from config.config import config
    import matplotlib.pyplot as plt

    set_working_dir()

    dataset = OxfordFlowersDataset(root_dir=config.data.root_dir, categories_subset=[1,2,34])

    print(dataset.stats())

    fig = plt.figure()

    for i in range(len(dataset)):
        sample = dataset[i]

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample %d - Class %d' % (i, dataset.labels_to_categories[sample[1]]))
        ax.axis('off')
        plt.imshow(sample[0])

        if i == 3:
            plt.show()
            break