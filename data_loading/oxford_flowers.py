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

    def __init__(self,
                 root_dir,
                 split='train',
                 transform=None,
                 target_transform=None,
                 force_download=False,
                 labels=None):

        self.root_dir = join(os.path.expanduser(root_dir), self.sub_root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.labels = []

        self.download(force=force_download)

        self.data = self.load_data_split()

        self.images_sub_root_dir = join(self.root_dir, 'jpg')

        if labels:
            self.labels_subset(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image_name, target_class = self.data[index]
        image_path = join(self.images_sub_root_dir, "image_%05d.jpg" % image_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target_class = self.target_transform(target_class)

        return image, target_class

    def download(self, force=False):
        download_url_prefix = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102'

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
        url = join(download_url_prefix, tar_filename)
        download_url(url, self.root_dir, tar_filename, None)
        with tarfile.open(join(self.root_dir, tar_filename), 'r') as tar_file:
            tar_file.extractall(self.root_dir)
        os.remove(join(self.root_dir, tar_filename))

        filename = 'imagelabels.mat'
        url = join(download_url_prefix, filename)
        download_url(url, self.root_dir, filename, None)

        filename = 'setid.mat'
        url = join(download_url_prefix, filename)
        download_url(url, self.root_dir, filename, None)

    def load_data_split(self):

        assert self.split in ['train', 'val', 'test']

        data = scipy.io.loadmat(join(self.root_dir, 'setid.mat'))
        labels = scipy.io.loadmat(join(self.root_dir, 'imagelabels.mat'))['labels']

        self.labels = list(np.unique(labels))

        if self.split == 'train':
            data = data['trnid']
        elif self.split == 'val':
            data = data['valid']
        elif self.split == 'test':
            data = data['tstid']

        data = list(data[0])
        labels = list(labels[0][data])
        return list(zip(data, labels))

    def labels_subset(self, labels):
        assert isinstance(labels, list)
        data = []
        for i, sample in enumerate(self.data):
            if sample[1] in labels:
                data.append(self.data[i])
        self.data = data
        self.labels = labels

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

    dataset = OxfordFlowersDataset(root_dir=config.data.root_dir)

    print(dataset.stats())

    fig = plt.figure()

    for i in range(len(dataset)):
        sample = dataset[i]

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample %d - Class %d' % (i, sample[1]))
        ax.axis('off')
        plt.imshow(sample[0])

        if i == 3:
            plt.show()
            break