"""
Oxford Pets dataset loader

http://www.robots.ox.ac.uk/~vgg/data/pets

This dataset contains images of pet animals (cats and dogs), covering 37 classes with each class consisting of around 200 images.
Images are RGB with different scales and ratios.

Official Splits (but the authors suggest random splitting)
TrainVal: 3680 samples spanning 37 classes (avg 99 per class)
Test: 3669 samples spanning 37 classes (avg 99 per class)
"""

from __future__ import print_function

from PIL import Image, ImageFile
from os.path import join
import os
import numpy as np
import tarfile
import shutil

from torch.utils.data.dataset import Dataset
from torchvision.datasets.utils import download_url


class OxfordPetsDataset(Dataset):
    # setup some class paths
    sub_root_dir = 'OxfordPets'
    download_url_prefix = 'http://www.robots.ox.ac.uk/~vgg/data/pets/data'
    images_dir = 'images'

    def __init__(self,
                 root_dir,
                 split='trainval',
                 transform=None,
                 target_transform=None,
                 force_download=False,
                 categories_subset=None):
        """
        :param root_dir: (string) the directory where the dataset will be stored
        :param split: (string) 'train', 'trainval', 'val' or 'test'
        :param transform: how to transform the input
        :param target_transform: how to transform the target
        :param force_download: (boolean) force a new download of the dataset
        :param categories_subset: (iterable) specify a subset of categories to build this set from
        """

        super(OxfordPetsDataset, self).__init__()

        # set instance variables
        self.root_dir = join(os.path.expanduser(root_dir), self.sub_root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.labels = []

        # check if data exists, if not download
        self.download(force=force_download)

        # load the data samples for this split
        self.data, self.labels, self.categories = self.load_data_split(categories_subset=categories_subset)
        self.samples = list(zip(self.data, self.labels))

        self.n_categories = len(np.unique(self.labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # get the data sample
        sample_data, sample_target = self.samples[index]

        # load the image
        x = self.load_img(join(join(self.root_dir, self.images_dir), "%s.jpg" % sample_data))
        y = sample_target

        # perform the transforms
        if self.transform:
            x = self.transform(x)

        if self.target_transform:
            y = self.target_transform(y)

        assert x.shape[0] == 3

        return x, y

    def download(self, force=False):
        # check for existence, if so return
        if os.path.exists(join(self.root_dir, 'images')) and os.path.exists(join(self.root_dir, 'annotations')):
            if not force and len(os.listdir(join(self.root_dir, 'images'))) == 7393:  # todo confirm this is the correct number
                print('Files already downloaded and verified')
                return
            else:
                shutil.rmtree(self.root_dir)

        # make the dirs and start the downloads
        os.makedirs(self.root_dir, exist_ok=True)
        filename = 'images'
        tar_filename = filename + '.tar.gz'
        url = join(self.download_url_prefix, tar_filename)
        download_url(url, self.root_dir, tar_filename, None)
        with tarfile.open(join(self.root_dir, tar_filename), 'r') as tar_file:
            tar_file.extractall(self.root_dir)
        os.remove(join(self.root_dir, tar_filename))

        filename = 'annotations'
        tar_filename = filename + '.tar.gz'
        url = join(self.download_url_prefix, tar_filename)
        download_url(url, self.root_dir, tar_filename, None)
        with tarfile.open(join(self.root_dir, tar_filename), 'r') as tar_file:
            tar_file.extractall(self.root_dir)
        os.remove(join(self.root_dir, tar_filename))

    def load_data_split(self, categories_subset=None):

        # assert we can do this split
        assert self.split in ['trainval', 'test', 'rand']

        # load the samples and their labels
        data = []
        categories = []
        if self.split == 'rand':
            with open(join(self.root_dir, 'annotations', 'list.txt')) as f:
                lines = f.readlines()
            lines = lines[6:]
            #todo will need to add some rand split settings
        elif self.split == 'trainval':
            with open(join(self.root_dir, 'annotations', 'trainval.txt')) as f:
                lines = f.readlines()
        elif self.split == 'test':
            with open(join(self.root_dir, 'annotations', 'test.txt')) as f:
                lines = f.readlines()

        lines = [line.rstrip().split() for line in lines]
        for line in lines:
            category = int(line[1])
            if categories_subset:
                if category in categories_subset:
                    data.append(line[0])
                    categories.append(category)
            else:  # categories_subset is None so add all
                data.append(line[0])
                categories.append(category)

        # Build categories to labels (cats can be names, labels are ints starting from 0)
        self.categories_to_labels = {}
        self.labels_to_categories = {}
        for c in categories:
            if c not in self.categories_to_labels:
                self.categories_to_labels[c] = len(self.categories_to_labels)
                self.labels_to_categories[self.categories_to_labels[c]] = c

        # Build labels list corresponding to each sample
        labels = []
        for c in categories:
            labels.append(self.categories_to_labels[c])

        # set the data, categories and labels used in this dataset
        # (initially ordered with self.samples and not unique, careful with access post shuffling)
        self.categories = categories
        self.labels = labels
        self.data = data

        return data, labels, categories

    @staticmethod
    def load_img(path):

        # todo either turn image to tensor in transform or do here
        # Load the image
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(path).convert('RGB')

        return image

    def stats(self):
        # get the stats to print
        counts = self.class_counts()

        return "%d samples spanning %d classes (avg %d per class)" % \
               (len(self.samples), len(counts), int(float(len(self.samples))/float(len(counts))))

    def class_counts(self):
        # calculate the number of samples per category
        counts = {}
        for index in range(len(self.samples)):
            sample_data, sample_target = self.samples[index]
            if sample_target not in counts:
                counts[sample_target] = 1
            else:
                counts[sample_target] += 1

        return counts


if __name__ == "__main__":
    # use this for debugging and checks
    from utils.debug import set_working_dir
    from config.config import config
    import matplotlib.pyplot as plt

    # set the working directory as appropriate
    set_working_dir()

    # load the dataset
    dataset = OxfordPetsDataset(root_dir=config.dataset.root_dir, categories_subset=[1,2,3])

    # print the stats
    print(dataset.stats())

    # lets plot some samples
    fig = plt.figure()

    for i in range(len(dataset)):
        sample = dataset[i]

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample %d - Class %s' % (i, dataset.labels_to_categories[sample[1]]))  # convert label to categ.
        ax.axis('off')
        plt.imshow(sample[0])  # todo when tensor will need to convert tensor to img

        if i == 3:
            plt.show()
            break
