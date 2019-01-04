"""
Omniglot dataset loader

82240 samples spanning 4112 classes (avg 20 per class)

https://github.com/brendenlake/omniglot

Code modified from https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch
which was
Inspired by https://github.com/pytorch/vision/pull/46
"""
from __future__ import print_function

from PIL import Image
import numpy as np
import shutil
import errno
import torch
import os
from os.path import join

from six.moves import urllib
import zipfile

from torch.utils.data.dataset import Dataset


class OmniglotDataset(Dataset):
    # setup some class paths
    sub_root_dir = 'Omniglot'

    download_url_prefix = 'https://raw.githubusercontent.com/jakesnell/prototypical-networks/master/data/omniglot/splits/vinyals/'
    download_url_splits = {
        'test': download_url_prefix + 'test.txt',
        'train': download_url_prefix + 'train.txt',
        'trainval': download_url_prefix + 'trainval.txt',
        'val': download_url_prefix + 'val.txt',
    }

    data_urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    splits_dir = join('splits', 'vinyals')
    data_dir = 'data'

    def __init__(self,
                 root_dir,
                 split='train',
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

        super(OmniglotDataset, self).__init__()

        # set instance variables
        self.root_dir = join(os.path.expanduser(root_dir), self.sub_root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

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
        x = self.load_img(join(self.root_dir, self.data_dir, sample_data[1], sample_data[0]), rotate=sample_data[2][-3:])
        y = sample_target

        # perform the transforms
        if self.transform:
            x = self.transform(x)
        
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def download(self, force=False):

        # check for existence, if so return
        if os.path.exists(join(self.root_dir, self.data_dir)):
            if not force and len(os.listdir(join(self.root_dir, self.data_dir))) == 50:
                print('Files already downloaded and verified')
                return
            else:
                shutil.rmtree(self.root_dir)

        # Make the dirs
        try:
            os.makedirs(join(self.root_dir, self.splits_dir))
            os.makedirs(join(self.root_dir, 'tmp'))
            os.makedirs(join(self.root_dir, self.data_dir))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # Download the split files
        for k, url in self.download_url_splits.items():
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[-1]
            file_path = join(self.root_dir, self.splits_dir, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())

        # Download the data
        for url in self.data_urls:
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = join(self.root_dir, 'tmp', filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
                
            tmp_dir = join(self.root_dir, 'tmp')
            print("== Unzip from " + file_path + " to " + tmp_dir)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(tmp_dir)
            zip_ref.close()
        
        # mv the data    
        file_processed = join(self.root_dir, self.data_dir)
        for p in ['images_background', 'images_evaluation']:
            for f in os.listdir(join(tmp_dir, p)):
                shutil.move(join(tmp_dir, p, f), file_processed)

        shutil.rmtree(tmp_dir)
        print("Download finished.")

    def load_data_split(self, categories_subset=None):

        # assert we can do this split
        assert self.split in ['train', 'trainval', 'val', 'test']

        # Load the categories of this split
        with open(join(self.root_dir, self.splits_dir, self.split + '.txt')) as f:
            split_categories = f.read().splitlines()

        # Load the data
        data = []
        categories = []
        rots = ['/rot000', '/rot090', '/rot180', '/rot270']
        for (root, dirs, files) in os.walk(join(self.root_dir, self.data_dir)):
            for f in files:
                r = root.split('/')
                lr = len(r)
                label = r[lr - 2] + "/" + r[lr - 1]
                for rot in rots:
                    category = label + rot
                    if category in split_categories and (f.endswith("png")):  # is this category part of the split?
                        
                        if categories_subset:
                            if category in categories_subset:
                                data.append((f, label, rot))
                                categories.append(category)
                        else:  # categories_subset is None so add all
                            data.append((f, label, rot))
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

        return data, labels, categories

    @staticmethod
    def load_img(path, rotate=0):

        # todo consider putting some of these transforms into transform functions that are passed
        x = Image.open(path)
        x = x.rotate(float(rotate))
        x = x.resize((28, 28))

        shape = 1, x.size[0], x.size[1]
        x = np.array(x, np.float32, copy=False)
        x = 1.0 - torch.from_numpy(x)
        x = x.transpose(0, 1).contiguous().view(shape)

        return x

    @staticmethod
    def tensor_to_image(tensor):
        np_tensor = tensor.numpy()  # convert to numpy
        np_tensor = np.uint8(np_tensor * 255)  # change to ints 0-255
        np_tensor = np.reshape(np_tensor, (np_tensor.shape[1], np_tensor.shape[2], np_tensor.shape[0]))  # reshape to w,h,c
        if np_tensor.shape[2] == 1:  # if single channel make into 3 channel image
            np_tensor = np.repeat(np_tensor, 3, axis=2)  # just repeat the single channel

        return Image.fromarray(np_tensor)

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
    dataset = OmniglotDataset(root_dir=config.dataset.root_dir)

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
        plt.imshow(dataset.tensor_to_image(sample[0]))  # convert tensor to img

        if i == 3:
            plt.show()
            break