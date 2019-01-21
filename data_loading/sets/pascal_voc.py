"""
Pascal VOC dataset loader

Train
# images: 5717
# boxes: 13609
# categories: 20
Boxes per image (min, avg, max): 1, 2, 27
Boxes per category (min, avg, max): 281, 680, 4194

Val
# images: 5823
# boxes: 13841
# categories: 20
Boxes per image (min, avg, max): 1, 2, 39
Boxes per category (min, avg, max): 285, 692, 4372

Flipped doubles the # boxes and images

http://host.robots.ox.ac.uk/pascal/VOC/
"""

import numpy as np
from PIL import Image, ImageFile
from os.path import join
import os
import scipy.io
import tarfile
import shutil
import pickle

import xml.etree.ElementTree as ET

from torch.utils.data.dataset import Dataset
from torchvision.datasets.utils import download_url

from utils.download import download


class PascalVOCDataset(Dataset):
    # setup some class paths
    sub_root_dir = 'PascalVOC'
    download_url_prefix = 'http://host.robots.ox.ac.uk/pascal/VOC/'

    def __init__(self,
                 root_dir,
                 split='train',
                 year='2012',
                 transform=None,
                 target_transform=None,
                 force_download=False,
                 categories_subset=None,
                 use_difficult=False,
                 use_flipped=False):
        """
        :param root_dir: (string) the directory where the dataset will be stored
        :param split: (string) 'train' or 'val'
        :param year: (string) the year of the set from 2012-2007
        :param transform: how to transform the input
        :param target_transform: how to transform the target
        :param force_download: (boolean) force a new download of the dataset
        :param categories_subset: (iterable) specify a subset of categories to build this set from
        :param use_difficult: (boolean) include samples marked as difficult
        :param use_flipped: (boolean) add horizontally flipped samples
        """

        super(PascalVOCDataset, self).__init__()

        # set instance variables
        self.root_dir = join(os.path.expanduser(root_dir), self.sub_root_dir)
        self.split = split
        self.year = year
        self.transform = transform
        self.target_transform = target_transform
        self.use_difficult = use_difficult
        self.use_flipped = use_flipped

        # setup the categories
        self.categories, self.categories_to_labels, self.labels_to_categories = self._init_categories(categories_subset)
        self.n_categories = len(self.categories)

        # check if data exists, if not download
        self.download(force=force_download)

        # load the data samples for this split
        self.data = self.load_data_split()  # self.data is gt_roidb (gt_roidb used in some other implementations)

        self.sample_ids = list(self.data.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get the data sample id
        sample_id = self.sample_ids[index]

        # load the image
        x = self.load_img(sample_id)
        y = self.data[sample_id]

        # perform the transforms
        if self.transform:
            x = self.transform(x)

        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def get_img_path(self, sample_id):
        if self.use_flipped and sample_id[-2:] == '_f':
            sample_id = sample_id[:-2]
        return join(self.root_dir, 'VOC'+self.year, 'JPEGImages', "%s.jpg" % sample_id)

    def download(self, force=False):

        if os.path.exists(join(self.root_dir, 'VOC'+self.year, 'JPEGImages'))\
                and os.path.exists(join(self.root_dir, 'VOC'+self.year, 'Annotations'))\
                and os.path.exists(join(self.root_dir, 'VOC'+self.year, 'ImageSets')):
            if not force:
                print('Files already downloaded and verified')
                return
            else:
                shutil.rmtree(join(self.root_dir, 'VOC'+self.year))

        # make the dirs and start the downloads
        os.makedirs(self.root_dir, exist_ok=True)
        if self.year == '2012':
            filenames = ['VOCtrainval_11-May-2012']
            md5s = ['6cd6e144f989b92b3379bac3b3de84fd']
        elif self.year == '2011':
            filenames = ['VOCtrainval_25-May-2011']
            md5s = ['6c3384ef61512963050cb5d687e5bf1e']
        elif self.year == '2010':
            filenames = ['VOCtrainval_03-May-2010']
            md5s = ['da459979d0c395079b5c75ee67908abb']
        elif self.year == '2009':
            filenames = ['VOCtrainval_11-May-2009']
            md5s = ['59065e4b188729180974ef6572f6a212']
        elif self.year == '2008':
            filenames = ['VOCtrainval_14-Jul-2008']
            md5s = ['2629fa636546599198acfcfbfcf1904a']
        elif self.year == '2007':
            filenames = ['VOCtrainval_06-Nov-2007', 'VOCtest_06-Nov-2007']
            md5s = ['c52e279531787c972589f7e41ab4ae64', '41a8d6e12baa5ab18ee7f8f8029b9e11805b4ef1']

        for filename in filenames:
            tar_filename = filename + '.tar'
            url = join(self.download_url_prefix, 'voc'+self.year, tar_filename)
            # download_url(url, self.root_dir, tar_filename, None)
            download(url, path=self.root_dir, overwrite=True)

            with tarfile.open(join(self.root_dir, tar_filename), 'r') as tar_file:
                tar_file.extractall(self.root_dir)

        shutil.move(os.path.join(self.root_dir, 'VOCdevkit', 'VOC'+self.year), os.path.join(self.root_dir, 'VOC'+self.year))
        shutil.rmtree(os.path.join(self.root_dir, 'VOCdevkit'))

        for filename in filenames:
            tar_filename = filename + '.tar'
            os.remove(join(self.root_dir, tar_filename))

    def load_data_split(self, cache=True):
        # assert we can do this split
        assert self.split in ['train', 'val', 'trainval', 'test']
        if self.split == 'test':
            assert self.year == '2007'

        # keep only the split samples
        if self.split == 'train':
            with open(join(self.root_dir, 'VOC'+self.year, 'ImageSets', 'Main', 'train.txt')) as f:
                lines = f.readlines()
        elif self.split == 'trainval':
            with open(join(self.root_dir, 'VOC'+self.year, 'ImageSets', 'Main', 'trainval.txt')) as f:
                lines = f.readlines()
        elif self.split == 'val':
            with open(join(self.root_dir, 'VOC'+self.year, 'ImageSets', 'Main', 'val.txt')) as f:
                lines = f.readlines()
        elif self.split == 'test':
            with open(join(self.root_dir, 'VOC'+self.year, 'ImageSets', 'Main', 'test.txt')) as f:
                lines = f.readlines()

        sample_ids = [line.strip() for line in lines]

        # load cache if able
        if cache:
            if self.n_categories != 21:
                print("Not using default categories so not caching for now...")
                # todo load normal cache and delete data entries
            else:
                if self.use_flipped:
                    cache_file = join(self.root_dir, 'VOC'+self.year, self.split+'_data_cache_wflipped.pkl')
                else:
                    cache_file = join(self.root_dir, 'VOC'+self.year, self.split+'_data_cache.pkl')
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as fid:
                        try:
                            data = pickle.load(fid)
                        except:
                            data = pickle.load(fid, encoding='bytes')
                    print('Cache data loaded from {}'.format(cache_file))
                    return data

        # build the data dict
        data = {}
        for sample_id in sample_ids:
            annotation = self._load_annotation(sample_id)
            if len(annotation['boxes'] > 0):  # only add sample to set if it contains at least one gt box
                data[sample_id] = annotation
                if self.use_flipped:
                    flipped_annotation = self._flip_annotation(annotation, sample_id)
                    data[sample_id+'_f'] = flipped_annotation

        # save cache if desired
        if cache and self.n_categories == 21:
            with open(cache_file, 'wb') as fid:
                pickle.dump(data, fid, pickle.HIGHEST_PROTOCOL)
            print('Cache data written to {}'.format(cache_file))

        return data

    def _load_annotation(self, sample_id):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC format.
        """
        width, height = self.load_img(sample_id).size

        filename = join(self.root_dir, 'VOC'+self.year, 'Annotations', sample_id + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.use_difficult:
            # Exclude the samples labeled as difficult
            non_diff_objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.n_categories), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self.categories_to_labels[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'width': width,
                'height': height,
                'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False}#,
                # 'seg_areas': seg_areas}

    def _flip_annotation(self, annotation, sample_id):

        width = self.load_img(sample_id).size[0]

        boxes = annotation['boxes'].copy()
        oldx1 = boxes[:, 0].copy()
        oldx2 = boxes[:, 2].copy()
        boxes[:, 0] = width - oldx2 - 1
        boxes[:, 2] = width - oldx1 - 1
        assert (boxes[:, 2] >= boxes[:, 0]).all()

        return {'width': annotation['width'],
                'height': annotation['height'],
                'boxes': boxes,
                'gt_classes': annotation['gt_classes'],
                'gt_overlaps': annotation['gt_overlaps'],
                'flipped': True}

    @staticmethod
    def load_img_from_path(path, flip=False):

        # todo either turn image to tensor in transform or do here
        # Load the image
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(path).convert('RGB')

        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        return image

    def load_img(self, sample_id):
        if self.use_flipped and sample_id[-2:] == '_f':
            return self.load_img_from_path(self.get_img_path(sample_id), flip=True)
        else:
            return self.load_img_from_path(self.get_img_path(sample_id), flip=False)


    def stats(self):
        # get the stats to print
        boxes_p_cls, boxes_p_img = self.class_counts()

        return "# images: %d\n" \
               "# boxes: %d\n"\
               "# categories: %d\n"\
               "Boxes per image (min, avg, max): %d, %d, %d\n"\
               "Boxes per category (min, avg, max): %d, %d, %d\n" % \
               (len(self.data), sum(boxes_p_img), len(boxes_p_cls),
                min(boxes_p_img), sum(boxes_p_img) / len(boxes_p_img), max(boxes_p_img),
                min(boxes_p_cls), sum(boxes_p_cls) / len(boxes_p_cls), max(boxes_p_cls))

    def class_counts(self):
        # calculate the number of samples per category, and per image
        boxes_p_cls = [0]*(self.n_categories-1)  # minus 1 for background removal
        boxes_p_img = []
        for sample_id in self.data.keys():
            boxes_this_img = 0
            annotations = self.data[sample_id]
            for label in annotations['gt_classes']:
                boxes_p_cls[label-1] += 1
                boxes_this_img += 1
            boxes_p_img.append(boxes_this_img)

        return boxes_p_cls, boxes_p_img

    @staticmethod
    def _init_categories(categories_subset):
        categories = ['__background__',  # always index 0
                      'aeroplane', 'bicycle', 'bird', 'boat',
                      'bottle', 'bus', 'car', 'cat', 'chair',
                      'cow', 'diningtable', 'dog', 'horse',
                      'motorbike', 'person', 'pottedplant',
                      'sheep', 'sofa', 'train', 'tvmonitor']

        if categories_subset:
            assert 0 in categories_subset  # ensure background is always included
            categories = [categories[i] for i in categories_subset]

        # Build categories to labels (cats can be names, labels are ints starting from 0)
        categories_to_labels = {}
        labels_to_categories = {}
        for c in categories:
            if c not in categories_to_labels:
                categories_to_labels[c] = len(categories_to_labels)
                labels_to_categories[categories_to_labels[c]] = c

        return categories, categories_to_labels, labels_to_categories

    @staticmethod
    def display(img, boxes, classes):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        # Display the image
        plt.imshow(img)

        # Add the boxes with labels
        for i in range(len(classes)):
            box = boxes[i]
            plt.gca().add_patch(Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none'))
            plt.text(box[0], box[1]+20, str(classes[i]), fontsize=12, color='r')

        return plt


if __name__ == "__main__":
    # use this for debugging and checks
    from utils.debug import set_working_dir
    from config.config import config
    import matplotlib.pyplot as plt

    # set the working directory as appropriate
    set_working_dir()

    # load the dataset
    dataset = PascalVOCDataset(root_dir=config.dataset.root_dir, split='train', year='2007', use_flipped=False)
    dataset = PascalVOCDataset(root_dir=config.dataset.root_dir, split='test', year='2007', use_flipped=False)
    dataset = PascalVOCDataset(root_dir=config.dataset.root_dir, split='train', year='2012', use_flipped=False)

    # print the stats
    print(dataset.stats())

    # lets plot some samples
    fig = plt.figure()

    for i in range(len(dataset)):
        sample = dataset[i]

        plt = dataset.display(sample[0], sample[1]['boxes'], sample[1]['gt_classes'])

        plt.show()
        if i == 3:
            break
