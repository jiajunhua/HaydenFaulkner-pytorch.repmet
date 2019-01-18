"""
detection_wrapper.py

A dataset object that wraps another dataset and returns samples that are in a form useful to fast/er-RCNN.

It processes the images and boxes, handles scaling, cropping and padding of both the img and gt_boxes.

Also will ensure all samples within a batch (defined by the batch_size) are of same dimensions.

This code was inspired by codes in:
https://github.com/jwyang/faster-rcnn.pytorch/tree/333ac7ac475c74ed5dfccf0d988a63d6c1b18b82/lib/roi_data_layer

"""

import torch

from torch.utils.data.dataset import Dataset

import numpy as np
import numpy.random as npr
from torchvision import transforms as trns


class DetectionWrapper(Dataset):
    def __init__(self,
                 dataset,
                 batch_size=128,
                 max_num_box=20,
                 scales=(600,),
                 max_size=1000,
                 use_all_gt=True,
                 training=True):

        super(DetectionWrapper, self).__init__()

        # Set instance variables
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_num_box = max_num_box
        self.scales = scales
        self.max_size = max_size
        self.use_all_gt = use_all_gt
        self.training = training

        # Set this data to that of the wrapped dataset

        self.data = dataset.data
        self.sample_ids = dataset.sample_ids
        self.n_categories = dataset.n_categories

        if training:  # todo not sure whether to do for testing too?
            self.prepare_dataset()

        # given the ratio_list, we want to make the ratio same for each batch.
        self.ratio_list, self.ratio_index = self.rank_data_ratio()

    def __getitem__(self, index):

        # Change index to be that of particular ratio'd image
        if self.training:
            index = int(self.ratio_index[index])

        # Get the sample from the inner dataset
        img, y = self.dataset[index]

        # Form the gt_boxes array
        gt_boxes = self.form_gt_boxes(y)

        # Scale the img and gt_boxes with the scales in self.scales
        img, gt_boxes, im_scale = self.scale(img, gt_boxes)

        # Convert from PIL img to tensor
        img = trns.ToTensor()(img)  #  -> torch.FloatTensor (C x H x W) [0.0, 1.0]

        # Make im_info tensor
        im_info = torch.from_numpy(np.array([img.shape[1], img.shape[2], im_scale], dtype=np.float32))

        if self.training:
            # we need to random shuffle the bounding boxes and convert to tensor
            np.random.shuffle(gt_boxes)
            gt_boxes = torch.from_numpy(gt_boxes)

            # if the image need to crop, crop to the target size.
            ratio = self.ratio_list[list(self.ratio_index).index(index)]  # todo index is not correct, should be 0

            # Crop samples if their aspect ratio is too great
            if y['need_crop']:
                img, gt_boxes = self.crop(ratio, img, gt_boxes)

            # Pad samples so they enter the network with same size
            padded_img, padded_gt_boxes, num_boxes = self.pad(ratio, img, gt_boxes)

            # Update the im_info tensors with padded img dims
            im_info[0] = padded_img.size(1)
            im_info[1] = padded_img.size(2)

            padded_img = padded_img.contiguous()

            return padded_img, im_info, padded_gt_boxes, num_boxes
        else:
            img = img.contiguous()

            gt_boxes = torch.FloatTensor([1, 1, 1, 1, 1])
            num_boxes = 0

            return img, im_info, gt_boxes, num_boxes

    def __len__(self):
        return len(self.data)

    def form_gt_boxes(self, y):
        """
        Form numpy gt_boxes array from the y dictionary

        Also filters base on self.use_all_gt which for COCO ground truth boxes, excludes the ones that are ''iscrowd''

        :param y: the dictionary
        :return: a numpy array of (x1, y1, x2, y2, cls)
        """

        if self.use_all_gt:
            # Include all ground truth boxes
            gt_inds = np.where(y['gt_classes'] != 0)[0]
        else:
            # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
            gt_inds = np.where((y['gt_classes'] != 0) & np.all(y['gt_overlaps'].toarray() > -1.0, axis=1))[0]

        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes[:, 0:4] = y['boxes'][gt_inds]
        gt_boxes[:, 4] = y['gt_classes'][gt_inds]

        return gt_boxes

    def scale(self, img, gt_boxes):
        """
        Scale images to be at least the scale we desire in self.scales

        :param img: the PIL image to scale
        :param gt_boxes: the numpy gt_boxes to scale
        :return: the scaled PIL img and numpy gt_boxes, plus the im_scale that we using
        """
        # Sample random scales from self.scales
        random_scale_inds = npr.randint(0, high=len(self.scales), size=1)
        target_size = self.scales[random_scale_inds[0]]

        # Calculate the scale factor based on the shortest edge
        w, h = img.size
        im_size_min = w if w < h else h
        im_size_max = w if w > h else h
        im_scale = float(target_size) / float(im_size_min)

        # todo max_size was commented out in base code, need to test its effects with cropping and padding
        # Prevent the biggest edge from being more than config.train.max_size
        # if np.round(im_scale * im_size_max) > self.max_size:
        #     im_scale = float(self.max_size) / float(im_size_max)

        # Scale the (PIL) image
        img = trns.Resize((int(np.round(h*im_scale)), int(np.round(w*im_scale))))(img)

        # Scale the (NUMPY) gt_boxes
        gt_boxes[:, 0:4] = gt_boxes[:, 0:4] * im_scale

        return img, gt_boxes, im_scale

    def crop(self, ratio, img, gt_boxes):
        """
        Crops an image and it's gt_boxes if the aspect ratio is too great
        :param ratio: the aspect ratio of the image (might not be equal to img_w/img_h since modified in batch)
        :param img: the img tensor
        :param gt_boxes: the gt_boxes tensor
        :return: cropped img and gt_boxes tensors
        """
        ch, img_height, img_width = img.shape

        if ratio < 1:
            # this means that img_width << img_height, we need to crop the img_height

            # get top and bottom coords of bboxs taking extremes (we dont want to crop off boxes)
            min_y = int(torch.min(gt_boxes[:, 1]))
            max_y = int(torch.max(gt_boxes[:, 3]))
            trim_size = int(np.floor(img_width / ratio))
            if trim_size > img_height:
                trim_size = img_height
            box_region = max_y - min_y + 1
            if min_y == 0:
                y_s = 0
            else:
                if (box_region - trim_size) < 0:  # box region height area smaller than the area we need to trim to
                    # what range can we crop too (just need to work out top coord, use trim_size to work out bottom)
                    y_s_min = max(max_y - trim_size, 0)
                    y_s_max = min(min_y, img_height - trim_size)
                    if y_s_min == y_s_max:
                        y_s = y_s_min
                    else:  # randomly select a crop height coord
                        y_s = np.random.choice(range(y_s_min, y_s_max))
                else:  # box region height area bigger than what we need to trim to
                    # so let's minimise the amount we are going to cut off
                    y_s_add = int((box_region - trim_size) / 2)
                    if y_s_add == 0:
                        y_s = min_y
                    else:
                        y_s = np.random.choice(range(min_y, min_y + y_s_add))
            # crop the image
            img = img[:, y_s:(y_s + trim_size), :]

            # shift y coordiante of gt_boxes
            gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_s)
            gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_s)

            # make sure the new positions are legal
            gt_boxes[:, 1].clamp_(0, trim_size - 1)
            gt_boxes[:, 3].clamp_(0, trim_size - 1)

        else:
            # this means that img_width >> img_height, we need to crop the img_width
            # get left and right coords of bboxs taking extremes (we dont want to crop off boxes)
            min_x = int(torch.min(gt_boxes[:, 0]))
            max_x = int(torch.max(gt_boxes[:, 2]))
            trim_size = int(np.ceil(img_height * ratio))
            if trim_size > img_width:
                trim_size = img_width
            box_region = max_x - min_x + 1
            if min_x == 0:
                x_s = 0
            else:
                if (box_region - trim_size) < 0:  # box region width area smaller than the area we need to trim to
                    # what range can we crop too (just need to work out left coord, use trim_size to work out right)
                    x_s_min = max(max_x - trim_size, 0)
                    x_s_max = min(min_x, img_width - trim_size)
                    if x_s_min == x_s_max:
                        x_s = x_s_min
                    else:  # randomly select a crop width coord
                        x_s = np.random.choice(range(x_s_min, x_s_max))
                else:  # box region width area bigger than what we need to trim to
                    # so let's minimise the amount we are going to cut off
                    x_s_add = int((box_region - trim_size) / 2)
                    if x_s_add == 0:
                        x_s = min_x
                    else:
                        x_s = np.random.choice(range(min_x, min_x + x_s_add))
            # crop the image
            img = img[:, :, x_s:(x_s + trim_size)]

            # shift x coordiante of gt_boxes
            gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
            gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)

            # make sure the new positions are legal
            gt_boxes[:, 0].clamp_(0, trim_size - 1)
            gt_boxes[:, 2].clamp_(0, trim_size - 1)

        return img, gt_boxes

    def pad(self, ratio, img, gt_boxes):
        """
        Pad an image and its gt_boxes so consitant size across batch
        Also pad the gt_boxes to config.model.max_n_gt_boxes

        With pad to smallest_edge x ceil(smallest_edge / ratio) where smallest edge could be either w or h

        :param ratio: the desired aspect ratio of the image
        :param img: the image tensor
        :param gt_boxes: the gt_boxes tensor
        :return: the padded img, padded gt_boxes, and number of gt boxes
        """
        ch, img_height, img_width = img.shape

        # based on the ratio, padding the image.
        if ratio < 1:
            # this means that img_width < img_height
            padded_img = torch.FloatTensor(3, int(np.ceil(img_width / ratio)), img_width).zero_()  # zeros (padding)
            t = padded_img.shape[1]
            tt = img_height
            try:
                padded_img[:, :img_height, :] = img  # place from top (padding is bottom)
            except RuntimeError:
                print('g')
        elif ratio > 1:
            # this means that img_width > img_height
            padded_img = torch.FloatTensor(3, img_height, int(np.ceil(img_height * ratio))).zero_()  # zeros (padding)
            t = padded_img.shape[2]
            tt =img_width
            try:
                padded_img[:, :, :img_width] = img  # place from left (padding is right)
            except RuntimeError:
                print('g')
        else:  # do some minor trimming into a square image
            trim_size = min(img_height, img_width)
            padded_img = img[:, :trim_size, :trim_size]
            # trim the gt_boxes a bit if need be, keeping them legal
            gt_boxes[:, :4].clamp_(0, trim_size)

        # check the bounding boxes that they haven't been squashed to nothing
        not_keep = (gt_boxes[:, 0] == gt_boxes[:, 2]) | (gt_boxes[:, 1] == gt_boxes[:, 3])
        keep = torch.nonzero(not_keep == 0).view(-1)

        # lets also pad the gt_boxes so net doesn't receive a variable amount every sample
        padded_gt_boxes = torch.FloatTensor(self.max_num_box, gt_boxes.size(1)).zero_()
        if keep.numel() != 0:
            gt_boxes = gt_boxes[keep]
            num_boxes = min(gt_boxes.size(0), self.max_num_box)
            padded_gt_boxes[:num_boxes, :] = gt_boxes[:num_boxes]
        else:
            num_boxes = 0

        return padded_img, padded_gt_boxes, num_boxes

    def rank_data_ratio(self):
        """
        Rank and index samples based on their aspect ratio

        :return: a tensor of the ratios smallest to largest and a sorted array of sample indexs corresponding to the ratios_list
        """

        # rank data based on the ratio between width and height.
        ratio_large = 2  # largest ratio to preserve.
        ratio_small = 0.5  # smallest ratio to preserve.

        # calculate, clip and store the ratios, add flag if imgs need crop
        ratio_list = []
        for sample_id in self.sample_ids:
            width = self.data[sample_id]['width']
            height = self.data[sample_id]['height']
            ratio = width / float(height)

            if ratio > ratio_large:
                self.data[sample_id]['need_crop'] = 1
                ratio = ratio_large
            elif ratio < ratio_small:
                self.data[sample_id]['need_crop'] = 1
                ratio = ratio_small
            else:
                self.data[sample_id]['need_crop'] = 0

            ratio_list.append(ratio)

        ratio_list = np.array(ratio_list)
        ratio_index = np.argsort(ratio_list)  # sort

        ratio_list = ratio_list[ratio_index]  # rearrange sample indexs in the ratio smallest to largest order

        # make sure batches have imgs of the same ratios by changing the target ratio data to be equal in each batch
        num_batch = int(np.ceil(len(ratio_index) / self.batch_size))
        for i in range(num_batch):
            left_idx = i * self.batch_size
            right_idx = min((i + 1) * self.batch_size - 1, len(ratio_list) - 1)

            if ratio_list[right_idx] < 1:
                # for ratio < 1, we preserve the leftmost in each batch.
                target_ratio = ratio_list[left_idx]
            elif ratio_list[left_idx] > 1:
                # for ratio > 1, we preserve the rightmost in each batch.
                target_ratio = ratio_list[right_idx]
            else:
                # for ratio cross 1, we make it to be 1.
                target_ratio = 1

            ratio_list[left_idx:(right_idx + 1)] = target_ratio

        ratio_list = torch.from_numpy(ratio_list)  # todo does it need to be a tensor?

        return ratio_list, ratio_index

    def prepare_dataset(self):
        """Enrich the dataset's roidb by adding some derived quantities that
        are useful for training. This function precomputes the maximum
        overlap, taken over ground-truth boxes, between each ROI and
        each ground-truth box. The class with maximum overlap is also
        recorded.
        """
        # todo check whether these actually helpful
        for sample_id in self.data.keys():
            self.data[sample_id]['img_id'] = sample_id
            self.data[sample_id]['image'] = self.dataset.get_img_path(sample_id)

            # need gt_overlaps as a dense array for argmax
            gt_overlaps = self.data[sample_id]['gt_overlaps'].toarray()

            # max overlap with gt over classes (columns)
            max_overlaps = gt_overlaps.max(axis=1)

            # gt class that had the max overlap
            max_classes = gt_overlaps.argmax(axis=1)
            self.data[sample_id]['max_classes'] = max_classes
            self.data[sample_id]['max_overlaps'] = max_overlaps

            # sanity checks
            # max overlap of 0 => class should be zero (background)
            zero_inds = np.where(max_overlaps == 0)[0]
            assert all(max_classes[zero_inds] == 0)
            # max overlap > 0 => class should not be zero (must be a fg class)
            nonzero_inds = np.where(max_overlaps > 0)[0]
            assert all(max_classes[nonzero_inds] != 0)


if __name__ == "__main__":
    # use this for debugging and checks
    from utils.debug import set_working_dir
    from config.config import config
    from data_loading.sets.pascal_voc import PascalVOCDataset
    from data_loading.sets.combined import CombinedDataset

    # set the working directory as appropriate
    set_working_dir()

    # load the dataset
    datasetA = PascalVOCDataset(root_dir=config.dataset.root_dir, split='train', use_flipped=False)
    datasetB = PascalVOCDataset(root_dir=config.dataset.root_dir, split='val', use_flipped=True)
    datasetC = CombinedDataset(datasets=[datasetA, datasetB])

    datasetD = DetectionWrapper(datasetC, training=True)

    for i in range(len(datasetD)):
        sample = datasetD[i]
