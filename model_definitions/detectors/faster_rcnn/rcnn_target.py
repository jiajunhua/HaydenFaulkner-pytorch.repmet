# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------
# --------------------------------------------------------
# Modified by Hayden Faulkner
# was proposal_target_layer_cascade
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np

from bbox_transform import bbox_overlaps_batch, bbox_transform_batch


class RCNNTargetSampler(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self,
                 nclasses,
                 batch_size,
                 fg_fraction,
                 fg_thresh,
                 bg_thresh_high,
                 bg_thresh_low,
                 bbox_normalize_means,
                 bbox_normalize_stds,
                 bbox_normalize_inside_weights):

        super(RCNNTargetSampler, self).__init__()

        self._num_classes = nclasses

        self.BATCH_SIZE = batch_size

        self.FG_FRACTION = fg_fraction
        self.FG_THRESH = fg_thresh
        self.BG_THRESH_HI = bg_thresh_high
        self.BG_THRESH_LO = bg_thresh_low

        # self.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = bbox_normalize_targets_precomputed
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(bbox_normalize_means)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(bbox_normalize_stds)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(bbox_normalize_inside_weights)

    def forward(self, all_rois, gt_boxes, num_boxes):

        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes)
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes)
        self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)

        gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_()
        gt_boxes_append[:, :, 1:5] = gt_boxes[:, :, :4]

        # Include ground-truth boxes in the set of candidate rois
        all_rois = torch.cat([all_rois, gt_boxes_append], 1)

        num_images = 1
        rois_per_image = int(self.BATCH_SIZE / num_images)
        fg_rois_per_image = int(np.round(self.FG_FRACTION * rois_per_image))
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image

        # sample function was here

        overlaps = bbox_overlaps_batch(all_rois, gt_boxes)

        max_overlaps, gt_assignment = torch.max(overlaps, 2)

        batch_size = overlaps.size(0)
        num_proposal = overlaps.size(1)
        num_boxes_per_img = overlaps.size(2)

        offset = torch.arange(0, batch_size)*gt_boxes.size(1)
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment

        # changed indexing way for pytorch 1.0
        labels = gt_boxes[:,:,4].contiguous().view(-1)[(offset.view(-1),)].view(batch_size, -1)

        labels_batch = labels.new(batch_size, rois_per_image).zero_()
        rois_batch  = all_rois.new(batch_size, rois_per_image, 5).zero_()
        gt_rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()
        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs
        for i in range(batch_size):

            fg_inds = torch.nonzero(max_overlaps[i] >= self.FG_THRESH).view(-1)
            fg_num_rois = fg_inds.numel()

            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = torch.nonzero((max_overlaps[i] < self.BG_THRESH_HI) &
                                    (max_overlaps[i] >= self.BG_THRESH_LO)).view(-1)
            bg_num_rois = bg_inds.numel()

            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_num_rois).long().cuda()
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

                # sampling bg
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image

                # Seems torch.rand has a bug, it will generate very large number and make an error.
                # We use numpy rand instead.
                #rand_num = (torch.rand(bg_rois_per_this_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]

            elif fg_num_rois > 0 and bg_num_rois == 0:
                # sampling fg
                #rand_num = torch.floor(torch.rand(rois_per_image) * fg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = rois_per_image
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                # sampling bg
                #rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()

                bg_inds = bg_inds[rand_num]
                bg_rois_per_this_image = rois_per_image
                fg_rois_per_this_image = 0
            else:
                raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")

            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)

            # Select sampled values from various arrays:
            labels_batch[i].copy_(labels[i][keep_inds])

            # Clamp labels for the background RoIs to 0
            if fg_rois_per_this_image < rois_per_image:
                labels_batch[i][fg_rois_per_this_image:] = 0

            rois_batch[i] = all_rois[i][keep_inds]
            rois_batch[i,:,0] = i

            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]

        matches = labels_batch
        rois = rois_batch
        gt_rois = gt_rois_batch

        return gt_rois, rois, matches

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


class RCNNTargetGenerator(nn.Module):
    """
    RCNN target encoder to generate matching target and regression target values.
    """

    def __init__(self,
                 bbox_normalize_targets_precomputed,
                 bbox_normalize_means,
                 bbox_normalize_stds,
                 bbox_normalize_inside_weights):

        super(RCNNTargetGenerator, self).__init__()

        self.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = bbox_normalize_targets_precomputed
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(bbox_normalize_means)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(bbox_normalize_stds)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(bbox_normalize_inside_weights)

    def forward(self, gt_rois, rois, labels):

        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_rois)
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_rois)
        self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_rois)

        targets = bbox_transform_batch(rois[:, :, 1:5], gt_rois[:, :, :4])

        if self.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets)) / self.BBOX_NORMALIZE_STDS.expand_as(targets))

        labels = labels.expand(1, -1)
        batch_size = labels.size(0)
        rois_per_image = labels.size(1)
        clss = labels
        bbox_targets = targets.new(batch_size, rois_per_image, 4).zero_()
        bbox_inside_weights = targets.new(bbox_targets.size()).zero_()

        for b in range(batch_size):
            # assert clss[b].sum() > 0
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).view(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                bbox_targets[b, ind, :] = targets[b, ind, :]
                bbox_inside_weights[b, ind, :] = self.BBOX_INSIDE_WEIGHTS

        bbox_outside_weights = (bbox_inside_weights > 0).float()

        return bbox_targets, bbox_inside_weights, bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass