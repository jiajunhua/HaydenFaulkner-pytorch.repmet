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
# was rpn_target.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np

from ..bbox_transform import clip_boxes, bbox_overlaps_batch, bbox_transform_batch


class RPNTargetSampler(nn.Module):
    """
        A sampler to choose positive/negative samples from anchors

        rpn_batch_size : int
            Number of samples for RCNN targets

        clobber_positives : bool
            If an anchor statisfied by positive and negative conditions set to negative

        fg_fraction : float
            Max number of foreground examples, fg_rois_per_image=this*rpn_batch_size

        positive_overlap : float
            IOU >= thresh: positive example

        negative_overlap : float
            IOU < thresh: negative example
    """
    def __init__(self, rpn_batch_size, clobber_positives, fg_fraction, positive_overlap, negative_overlap):
        super(RPNTargetSampler, self).__init__()

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0  # default is 0

        self.RPN_CLOBBER_POSITIVES = clobber_positives
        self.RPN_POSITIVE_OVERLAP = positive_overlap
        self.RPN_NEGATIVE_OVERLAP = negative_overlap

        self.RPN_FG_FRACTION = fg_fraction

        self.RPN_BATCHSIZE = rpn_batch_size

    def forward(self, rpn_cls_score, gt_boxes, im_info, num_boxes, anchors):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors

        batch_size = gt_boxes.size(0)
        anchors = anchors[0]  # removing batch axis
        # filter out-of-image anchors
        img_width = int(im_info[0][1])
        img_height = int(im_info[0][0])
        keep = ((anchors[:, 0] >= -self._allowed_border) &
                (anchors[:, 1] >= -self._allowed_border) &
                (anchors[:, 2] < img_width + self._allowed_border) &
                (anchors[:, 3] < img_height + self._allowed_border))

        inds_inside = torch.nonzero(keep).view(-1)

        anchors = anchors[inds_inside, :]

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)

        # iou of anchors with gt_boxes
        overlaps = bbox_overlaps_batch(anchors, gt_boxes)

        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        gt_max_overlaps, _ = torch.max(overlaps, 1)

        if not self.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < self.RPN_NEGATIVE_OVERLAP] = 0

        gt_max_overlaps[gt_max_overlaps==0] = 1e-5
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)), 2)

        if torch.sum(keep) > 0:
            labels[keep>0] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= self.RPN_POSITIVE_OVERLAP] = 1

        if self.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < self.RPN_NEGATIVE_OVERLAP] = 0

        num_fg = int(self.RPN_FG_FRACTION * self.RPN_BATCHSIZE)

        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)

        for i in range(batch_size):
            # subsample positive labels if we have too many
            if sum_fg[i] > num_fg:
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                labels[i][disable_inds] = -1

#           num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]
            num_bg = self.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i]

            # subsample negative labels if we have too many
            if sum_bg[i] > num_bg:
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                #rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()

                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
                labels[i][disable_inds] = -1

        offset = torch.arange(0, batch_size)*gt_boxes.size(1)

        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)

        samples = labels
        matches = argmax_overlaps

        return samples, matches, inds_inside, anchors


class RPNTargetGenerator(nn.Module):
    """
        RPN target generator network.

        positive_weight : float
            Give the positive RPN examples weight of p * 1 / {num positives} and give negatives a weight of (1 - p)
            Set to -1.0 to use uniform example weighting

        bbox_inside_weights : tuple

    """

    def __init__(self, rpn_batch_size, positive_overlap, negative_overlap, fg_fraction, clobber_positives, n_base_anchors, positive_weight, bbox_inside_weights):
        super(RPNTargetGenerator, self).__init__()

        self.n_base_anchors = n_base_anchors

        self.RPN_POSITIVE_WEIGHT = positive_weight
        self.RPN_BBOX_INSIDE_WEIGHTS = bbox_inside_weights

        self._sampler = RPNTargetSampler(rpn_batch_size, clobber_positives, fg_fraction, positive_overlap, negative_overlap)

    def forward(self, rpn_cls_score, gt_boxes, im_info, num_boxes, anchors):

        # map of shape (..., H, W)
        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)

        batch_size = gt_boxes.size(0)

        total_anchors = anchors.shape[1]

        samples, matches, inds_inside, anchors = self._sampler(rpn_cls_score, gt_boxes, im_info, num_boxes, anchors)
        cls_targets = samples
        argmax_overlaps = matches

        bbox_inside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()
        bbox_outside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()

        bbox_targets = _compute_targets_batch(anchors, gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))

        # use a single value instead of 4 values for easy index.
        bbox_inside_weights[cls_targets==1] = self.RPN_BBOX_INSIDE_WEIGHTS[0]
        i = 0
        if self.RPN_POSITIVE_WEIGHT < 0:
            num_examples = torch.sum(cls_targets[i] >= 0)
            positive_weights = 1.0 / num_examples.item()
            negative_weights = 1.0 / num_examples.item()
        else:
            assert ((self.RPN_POSITIVE_WEIGHT > 0) &
                    (self.RPN_POSITIVE_WEIGHT < 1))

        bbox_outside_weights[cls_targets == 1] = positive_weights
        bbox_outside_weights[cls_targets == 0] = negative_weights

        cls_targets = _unmap(cls_targets, total_anchors, inds_inside, batch_size, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)

        cls_targets = cls_targets.view(batch_size, height, width, self.n_base_anchors).permute(0, 3, 1, 2).contiguous()
        cls_targets = cls_targets.view(batch_size, 1, self.n_base_anchors * height, width)

        bbox_targets = bbox_targets.view(batch_size, height, width, self.n_base_anchors*4).permute(0, 3, 1, 2).contiguous()

        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.view(batch_size, anchors_count, 1).expand(batch_size, anchors_count, 4)
        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4*self.n_base_anchors).permute(0, 3, 1, 2).contiguous()

        bbox_outside_weights = bbox_outside_weights.view(batch_size, anchors_count, 1).expand(batch_size, anchors_count, 4)
        bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4*self.n_base_anchors).permute(0, 3, 1, 2).contiguous()

        return cls_targets, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds,:] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])
