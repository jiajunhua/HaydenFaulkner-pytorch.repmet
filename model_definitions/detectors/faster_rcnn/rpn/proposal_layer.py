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
# --------------------------------------------------------

import torch
import torch.nn as nn

from ..bbox_transform import bbox_transform_inv, clip_boxes  #, clip_boxes_batch
from .generate_anchors import shift_anchor_bases

from roi_layers import nms


class RPNProposal(nn.Module):
    """
    RPNProposal takes RPN anchors, RPN prediction scores and box regression predictions.
    It will transform anchors, apply NMS to get clean foreground proposals.
    """

    def __init__(self,
                 anchor_bases,
                 stride,
                 pre_nms_top_n,
                 post_nms_top_n,
                 nms_thresh,
                 min_size):
        super(RPNProposal, self).__init__()

        self._stride = stride
        self._anchor_bases = anchor_bases
        self._num_anchors = self._anchor_bases.shape[0]  # 9

        self.pre_nms_top_n  = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh     = nms_thresh
        self.min_size       = min_size

    def forward(self, scores, bbox_deltas, im_info):

        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)


        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs
        scores = scores[:, self._num_anchors:, :, :]

        cfg_key = 'TRAIN' if self.training else 'TEST'
        pre_nms_topN  = self.pre_nms_top_n[cfg_key]
        post_nms_topN = self.post_nms_top_n[cfg_key]
        nms_thresh    = self.nms_thresh[cfg_key]
        min_size      = self.min_size[cfg_key]

        batch_size = bbox_deltas.size(0)

        anchors = shift_anchor_bases(self._anchor_bases, self._stride, (scores.size(2), scores.size(3)))
        anchors = torch.from_numpy(anchors).type_as(scores)

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)

        # Same story for the scores:
        scores = scores.permute(0, 2, 3, 1).contiguous()
        scores = scores.view(batch_size, -1)

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)  # box_decoder in gluoncv (l64 rpn/proposal.py)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info, batch_size)  # clipper in gluoncv (l68rpn/proposal.py)
        # proposals = clip_boxes_batch(proposals, im_info, batch_size)

        # assign the score to 0 if it's non keep.
        # keep = self._filter_boxes(proposals, min_size * im_info[:, 2])

        # trim keep index to make it equal over batch
        # keep_idx = torch.cat(tuple(keep_idx), 0)

        # scores_keep = scores.view(-1)[keep_idx].view(batch_size, trim_size)
        # proposals_keep = proposals.view(-1, 4)[keep_idx, :].contiguous().view(batch_size, trim_size, 4)

        # _, order = torch.sort(scores_keep, 1, True)

        scores_keep = scores
        proposals_keep = proposals
        _, order = torch.sort(scores_keep, 1, True)

        rpn_scores = scores.new(batch_size, post_nms_topN, 1).zero_()
        rpn_bbox = scores.new(batch_size, post_nms_topN, 5).zero_()
        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]

            if 0 < pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]

            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1,1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)
            keep_idx_i = nms(proposals_single, scores_single.squeeze(1), nms_thresh)
            keep_idx_i = keep_idx_i.long().view(-1)

            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            num_proposal = proposals_single.size(0)
            rpn_scores[i, :, 0] = i  # first item is the batch index
            rpn_scores[i, :num_proposal, :] = scores_single

            # padding 0 at the end.
            # num_proposal = proposals_single.size(0)
            rpn_bbox[i, :, 0] = i
            rpn_bbox[i, :num_proposal, 1:] = proposals_single

        return rpn_scores, rpn_bbox, anchors

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size.view(-1, 1).expand_as(ws)) & (hs >= min_size.view(-1, 1).expand_as(hs)))
        return keep
